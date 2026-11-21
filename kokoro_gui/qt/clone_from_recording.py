"""Clone a speaker from their own recording, to fix a flub (plan 33 part A).

Right-click a selection inside an imported recording and pick "Clone this
speaker from the selection...". `clone_span` finds the selection's timed
words (one recording file, `imported` words) and their first and last word
times. `CloneDialog` shows the selected text as an editable transcript, the
length (5 to 30 s), a Play button, a name and the consent checkbox (grill
PG9). On Clone `save_clone` slices that stretch into a wav, saves it with the
transcript as an Audio8 reference in the global store
(`ReferenceStore.save_reference`), and gives it a speaker: a variant named
after the reference on the recording's own character when that character
is already on Audio8, else a new local Audio8 character whose voice is the
reference.

The user then types the correction inside the recording. The typed text
splits it and is untagged (`Document.replace_text`), so they select it and
give it the clone (Characters menu, plus the Transcript dock's variant on
the variant route). The final message says which.

Consent (PG9): a checkbox the user ticks each time before Clone is enabled.
Nothing about it is stored: not with the reference, not in the `.tbaw`, no
badge on the character. Its "Don't ask again" box turns the program setting
`clone_consent_confirm` off (`config_qt.json`, Options > Confirm consent
before cloning turns it back on); the project never holds it.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Optional

import playback
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPlainTextEdit, QPushButton, QVBoxLayout,
)

from kokoro_gui.daw import imported
from kokoro_gui.daw.models import DEFAULT_HIGHLIGHT_PALETTE, Character
from kokoro_gui.engines import registry as engine_registry

ENGINE_ID = "audio8"
MIN_S = 5.0
MAX_S = 30.0
# The program setting (config_qt.json) that asks for consent each time.
CONSENT_SETTING = "clone_consent_confirm"
CONSENT_TEXT = "I have this speaker's permission to clone their voice."
MENU_TEXT = "Clone this speaker from the selection..."


@dataclass
class CloneSpan:
    """What a selection inside a recording covers: the clip of its first
    timed word and that clip's character, the recording (`source`, a key of
    `Document.sources`, and its file) and the first and last word times."""

    clip_id: str
    character_id: Optional[str]
    source: str
    path: str
    start_s: float
    end_s: float
    text: str

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


def clone_span(document, start: int, end: int) -> Optional[CloneSpan]:
    """The `CloneSpan` of the text `[start, end)`, or None when no timed
    recording word is in it, its words come from more than one recording
    file, or that file is not on disk."""
    if end <= start:
        return None
    rows = []
    for run, r_start, r_end in document._iter_runs_with_offsets():
        if r_end <= start or r_start >= end or not run.words or run.clip_id is None:
            continue
        clip = document.get_clip(run.clip_id)
        if clip is None or not imported.is_recording_clip(clip):
            continue
        for word in run.words:
            try:
                w_start, w_end = r_start + int(word[0]), r_start + int(word[1])
                source, start_s, end_s = str(word[2]), float(word[3]), float(word[4])
            except (TypeError, ValueError, IndexError):
                continue
            if w_end > start and w_start < end:
                rows.append((w_start, clip, source, start_s, end_s))
    if not rows:
        return None
    rows.sort(key=lambda row: row[0])
    sources = {row[2] for row in rows}
    if len(sources) != 1:
        return None
    source = rows[0][2]
    path = document.source_path(source)
    if not path or not os.path.isfile(path):
        return None
    first_clip = rows[0][1]
    return CloneSpan(clip_id=first_clip.id, character_id=first_clip.character_id, source=source, path=path,
                     start_s=min(row[3] for row in rows), end_s=max(row[4] for row in rows),
                     text=" ".join(document.text[start:end].split()))


def duration_problem(duration_s: float) -> Optional[str]:
    """Why a stretch this long can't be a reference, or None."""
    if duration_s < MIN_S:
        return f"The selection is {duration_s:.1f} s. Select at least {MIN_S:.0f} s of speech."
    if duration_s > MAX_S:
        return f"The selection is {duration_s:.1f} s. Select at most {MAX_S:.0f} s of speech."
    return None


def reference_store():
    """Audio8's reference store, or None when Audio8 isn't installed."""
    if ENGINE_ID not in engine_registry.list_engines():
        return None
    return getattr(engine_registry.get_factory(ENGINE_ID), "voice_store", None)


def as_variant(character) -> bool:
    """True when the clone becomes a variant of `character` (already on
    Audio8) rather than a new character."""
    return character is not None and character.backend_id == ENGINE_ID


def default_name(document, character) -> str:
    """"<character> (clone)", numbered when the project already has a
    character or variant of that name."""
    base = f"{character.name if character is not None else 'Speaker'} (clone)"
    taken = {c.name for c in document.characters} | set((character.variants or {}) if character else ())
    name, n = base, 2
    while name in taken:
        name, n = f"{base} {n}", n + 1
    return name


def name_problem(name: str, store, document, character, project_dir: Optional[str] = None) -> Optional[str]:
    """Why `name` can't name the clone, or None. A name already used by a
    reference (here or in the library), a character or a variant is
    refused, so no other voice changes under it."""
    name = name.strip()
    if not name:
        return "Enter a name."
    safe = store._safe_name(name)
    if not safe:
        return "That name can't be a file name."
    if store.find_wav(safe, project_dir) is not None:
        return f"A reference named {safe} already exists."
    if as_variant(character):
        if name in (character.variants or {}):
            return f"{character.name} already has a variant named {name}."
    elif any(c.name == name for c in document.characters):
        return f"A character named {name} already exists."
    return None


def slice_to_wav(path: str, start_s: float, end_s: float) -> str:
    """`[start_s, end_s]` of `path` written to a new temporary wav, at the
    file's own rate and channels. The caller removes it."""
    import soundfile as sf

    with sf.SoundFile(path) as f:
        rate = f.samplerate
        start = max(0, min(int(round(start_s * rate)), f.frames))
        f.seek(start)
        data = f.read(max(0, int(round(end_s * rate)) - start), dtype="float32")
    fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="kokorogui-clone-")
    os.close(fd)
    sf.write(tmp, data, rate)
    return tmp


@dataclass
class CloneResult:
    """What `save_clone` made: the reference's name and wav, and the speaker
    of a correction, `character` plus `variant` (None for a new character)."""

    reference: str
    wav_path: str
    character: Character
    variant: Optional[str]

    def how_to_use(self) -> str:
        """The final message's instructions."""
        if self.variant is None:
            return (f"Type your correction inside the recording, select it and pick {self.character.name} "
                    "from the Characters menu. Then generate it as usual.")
        return (f"Type your correction inside the recording, select it and pick {self.character.name} from "
                f"the Characters menu, then pick the variant {self.variant} in the Transcript dock. "
                "Then generate it as usual.")


def save_clone(app, document, span: CloneSpan, name: str, transcript: str, store=None) -> CloneResult:
    """Slices `span` into a wav, saves it with `transcript` as the Audio8
    reference `name` in the global store and gives it a speaker (see the
    module docstring). Raises `ValueError` for a refused name or
    transcript, `OSError` when the file can't be read or written."""
    store = store if store is not None else reference_store()
    if store is None:
        raise ValueError("Audio8 is not installed.")
    character = document.get_character(span.character_id)
    problem = name_problem(name, store, document, character, getattr(app, "project_dir", None))
    if problem:
        raise ValueError(problem)
    if not transcript.strip():
        raise ValueError("Enter the transcript of the selection.")
    tmp = slice_to_wav(span.path, span.start_s, span.end_s)
    try:
        wav_path = store.save_reference(name.strip(), tmp, transcript)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    reference = os.path.splitext(os.path.basename(wav_path))[0]
    if as_variant(character):
        character.variants = {**(character.variants or {}), name.strip(): reference}
        app.commit_character_edit(character)
        result = CloneResult(reference, wav_path, character, name.strip())
    else:
        color = DEFAULT_HIGHLIGHT_PALETTE[len(document.characters) % len(DEFAULT_HIGHLIGHT_PALETTE)]
        clone = Character.from_preset_dict(name.strip(), {"voice": reference}, highlight_color=color,
                                           backend_id=ENGINE_ID)
        document.characters.append(clone)
        app.commit_character_edit(clone, write_through=False)
        result = CloneResult(reference, wav_path, clone, None)
    # A new voice file: the voice lists and the file checks start over.
    app.refresh_voice_choices()
    return result


class CloneDialog(QDialog):
    """The clone dialog. `values()` is `(name, transcript, dont_ask)`."""

    def __init__(self, span: CloneSpan, name: str, problem_of_name, ask_consent: bool = True, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Clone this speaker")
        self._span = span
        self._problem_of_name = problem_of_name
        self.ask_consent = ask_consent
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("The selection becomes an Audio8 reference: its audio and what is said in it."))

        form = QFormLayout()
        self.transcript_edit = QPlainTextEdit(span.text)
        self.transcript_edit.setMaximumHeight(100)
        form.addRow("Transcript:", self.transcript_edit)
        length_row = QHBoxLayout()
        self.duration_label = QLabel(f"{span.duration_s:.1f} s")
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.play)
        length_row.addWidget(self.duration_label)
        length_row.addWidget(self.play_button)
        length_row.addStretch(1)
        form.addRow("Length:", length_row)
        self.name_edit = QLineEdit(name)
        form.addRow("Name:", self.name_edit)
        layout.addLayout(form)

        self.consent_check = QCheckBox(CONSENT_TEXT)
        self.dont_ask_check = QCheckBox("Don't ask again")
        self.dont_ask_check.setToolTip("Stored in the program's settings, never in the project. "
                                       "Options > Confirm consent before cloning asks again.")
        self.consent_check.setVisible(ask_consent)
        self.dont_ask_check.setVisible(ask_consent)
        layout.addWidget(self.consent_check)
        layout.addWidget(self.dont_ask_check)

        self.problem_label = QLabel()
        self.problem_label.setWordWrap(True)
        layout.addWidget(self.problem_label)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
                                        self)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Clone")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.transcript_edit.textChanged.connect(self._sync)
        self.name_edit.textChanged.connect(self._sync)
        self.consent_check.toggled.connect(self._sync)
        self.consent_check.toggled.connect(self.dont_ask_check.setEnabled)
        self.dont_ask_check.setEnabled(False)
        self._sync()

    @property
    def ok_button(self):
        return self.buttons.button(QDialogButtonBox.StandardButton.Ok)

    def problem(self) -> Optional[str]:
        """Why Clone is off, or None."""
        problem = duration_problem(self._span.duration_s)
        if problem:
            return problem
        if not self.transcript_edit.toPlainText().strip():
            return "Enter the transcript of the selection."
        problem = self._problem_of_name(self.name_edit.text())
        if problem:
            return problem
        if self.ask_consent and not self.consent_check.isChecked():
            return "Tick the box to confirm you have the speaker's permission."
        return None

    def _sync(self) -> None:
        problem = self.problem()
        self.problem_label.setText(problem or "")
        self.ok_button.setEnabled(problem is None)

    def play(self) -> None:
        if self._span.end_s > self._span.start_s:
            playback.play_range(self._span.path, self._span.start_s, self._span.end_s)

    def values(self) -> tuple:
        return (self.name_edit.text().strip(), self.transcript_edit.toPlainText().strip(),
                self.ask_consent and self.dont_ask_check.isChecked())


def ask(dialog: CloneDialog) -> bool:
    """Runs the dialog; True on Clone. Its own function so tests answer it
    without a modal."""
    return dialog.exec() == QDialog.DialogCode.Accepted


def clone_from_selection(app, start: int, end: int) -> Optional[CloneResult]:
    """The whole flow for the text `[start, end)` of the focus project:
    the dialog, then `save_clone` and the final message. None when it is
    refused, cancelled or fails (the status line says why)."""
    if app.is_busy():
        app.set_status("Finish or cancel the current job before cloning a voice.", "warning")
        return None
    store = reference_store()
    if store is None:
        app.set_status("Cloning a speaker needs Audio8, which is not installed.", "warning")
        return None
    document = app.document
    span = clone_span(document, start, end)
    if span is None:
        app.set_status("Select words inside one imported recording to clone its speaker.", "warning")
        return None
    character = document.get_character(span.character_id)
    project_dir = getattr(app, "project_dir", None)
    dialog = CloneDialog(span, default_name(document, character),
                         lambda name: name_problem(name, store, document, character, project_dir),
                         ask_consent=bool(app.settings.get(CONSENT_SETTING, True)), parent=app)
    if not ask(dialog):
        return None
    name, transcript, dont_ask = dialog.values()
    if dont_ask:
        app.settings[CONSENT_SETTING] = False
        action = getattr(app, "clone_consent_action", None)
        if action is not None:
            action.setChecked(False)
        app.schedule_save()
    try:
        result = save_clone(app, document, span, name, transcript, store=store)
    except (OSError, ValueError, RuntimeError) as e:
        QMessageBox.warning(app, "Clone failed", str(e))
        app.set_status(f"Clone failed: {e}", "error")
        return None
    app.set_status(f"Saved the reference {result.reference}.", "success")
    QMessageBox.information(app, "Speaker cloned", result.how_to_use())
    return result
