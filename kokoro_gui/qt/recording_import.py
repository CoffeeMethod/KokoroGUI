"""File > Import Audio's recording path (phase 5 P3, grill Q19/Q24/Q25/Q32/Q33).

`ImportAudioDialog` is the first page: what the file is (a music bed, P2's
path, or a recording to edit as text) and, for a recording, where its
transcript comes from: Whisper, or a caption file (SRT, VTT, ASS) with an
opt-in "Refine word timing with Whisper".

Every clip-to-be is a `PendingRow`: its text, its `Run.words`, and what was
heard (the words the text is aligned against when the user corrects it).
`rows_from_asr_words` cuts a Whisper pass into rows
(`imported.group_asr_words`: a sentence end, a pause over 0.7 s or 15 s of
speech ends one); `rows_from_cues` makes one row per caption cue with
proportional word times (`imported.words_from_cue`); `refine_row` replaces
a cue's proportional times with Whisper's on the cue's slice
(`slice_words`), keeping the text.

`RecordingReviewDialog` (grill Q19) lists the rows with a play button each
(the row's slice of the file, through `playback.play_range`) and the text
editable. A corrected line is realigned to what was heard
(`imported.realign_words`), so a fixed spelling keeps its times. The
character is picked once for the whole recording, from the characters
whose engine can clone a voice (Q24/Q25), or "Unknown speaker": a new local
character with no voice. A caption file that names speakers has already
been through the speaker mapping, so the picker is hidden.

The app (`QtTTSApp.import_recording`) runs Whisper on a worker thread and
commits the review as one `undo.ImportRecordingCommand`.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Optional

import playback
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QRadioButton, QScrollArea, QToolButton, QVBoxLayout, QWidget,
)

from kokoro_gui.daw import imported, subtitles

MUSIC_BED = "bed"
RECORDING = "recording"
WHISPER = "whisper"
CAPTIONS = "captions"
# The review dialog's character choice for "a voice nobody has cloned".
UNKNOWN_SPEAKER = "__unknown__"
UNKNOWN_SPEAKER_NAME = "Unknown speaker"


@dataclass
class PendingRow:
    """One clip-to-be. `words` are `Run.words` for `text`; `heard` is
    `[(word, start_s, end_s)]` the text is realigned against when edited;
    `span` the `(start_s, end_s)` of the file it covers (a caption cue's
    times, else its words'); `character_id` a caption speaker's mapped
    character, or None for the one picked in the review."""
    text: str
    words: list
    heard: list
    span: tuple
    character_id: Optional[str] = None


@dataclass
class RecordingJob:
    """What an import carries from the first page to the commit: the copied
    file, its `Document.sources` name and entry, the rows, and the
    characters a caption speaker mapping made (`mapped` when it ran)."""
    path: str
    name: str
    source: str
    entry: dict
    transcript: str = WHISPER
    refine: bool = False
    rows: list = field(default_factory=list)
    new_characters: list = field(default_factory=list)
    mapped: bool = False


def _span_of(words: list) -> tuple:
    return (float(words[0][3]), float(words[-1][4])) if words else (0.0, 0.0)


def rows_from_asr_words(words: list, source: str) -> list:
    """`PendingRow`s for a Whisper pass over the whole file, one per
    `imported.group_asr_words` group."""
    rows = []
    for group in imported.group_asr_words(words):
        text, run_words = imported.run_from_asr_words(group, source)
        if text and run_words:
            rows.append(PendingRow(text=text, words=run_words, heard=list(group), span=_span_of(run_words)))
    return rows


def rows_from_cues(cues, source: str, character_ids: Optional[dict] = None) -> list:
    """One `PendingRow` per caption cue: the cue's text on one line, words
    spread over the cue in proportion to their length. `character_ids`
    maps a cue's speaker (or `""` for none) to a character id."""
    rows = []
    for cue in cues:
        text = " ".join(cue.text.split())
        words = imported.words_from_cue(text, cue.start_s, cue.end_s, source)
        if not text or not words:
            continue
        character_id = character_ids.get(cue.speaker or "") if character_ids else None
        rows.append(PendingRow(text=text, words=words, heard=imported.heard_words(text, words),
                               span=(float(cue.start_s), float(cue.end_s)), character_id=character_id))
    return rows


def slice_words(path: str, start_s: float, end_s: float, transcribe: Callable[[str], list]) -> list:
    """What `transcribe(wav_path)` hears in `[start_s, end_s]` of `path`,
    with times into the whole file. The slice goes through a temporary wav
    file, removed afterwards."""
    import soundfile as sf

    with sf.SoundFile(path) as f:
        rate = f.samplerate
        start = max(0, min(int(round(start_s * rate)), f.frames))
        f.seek(start)
        data = f.read(max(0, int(round(end_s * rate)) - start), dtype="float32")
    fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="kokorogui-slice-")
    os.close(fd)
    try:
        sf.write(tmp, data, rate)
        heard = transcribe(tmp)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    out = []
    for item in heard or []:
        try:
            out.append((str(item[0]), float(item[1]) + start_s, float(item[2]) + start_s))
        except (TypeError, ValueError, IndexError):
            continue
    return out


def refine_row(row: PendingRow, path: str, source: str, transcribe: Callable[[str], list]) -> bool:
    """Replaces a caption row's proportional word times with Whisper's on
    its slice of the file, keeping its text (grill Q33). False, and the row
    left as it was, when Whisper heard nothing there."""
    heard = slice_words(path, row.span[0], row.span[1], transcribe)
    words = imported.realign_words(row.text, heard, source) if heard else []
    if not words:
        return False
    row.words, row.heard = words, heard
    return True


class ImportAudioDialog(QDialog):
    """File > Import Audio's first page. `choice()` is `{"kind":
    MUSIC_BED | RECORDING, "transcript": WHISPER | CAPTIONS,
    "caption_path": str, "refine": bool}`."""

    def __init__(self, file_name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import audio")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"Import {file_name} as:"))
        self.bed_radio = QRadioButton("Music bed (plays under the speech)")
        self.recording_radio = QRadioButton("Recording to edit as text")
        self.bed_radio.setChecked(True)
        kinds = QButtonGroup(self)
        kinds.addButton(self.bed_radio)
        kinds.addButton(self.recording_radio)
        layout.addWidget(self.bed_radio)
        layout.addWidget(self.recording_radio)

        self.transcript_box = QGroupBox("Transcript")
        box = QVBoxLayout(self.transcript_box)
        self.whisper_radio = QRadioButton("Transcribe with Whisper")
        self.captions_radio = QRadioButton("Caption file (SRT, VTT, ASS)")
        self.whisper_radio.setChecked(True)
        sources = QButtonGroup(self)
        sources.addButton(self.whisper_radio)
        sources.addButton(self.captions_radio)
        box.addWidget(self.whisper_radio)
        box.addWidget(self.captions_radio)
        row = QHBoxLayout()
        self.caption_edit = QLineEdit()
        self.caption_edit.setPlaceholderText("Caption file")
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self._browse)
        row.addWidget(self.caption_edit)
        row.addWidget(self.browse_button)
        box.addLayout(row)
        self.refine_check = QCheckBox("Refine word timing with Whisper")
        box.addWidget(self.refine_check)
        layout.addWidget(self.transcript_box)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
                                        self)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Next")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        for widget in (self.bed_radio, self.recording_radio, self.whisper_radio, self.captions_radio):
            widget.toggled.connect(self._sync)
        self.caption_edit.textChanged.connect(self._sync)
        self._sync()

    def _browse(self) -> None:
        patterns = " ".join("*" + ext for ext in subtitles.SUBTITLE_EXTENSIONS)
        path, _ = QFileDialog.getOpenFileName(self, "Caption file", filter=f"Captions ({patterns})")
        if path:
            self.caption_edit.setText(path)
            self.captions_radio.setChecked(True)

    def _sync(self) -> None:
        recording = self.recording_radio.isChecked()
        self.transcript_box.setEnabled(recording)
        captions = self.captions_radio.isChecked()
        self.caption_edit.setEnabled(captions)
        self.browse_button.setEnabled(captions)
        self.refine_check.setEnabled(captions)
        ready = not recording or not captions or bool(self.caption_edit.text().strip())
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(ready)

    def choice(self) -> dict:
        captions = self.captions_radio.isChecked()
        return {
            "kind": RECORDING if self.recording_radio.isChecked() else MUSIC_BED,
            "transcript": CAPTIONS if captions else WHISPER,
            "caption_path": self.caption_edit.text().strip() if captions else "",
            "refine": captions and self.refine_check.isChecked(),
        }

    @classmethod
    def ask(cls, file_name: str, parent=None) -> Optional[dict]:
        """Runs the page modally: `choice()`, or None on Cancel."""
        dialog = cls(file_name, parent)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return dialog.choice()


class RecordingReviewDialog(QDialog):
    """The review before commit (grill Q19): one editable line per
    clip-to-be with a play button, and the character picker. `characters`
    is `[(character_id, name)]` of the cloning-capable characters, or None
    when a caption speaker mapping already chose them (the picker is
    hidden). `rows()` is what `ImportRecordingCommand` takes."""

    def __init__(self, rows: list, source: str, audio_path: str, characters: Optional[list] = None,
                 parent=None, title: str = "Review the transcript"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(720, 520)
        self._rows = list(rows)
        self._source = source
        self._audio_path = audio_path
        self._edits: list = []
        self.play_buttons: list = []

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Check the text of each clip. A corrected word keeps its place in the recording."))
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        body = QWidget()
        lines = QVBoxLayout(body)
        for index, row in enumerate(self._rows):
            line = QHBoxLayout()
            button = QToolButton()
            button.setText("▶")
            button.setToolTip(f"Play {row.span[0]:.1f}s - {row.span[1]:.1f}s")
            button.clicked.connect(lambda _checked=False, i=index: self.play(i))
            edit = QLineEdit(row.text)
            line.addWidget(button)
            line.addWidget(edit, 1)
            lines.addLayout(line)
            self.play_buttons.append(button)
            self._edits.append(edit)
        lines.addStretch(1)
        scroll.setWidget(body)
        layout.addWidget(scroll, 1)

        self.character_combo: Optional[QComboBox] = None
        if characters is not None:
            form = QFormLayout()
            self.character_combo = QComboBox()
            for character_id, name in characters:
                self.character_combo.addItem(name, character_id)
            self.character_combo.addItem(f"{UNKNOWN_SPEAKER_NAME} (new character, no voice)", UNKNOWN_SPEAKER)
            form.addRow("Speaker", self.character_combo)
            layout.addLayout(form)
        else:
            layout.addWidget(QLabel("Speakers: as mapped from the caption file."))

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Import")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def row_count(self) -> int:
        return len(self._rows)

    def text(self, index: int) -> str:
        return " ".join(self._edits[index].text().split())

    def set_text(self, index: int, text: str) -> None:
        self._edits[index].setText(text)

    def words(self, index: int) -> list:
        """`Run.words` for the line as it reads now: the row's own when
        unchanged, else realigned to what was heard."""
        row, text = self._rows[index], self.text(index)
        if text == row.text:
            return [list(w) for w in row.words]
        return imported.realign_words(text, row.heard, self._source)

    def play(self, index: int) -> None:
        """Plays the line's slice of the recording."""
        words = self.words(index)
        start_s, end_s = _span_of(words) if words else self._rows[index].span
        if end_s > start_s:
            playback.play_range(self._audio_path, start_s, end_s)

    def set_character(self, choice: str) -> bool:
        if self.character_combo is None:
            return False
        index = self.character_combo.findData(choice)
        if index < 0:
            return False
        self.character_combo.setCurrentIndex(index)
        return True

    def character_choice(self) -> Optional[str]:
        """A character id, `UNKNOWN_SPEAKER`, or None when the picker is
        hidden."""
        return self.character_combo.currentData() if self.character_combo is not None else None

    def rows(self) -> list:
        """`[{"text", "words", "character_id"}]` for the lines left with
        text, in order."""
        out = []
        for index, row in enumerate(self._rows):
            text = self.text(index)
            if text:
                out.append({"text": text, "words": self.words(index), "character_id": row.character_id})
        return out
