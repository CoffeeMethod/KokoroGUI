"""Speaker mapping (phase 5 D2): which character voices each speaker a
subtitle or caption file names.

`SpeakerMappingDialog(speakers, characters)` shows one row per speaker with
a choice of the narrator, a new character named after the speaker, or any
existing character, and `mapping()` returns `{speaker: choice}`: a
character id, `NARRATOR` or `NEW_CHARACTER`. `NO_SPEAKER` ("") is the row
for cues without a speaker tag; it offers the narrator and the existing
characters. `SpeakerMappingDialog.ask(...)` runs it modally and returns the
mapping, or None on Cancel.

`resolve_mapping` turns a mapping into character ids plus the new
`Character` records to add, so File > Import Subtitles and the caption path
of Import Recording (P3) share it. The narrator is the document's
"Default" character when it has one (the character a new document seeds
from the app settings), else its first character; with no characters at
all the narrator is a new "Default".
"""
from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QVBoxLayout

from kokoro_gui.daw.migration import DEFAULT_CHARACTER_NAME

NARRATOR = "__narrator__"
NEW_CHARACTER = "__new__"
NO_SPEAKER = ""
NO_SPEAKER_LABEL = "(no speaker)"


def speaker_rows(cues) -> list:
    """The dialog's rows for `cues` (anything with a `speaker`): each
    speaker once, in the order they first speak, then `NO_SPEAKER` when
    some cues have none. Empty when no cue names a speaker: there is
    nothing to ask."""
    rows: list = []
    untagged = False
    for cue in cues:
        speaker = (cue.speaker or "").strip()
        if not speaker:
            untagged = True
        elif speaker not in rows:
            rows.append(speaker)
    if rows and untagged:
        rows.append(NO_SPEAKER)
    return rows


def narrator_character(characters):
    """The character "Narrator" means: the one named "Default", else the
    first, else None."""
    characters = list(characters)
    default = next((c for c in characters if c.name.strip().lower() == DEFAULT_CHARACTER_NAME.lower()), None)
    return default or (characters[0] if characters else None)


def default_choice(speaker: str, characters) -> str:
    """What a row starts on: the character whose name matches the speaker
    (ignoring case), else a new character, and the narrator for
    `NO_SPEAKER`."""
    if speaker == NO_SPEAKER:
        return NARRATOR
    target = speaker.strip().lower()
    match = next((c for c in characters if c.name.strip().lower() == target), None)
    return match.id if match is not None else NEW_CHARACTER


def resolve_mapping(mapping: dict, characters, make_character: Callable) -> tuple:
    """`(character_ids, new_characters)`: `character_ids` maps every key of
    `mapping` to a character id; `new_characters` are the records to add
    to the document for `NEW_CHARACTER` rows (and for the narrator when
    `characters` is empty). `make_character(name, index)` builds one;
    `index` counts the characters made so far, for picking a color. A new
    character whose name is taken gets a number after it."""
    characters = list(characters)
    names = {c.name for c in characters}
    known = {c.id for c in characters}
    new_characters: list = []

    def _make(name: str):
        base = name.strip() or "Speaker"
        unique, n = base, 2
        while unique in names:
            unique = f"{base} {n}"
            n += 1
        names.add(unique)
        character = make_character(unique, len(characters) + len(new_characters))
        new_characters.append(character)
        return character

    narrator: list = []

    def _narrator_id() -> str:
        if not narrator:
            existing = narrator_character(characters)
            narrator.append(existing if existing is not None else _make(DEFAULT_CHARACTER_NAME))
        return narrator[0].id

    character_ids = {}
    for speaker, choice in mapping.items():
        if choice == NEW_CHARACTER and speaker != NO_SPEAKER:
            character_ids[speaker] = _make(speaker).id
        elif choice in known:
            character_ids[speaker] = choice
        else:
            character_ids[speaker] = _narrator_id()
    return character_ids, new_characters


class SpeakerMappingDialog(QDialog):
    """One combo per speaker. `speakers` are the names in the file (the
    order rows appear in), `NO_SPEAKER` for untagged cues; `characters`
    the document's `Character` records. Nothing here touches a document."""

    def __init__(self, speakers, characters, parent=None, title: str = "Map speakers to characters"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self._characters = list(characters)
        self._combos: dict = {}

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Pick the character that voices each speaker in the file."))
        form = QFormLayout()
        layout.addLayout(form)
        narrator = narrator_character(self._characters)
        narrator_label = f"Narrator ({narrator.name})" if narrator is not None else \
            f"Narrator (new {DEFAULT_CHARACTER_NAME})"
        for speaker in speakers:
            if speaker in self._combos:
                continue
            combo = QComboBox(self)
            combo.addItem(narrator_label, NARRATOR)
            if speaker != NO_SPEAKER:
                combo.addItem(f"New character: {speaker}", NEW_CHARACTER)
            if self._characters:
                combo.insertSeparator(combo.count())
            for character in self._characters:
                combo.addItem(character.name, character.id)
            self._combos[speaker] = combo
            self.set_choice(speaker, default_choice(speaker, self._characters))
            form.addRow(NO_SPEAKER_LABEL if speaker == NO_SPEAKER else speaker, combo)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Import")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def speakers(self) -> list:
        return list(self._combos)

    def combo_for(self, speaker: str) -> Optional[QComboBox]:
        return self._combos.get(speaker)

    def set_choice(self, speaker: str, choice: str) -> bool:
        """Selects `choice` (a character id, `NARRATOR` or `NEW_CHARACTER`)
        in `speaker`'s row; False when the row doesn't offer it."""
        combo = self._combos.get(speaker)
        if combo is None:
            return False
        index = combo.findData(choice)
        if index < 0:
            return False
        combo.setCurrentIndex(index)
        return True

    def mapping(self) -> dict:
        """`{speaker: character id | NARRATOR | NEW_CHARACTER}`."""
        return {speaker: combo.currentData() for speaker, combo in self._combos.items()}

    @classmethod
    def ask(cls, speakers, characters, parent=None, **kwargs) -> Optional[dict]:
        """Runs the dialog modally: the mapping, or None when cancelled."""
        dialog = cls(speakers, characters, parent, **kwargs)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return dialog.mapping()
