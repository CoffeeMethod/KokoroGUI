"""Options > Spellcheck (plan 22, part B): the wave under words the character's
dictionary lacks, per clip language, no wave inside a tag, and the dictionary
`app.spell_dictionary_for` builds. The word lists are stubbed except where a
test says it needs the real one."""
import pytest
from PySide6.QtGui import QTextCharFormat, QTextCursor

from kokoro_gui.daw.models import Character
from kokoro_gui.daw.spell import Dictionary

WAVE = QTextCharFormat.UnderlineStyle.SpellCheckUnderline
ENGLISH = {"the", "dog", "barked", "at", "a", "cat"}
SPANISH = {"el", "perro", "ladro"}


def _type(editor, text):
    cursor = editor.textCursor()
    cursor.select(QTextCursor.SelectionType.Document)
    cursor.insertText(text)


def _waved(editor, text):
    """The words of `text` (the editor's one line) that carry the wave."""
    block = editor.document().begin()
    offsets = {i for fmt_range in block.layout().formats()
               if fmt_range.format.underlineStyle() == WAVE
               for i in range(fmt_range.start, fmt_range.start + fmt_range.length)}
    return [word for word in text.split() if text.index(word) in offsets]


def _stub_dictionaries(qt_app, by_character_name):
    """`spell_dictionary_for` answers a stub list per character name."""
    def provider(character=None):
        character = character or qt_app.active_character()
        words = by_character_name.get(character.name if character is not None else None)
        return Dictionary("en", checker=words) if words is not None else None

    qt_app.spell_dictionary_for = provider


def _one_clip(qt_app, text):
    _type(qt_app.editor, text)
    alice = qt_app.document.characters[0]
    qt_app.document.assign_character_to_range(0, len(text), alice.id)
    return alice


def test_spellcheck_is_its_own_options_entry_and_off_by_default(qt_app):
    assert qt_app.spellcheck_action in qt_app.options_menu.actions()
    assert qt_app.spellcheck_action.isCheckable()
    assert qt_app.settings["spellcheck"] is False and not qt_app.spellcheck_action.isChecked()
    assert qt_app.spellcheck_dictionary() is None


def test_an_unknown_word_gets_the_wave_and_a_known_one_does_not(qt_app):
    text = "The dog barked at a qwxz cat"
    alice = _one_clip(qt_app, text)
    _stub_dictionaries(qt_app, {alice.name: ENGLISH})
    qt_app.editor.rehighlight()
    assert _waved(qt_app.editor, text) == []

    qt_app.spellcheck_action.setChecked(True)
    assert qt_app.settings["spellcheck"] is True
    assert _waved(qt_app.editor, text) == ["qwxz"]


def test_turning_it_off_removes_the_wave(qt_app):
    text = "The dog barked at a qwxz cat"
    alice = _one_clip(qt_app, text)
    _stub_dictionaries(qt_app, {alice.name: ENGLISH})
    qt_app.spellcheck_action.setChecked(True)
    assert _waved(qt_app.editor, text) == ["qwxz"]
    qt_app.spellcheck_action.setChecked(False)
    assert qt_app.settings["spellcheck"] is False
    assert _waved(qt_app.editor, text) == []


def test_typing_a_new_unknown_word_is_underlined(qt_app):
    alice = _one_clip(qt_app, "The dog")
    _stub_dictionaries(qt_app, {alice.name: ENGLISH})
    qt_app.spellcheck_action.setChecked(True)
    assert _waved(qt_app.editor, "The dog") == []
    cursor = qt_app.editor.textCursor()
    cursor.movePosition(QTextCursor.MoveOperation.End)
    cursor.insertText(" barkd")
    assert _waved(qt_app.editor, "The dog barkd") == ["barkd"]


def test_tags_and_pause_markers_are_not_underlined(qt_app):
    text = "[Zorblax:Radio]: the dog [pause:1.5] barked"
    alice = _one_clip(qt_app, text)
    _stub_dictionaries(qt_app, {alice.name: ENGLISH})
    qt_app.spellcheck_action.setChecked(True)
    assert _waved(qt_app.editor, text) == []


def test_each_clip_is_checked_in_its_characters_language(qt_app):
    text = "the dog el perro"
    first = qt_app.document.characters[0]
    second = Character(name="Rosa")
    qt_app.document.characters.append(second)
    _type(qt_app.editor, text)
    qt_app.document.assign_character_to_range(0, 8, first.id)
    qt_app.document.assign_character_to_range(8, len(text), second.id)
    _stub_dictionaries(qt_app, {first.name: ENGLISH, second.name: SPANISH})
    qt_app.spellcheck_action.setChecked(True)
    assert _waved(qt_app.editor, text) == []  # each half is known to its own list

    _stub_dictionaries(qt_app, {first.name: SPANISH, second.name: ENGLISH})
    qt_app.editor.rehighlight()
    assert _waved(qt_app.editor, text) == ["the", "dog", "el", "perro"]


def test_text_outside_a_clip_is_checked_as_the_active_character(qt_app):
    text = "The dog qwxz"
    _type(qt_app.editor, text)  # no clip covers it
    alice = qt_app.document.characters[0]
    _stub_dictionaries(qt_app, {alice.name: ENGLISH})
    qt_app.spellcheck_action.setChecked(True)
    assert _waved(qt_app.editor, text) == ["qwxz"]


def test_a_character_without_a_dictionary_underlines_nothing(qt_app):
    text = "The dog qwxz"
    alice = _one_clip(qt_app, text)
    _stub_dictionaries(qt_app, {})
    qt_app.spellcheck_action.setChecked(True)
    assert _waved(qt_app.editor, text) == []
    assert alice is not None


# -- the dictionary the app builds -----------------------------------------------


def test_the_dictionary_counts_character_names_and_plain_lexicon_finds_as_known(qt_app):
    pytest.importorskip("spellchecker")
    qt_app.document.characters[0].name = "Qwxzv"
    qt_app.settings["lexicon"] = [
        {"find": "Zxqwy", "replace": "x", "mode": "word", "case": False},
        {"find": "Pqrst+", "replace": "x", "mode": "regex", "case": False},
    ]
    dictionary = qt_app.spell_dictionary_for()
    assert "hello" in dictionary and "qwxzv" in dictionary and "zxqwy" in dictionary
    assert "Pqrst" not in dictionary and "Pqrst+" not in dictionary


def test_the_dictionary_follows_the_characters_language_and_is_kept(qt_app):
    pytest.importorskip("spellchecker")
    character = qt_app.document.characters[0]
    character.preset_data["lang_code"] = "e"
    spanish = qt_app.spell_dictionary_for(character)
    assert spanish.language == "es" and "perro" in spanish and "dog" not in spanish
    assert qt_app.spell_dictionary_for(character) is spanish
    character.preset_data["lang_code"] = "j"
    assert qt_app.spell_dictionary_for(character) is None
    character.preset_data.pop("lang_code")
    assert qt_app.spell_dictionary_for(character).language == "en"


def test_the_language_falls_back_to_the_engines_setting(qt_app):
    character = qt_app.document.characters[0]
    assert qt_app.character_lang_code(character) == "a"
    character.preset_data["lang_code"] = "f"
    assert qt_app.character_lang_code(character) == "f"
