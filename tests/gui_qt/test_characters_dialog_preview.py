"""The Preview button in Edit > Characters: it speaks the dialog's character as
it is right now (an unsaved voice, engine or variant included) through
`app.preview_text`, and is disabled while a preview runs."""
from concurrent.futures import Future
from unittest.mock import MagicMock

import pytest

import playback
from kokoro_gui.daw.models import Character
from kokoro_gui.qt.characters_dialog import CharactersDialog, cut_sample


@pytest.fixture(autouse=True)
def _no_audio(monkeypatch):
    monkeypatch.setattr(playback, "play", MagicMock())
    monkeypatch.setattr(playback, "stop", MagicMock())


def _calls(qt_app):
    return qt_app.engine.generate_preview.call_args


def test_cut_sample_keeps_short_text_whole():
    assert cut_sample("  One   line.\nTwo. ") == "One line. Two."


def test_cut_sample_ends_on_a_sentence_end_within_the_limit():
    text = "First sentence here. " + "Second sentence goes on and on. " * 20
    cut = cut_sample(text)
    assert len(cut) <= 200 and cut.endswith(".")
    assert cut.startswith("First sentence here. Second")


def test_cut_sample_falls_back_to_a_word_then_a_hard_cut():
    assert cut_sample("word " * 100) == ("word " * 40).strip()
    assert cut_sample("x" * 500) == "x" * 200


def test_preview_speaks_the_edited_voice_and_the_stock_sentence_without_clips(qt_app):
    dialog = CharactersDialog(qt_app)
    dialog.voice_combo.setCurrentText("bf_emma")  # not saved anywhere else
    character = dialog._current
    character.preset_data["speed"] = 1.25

    assert dialog.preview_btn.isEnabled()
    assert dialog.preview_character() is True
    (text, voice, speed, path, _extra), kwargs = _calls(qt_app)
    assert (voice, speed) == ("bf_emma", 1.25)
    assert text == qt_app.backend.preview_text(kwargs["lang_code"])
    assert path.endswith(".wav") and "kokorogui-preview-" in path
    assert not dialog.preview_btn.isEnabled()
    assert dialog.preview_status_label.text() == "Generating preview..."


def test_the_button_comes_back_when_the_preview_finishes_or_fails(qt_app):
    dialog = CharactersDialog(qt_app)
    dialog.preview_character()
    assert not dialog.preview_btn.isEnabled()
    assert dialog.preview_character() is False  # one at a time

    qt_app.previewFinished.emit(False, "Preview failed.")
    assert dialog.preview_btn.isEnabled()
    assert dialog.preview_status_label.text() == "Preview failed."

    dialog.preview_character()
    qt_app.previewFinished.emit(True, "x.wav")
    assert dialog.preview_btn.isEnabled()
    assert dialog.preview_status_label.text() == "Playing preview."


def test_preview_speaks_the_characters_last_clip(qt_app):
    doc = qt_app.document
    doc.text = "Early words here. A later line follows. " + "Tail sentence. " * 30
    character = doc.characters[0]
    other = Character(name="Other")
    doc.characters.append(other)
    first = doc.assign_character_to_range(0, 17, character.id)
    doc.assign_character_to_range(18, 39, other.id)
    last_start = doc.text.index("Tail")
    last = doc.assign_character_to_range(last_start, len(doc.text), character.id)
    assert first.id != last.id

    dialog = CharactersDialog(qt_app)
    dialog.preview_character()
    text = _calls(qt_app)[0][0]
    assert text.startswith("Tail sentence.") and text.endswith(".") and len(text) <= 200


def test_preview_leaves_the_character_alone(qt_app):
    dialog = CharactersDialog(qt_app)
    character = dialog._current
    before = (dict(character.preset_data), dict(character.variants or {}), character.backend_id)
    dialog.preview_character()
    assert (dict(character.preset_data), dict(character.variants or {}), character.backend_id) == before


def test_a_cloning_character_previews_the_picked_variants_reference(qt_app, toneclone_plugin):
    character = qt_app.document.characters[0]
    assert qt_app.set_character_engine(character, "toneclone")
    character.preset_data["voice"] = "Echo"
    character.variants = {"whisper": "Hush", "shout": "Bang"}
    backend = qt_app.backend_for_character(character)
    seen = []

    def fake_preview(text, voice, speed, output_path, extra_config=None, lang_code=None):
        seen.append((voice, speed))
        return Future()

    backend.preview = fake_preview
    backend.is_ready = lambda: True

    dialog = CharactersDialog(qt_app)
    assert dialog.preview_variant_combo.isVisibleTo(dialog)
    assert [dialog.preview_variant_combo.itemText(i) for i in range(dialog.preview_variant_combo.count())] == \
        ["(default)", "shout", "whisper"]

    dialog.preview_character()
    assert seen[-1][0] == "Echo"
    qt_app.previewFinished.emit(True, "x.wav")

    dialog.preview_variant_combo.setCurrentIndex(dialog.preview_variant_combo.findData("whisper"))
    dialog.preview_character()
    assert seen[-1][0] == "Hush"
    assert character.preset_data["voice"] == "Echo"  # the character kept its voice


def test_a_character_without_variants_has_no_variant_combo(qt_app):
    dialog = CharactersDialog(qt_app)
    assert not dialog.preview_variant_combo.isVisibleTo(dialog)


def test_a_missing_engine_says_so_and_speaks_nothing(qt_app):
    ghost = Character.from_preset_dict("Ghost", {"voice": "boo"}, backend_id="ghost")
    qt_app.document.characters.append(ghost)
    dialog = CharactersDialog(qt_app)
    dialog._select(ghost)

    assert dialog.preview_character() is False
    assert "isn't installed" in dialog.preview_status_label.text()
    assert dialog.preview_btn.isEnabled()
    qt_app.engine.generate_preview.assert_not_called()


def test_an_engine_that_is_still_loading_leaves_the_button_usable(qt_app):
    qt_app.backend.engine.pipeline = None
    dialog = CharactersDialog(qt_app)
    assert dialog.preview_character() is False
    assert dialog.preview_btn.isEnabled()
    assert dialog.preview_status_label.text() == "Nothing to preview with yet."
