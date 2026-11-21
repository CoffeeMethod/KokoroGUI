"""Clone a speaker from a selection inside their recording, to fix a flub
(plan 33 part A, grill PG9): the selection's span, the 5 to 30 s rule, the
reference saved to the global Audio8 store, the new character or variant,
and the consent checkbox that is never stored anywhere but the program's
own "don't ask again" setting."""
import json
import os

import numpy as np
import pytest
import soundfile as sf

from kokoro_gui.daw import imported
from kokoro_gui.daw.models import Character
from kokoro_gui.daw.serialization import document_to_dict
from kokoro_gui.daw.undo import ImportRecordingCommand
from kokoro_gui.engines import audio8_tts
from kokoro_gui.qt import clone_from_recording as clone
from kokoro_gui.qt import project as project_io
from kokoro_gui.qt import settings as qt_settings
from kokoro_gui.qt import spec

RATE = 16000
# Twenty words, one per half second: 0.0 to 9.9 s.
WORDS = [(f"word{i}", i * 0.5, i * 0.5 + 0.4) for i in range(20)]


@pytest.fixture
def refs(qt_app, tmp_path, monkeypatch):
    """The global Audio8 store in tmp_path; the model is never loaded."""
    from kokoro_gui.engine import runtime

    monkeypatch.setattr(runtime, "CUSTOM_VOICES_DIR", str(tmp_path / "custom_voices"))
    monkeypatch.setattr(audio8_tts, "AUDIO8_REFS_DIR", str(tmp_path / "audio8_refs"))
    monkeypatch.setattr(audio8_tts, "_get_model", lambda: pytest.fail("the Audio8 model was loaded"))
    return tmp_path / "audio8_refs"


def _recording(qt_app, tmp_path, character):
    """A 10 s recording of WORDS as one clip voiced by `character`.
    Returns the clip id."""
    path = str(tmp_path / "talk.wav")
    sf.write(path, np.linspace(-0.2, 0.2, RATE * 10).astype(np.float32), RATE)
    stored = project_io.import_audio_file(path, qt_app.project_dir)
    source, entry = imported.source_entry(stored)
    text, words = imported.run_from_asr_words(WORDS, source)
    command = ImportRecordingCommand([{"text": text, "words": words, "character_id": character.id}],
                                     {source: entry})
    qt_app.document.undo_stack.push(command)
    qt_app.editor.load_text(qt_app.document.text)
    return command.clip_ids[0]


def _span_of(document, first: str, last: str) -> tuple:
    """Text offsets from the start of word `first` to the end of `last`."""
    text = document.text
    return text.index(first + " "), text.index(last) + len(last)


def _times(document, first: str, last: str) -> tuple:
    """The recording's start time of word `first` and end time of `last`
    (the import tiles the gaps between words)."""
    times = {document.text[a:b]: (s, e) for a, b, _source, s, e in imported.clip_words(document, document.clips[-1])}
    return times[first][0], times[last][1]


def _select(editor, start: int, end: int) -> None:
    cursor = editor.textCursor()
    cursor.setPosition(start)
    cursor.setPosition(end, cursor.MoveMode.KeepAnchor)
    editor.setTextCursor(cursor)


def _answer(monkeypatch, consent=True, dont_ask=False, accept=True, name=None):
    """Patches the dialog: ticks the boxes, then reports Clone (when it is
    enabled) or Cancel. Returns the list of dialogs shown."""
    shown = []

    def _ask(dialog):
        shown.append(dialog)
        if dialog.ask_consent:
            dialog.consent_check.setChecked(consent)
            dialog.dont_ask_check.setChecked(dont_ask)
        if name is not None:
            dialog.name_edit.setText(name)
        return accept and dialog.ok_button.isEnabled()

    monkeypatch.setattr(clone, "ask", _ask)
    return shown


def _host(qt_app, engine="kokoro"):
    host = Character.from_preset_dict("Host", {}, backend_id=engine)
    qt_app.document.characters.append(host)
    return host


def test_the_span_is_the_selected_words_first_and_last_times(qt_app, tmp_path, refs):
    host = _host(qt_app)
    clip_id = _recording(qt_app, tmp_path, host)
    start, end = _span_of(qt_app.document, "word2", "word15")

    span = clone.clone_span(qt_app.document, start, end)

    assert span.clip_id == clip_id and span.character_id == host.id
    assert (span.start_s, span.end_s) == _times(qt_app.document, "word2", "word15")
    assert span.text.startswith("word2 ") and span.text.endswith("word15")
    assert os.path.isfile(span.path)


def test_the_menu_entry_needs_a_selection_inside_a_recording(qt_app, tmp_path, refs):
    host = _host(qt_app)
    _recording(qt_app, tmp_path, host)
    editor = qt_app.editor

    def entry():
        return next(a for a in editor._build_context_menu().actions() if a.text() == clone.MENU_TEXT)

    _select(editor, 0, 0)
    assert not entry().isEnabled()
    _select(editor, *_span_of(qt_app.document, "word0", "word12"))
    assert entry().isEnabled()


def test_a_selection_outside_any_recording_has_no_span(qt_app, refs):
    qt_app.editor.setPlainText("Plain typed narration with no recording at all.")
    assert clone.clone_span(qt_app.document, 0, 20) is None


def test_under_five_seconds_is_refused(qt_app, tmp_path, refs, monkeypatch):
    host = _host(qt_app)
    _recording(qt_app, tmp_path, host)
    shown = _answer(monkeypatch)

    result = clone.clone_from_selection(qt_app, *_span_of(qt_app.document, "word0", "word4"))

    assert result is None
    assert "at least 5 s" in shown[0].problem()
    assert not shown[0].ok_button.isEnabled()
    assert not refs.exists() or not os.listdir(refs)


def test_the_length_rule_has_both_ends():
    assert clone.duration_problem(4.9)
    assert clone.duration_problem(5.0) is None
    assert clone.duration_problem(30.0) is None
    assert "at most 30 s" in clone.duration_problem(30.1)


def test_clone_saves_the_selection_as_a_reference_and_makes_a_character(qt_app, tmp_path, refs, monkeypatch):
    host = _host(qt_app)
    _recording(qt_app, tmp_path, host)
    played = []
    monkeypatch.setattr(clone.playback, "play_range", lambda *a: played.append(a))
    shown = _answer(monkeypatch)
    before = len(qt_app.document.characters)

    result = clone.clone_from_selection(qt_app, *_span_of(qt_app.document, "word2", "word15"))

    assert result is not None and result.reference == "Host (clone)"
    # The wav holds exactly the selected stretch.
    start_s, end_s = _times(qt_app.document, "word2", "word15")
    data, rate = sf.read(result.wav_path)
    assert rate == RATE and len(data) == round(end_s * RATE) - round(start_s * RATE)
    assert os.path.dirname(result.wav_path) == str(refs)
    with open(os.path.join(refs, "Host (clone).txt"), encoding="utf-8") as f:
        assert f.read() == shown[0].transcript_edit.toPlainText().strip()
    # A new local Audio8 character speaks it; the recording's own is unchanged.
    assert len(qt_app.document.characters) == before + 1
    character = qt_app.document.get_character_by_name("Host (clone)")
    assert character is result.character and result.variant is None
    assert character.backend_id == "audio8" and character.preset_data["voice"] == "Host (clone)"
    assert character.library_id is None
    assert host.backend_id == "kokoro" and not host.variants
    shown[0].play()
    assert played[0][1:] == (start_s, end_s)


def test_the_permission_box_is_required_and_never_stored(qt_app, tmp_path, refs, monkeypatch):
    host = _host(qt_app)
    _recording(qt_app, tmp_path, host)
    span = _span_of(qt_app.document, "word2", "word15")

    shown = _answer(monkeypatch, consent=False)
    assert clone.clone_from_selection(qt_app, *span) is None
    assert "permission" in shown[0].problem()
    assert shown[0].consent_check.isVisibleTo(shown[0])

    _answer(monkeypatch, consent=True)
    result = clone.clone_from_selection(qt_app, *span)
    assert result is not None
    # Nothing records it: the store holds the wav and transcript only, the
    # character carries no marker and the project's document has no word of it.
    assert sorted(os.listdir(refs)) == ["Host (clone).txt", "Host (clone).wav"]
    assert "consent" not in json.dumps(document_to_dict(qt_app.document)).lower()
    assert qt_app.settings[clone.CONSENT_SETTING] is True


def test_dont_ask_again_lives_in_program_settings_only(qt_app, tmp_path, refs, monkeypatch):
    host = _host(qt_app)
    _recording(qt_app, tmp_path, host)
    _answer(monkeypatch, dont_ask=True)

    assert clone.clone_from_selection(qt_app, *_span_of(qt_app.document, "word2", "word15")) is not None

    assert qt_app.settings[clone.CONSENT_SETTING] is False
    assert not qt_app.clone_consent_action.isChecked()
    assert "consent" not in json.dumps(document_to_dict(qt_app.document)).lower()
    # The next clone asks nothing and Clone is on without the box.
    shown = _answer(monkeypatch, name="Second take")
    assert clone.clone_from_selection(qt_app, *_span_of(qt_app.document, "word3", "word16")) is not None
    assert not shown[0].ask_consent and not shown[0].consent_check.isVisibleTo(shown[0])
    # Options > Confirm consent before cloning asks again.
    qt_app.clone_consent_action.setChecked(True)
    assert qt_app.settings[clone.CONSENT_SETTING] is True


def test_the_setting_goes_through_the_settings_type_check():
    assert spec.SETTINGS_DEFAULTS[clone.CONSENT_SETTING] is True
    cleaned = qt_settings.clean_loaded_settings({clone.CONSENT_SETTING: "no"}, {clone.CONSENT_SETTING: True})
    assert cleaned[clone.CONSENT_SETTING] is True
    cleaned = qt_settings.clean_loaded_settings({clone.CONSENT_SETTING: False}, {clone.CONSENT_SETTING: True})
    assert cleaned[clone.CONSENT_SETTING] is False


def test_an_audio8_speaker_gets_the_clone_as_a_variant(qt_app, tmp_path, refs, monkeypatch):
    host = _host(qt_app, engine="audio8")
    _recording(qt_app, tmp_path, host)
    _answer(monkeypatch)
    before = len(qt_app.document.characters)

    result = clone.clone_from_selection(qt_app, *_span_of(qt_app.document, "word2", "word15"))

    assert result.character is host and result.variant == "Host (clone)"
    assert host.variants == {"Host (clone)": "Host (clone)"}
    assert len(qt_app.document.characters) == before
    assert "variant Host (clone)" in result.how_to_use()


def test_a_name_already_in_use_is_refused(qt_app, tmp_path, refs, monkeypatch):
    host = _host(qt_app)
    _recording(qt_app, tmp_path, host)
    os.makedirs(refs, exist_ok=True)
    sf.write(str(refs / "Taken.wav"), np.zeros(RATE, dtype=np.float32), RATE)
    (refs / "Taken.txt").write_text("hello", encoding="utf-8")
    span = _span_of(qt_app.document, "word2", "word15")

    shown = _answer(monkeypatch, name="Taken")
    assert clone.clone_from_selection(qt_app, *span) is None
    assert "already exists" in shown[0].problem()

    shown = _answer(monkeypatch, name="Host")
    assert clone.clone_from_selection(qt_app, *span) is None
    assert "character named Host" in shown[0].problem()
    with open(refs / "Taken.txt", encoding="utf-8") as f:
        assert f.read() == "hello"


def test_cloning_is_refused_while_busy(qt_app, tmp_path, refs, monkeypatch):
    host = _host(qt_app)
    _recording(qt_app, tmp_path, host)
    shown = _answer(monkeypatch)
    monkeypatch.setattr(qt_app, "is_busy", lambda: True)

    assert clone.clone_from_selection(qt_app, *_span_of(qt_app.document, "word2", "word15")) is None
    assert shown == []


def test_the_correction_typed_inside_the_recording_takes_the_clone(qt_app, tmp_path, refs, monkeypatch):
    """The flow the final message describes: typed text splits the
    recording, and the Characters menu gives it the clone."""
    host = _host(qt_app)
    clip_id = _recording(qt_app, tmp_path, host)
    _answer(monkeypatch)
    result = clone.clone_from_selection(qt_app, *_span_of(qt_app.document, "word2", "word15"))
    document, editor = qt_app.document, qt_app.editor
    at = document.text.index("word10")
    cursor = editor.textCursor()
    cursor.setPosition(at)
    editor.setTextCursor(cursor)
    editor.insertPlainText("fixed ")
    _select(editor, at, at + len("fixed"))

    editor._assign_character(result.character.id)

    correction = document.clip_covering(at)
    assert correction.id != clip_id and correction.character_id == result.character.id
    assert document.get_clip(clip_id).source == "imported"
