"""File > Import Audio as a recording to edit as text (phase 5 P3, grill
Q19/Q24/Q25/Q32/Q33): the first page, Whisper on a worker thread, caption
files with proportional times, speaker mapping and the optional Whisper
refine, the review dialog (play a line, correct a line and keep its times),
the cloning-capable character picker and one undo step."""
import os
import threading
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtWidgets import QFileDialog

from kokoro_gui.daw import imported
from kokoro_gui.daw.models import Character
from kokoro_gui.engine import asr
from kokoro_gui.engines import audio8_tts  # noqa: F401 - registers the cloning engine
from kokoro_gui.qt import recording_import
from kokoro_gui.qt.recording_import import CAPTIONS, RECORDING, UNKNOWN_SPEAKER, WHISPER, ImportAudioDialog
from kokoro_gui.qt.speaker_mapping_dialog import NARRATOR, NEW_CHARACTER

RATE = 16000
HEARD = [("Hello", 0.2, 0.5), ("there.", 0.6, 1.0), ("How", 1.1, 1.3), ("are", 1.3, 1.5), ("you?", 1.5, 1.9),
         ("Fine", 3.0, 3.4)]


def _wav(tmp_path, name="interview.wav", seconds=5.0):
    path = str(tmp_path / name)
    sf.write(path, np.linspace(-0.2, 0.2, int(RATE * seconds)).astype(np.float32), RATE)
    return path


def _host(qt_app):
    host = Character.from_preset_dict("Host", {"voice": "host_ref"}, backend_id="audio8")
    qt_app.document.characters.append(host)
    return host


def _answer_review(qt_app, monkeypatch, edit=None, character=None, accept=True):
    """Patches the review dialog: `edit(dialog)` runs first, then the
    character is picked and the dialog accepted (or cancelled). Returns the
    list the dialogs it was shown land in."""
    shown = []

    def _review(dialog):
        shown.append(dialog)
        if edit is not None:
            edit(dialog)
        if character is not None:
            assert dialog.set_character(character)
        return accept

    monkeypatch.setattr(qt_app, "_ask_recording_review", _review)
    return shown


def _heard_by(monkeypatch, words, calls=None, gate=None):
    def _transcribe(path, engine="whisper", model_path=None):
        if calls is not None:
            calls.append((path, engine, os.path.isfile(path)))
        if gate is not None:
            gate.wait(10)
        return list(words(path) if callable(words) else words)

    monkeypatch.setattr(asr, "transcribe_wav_words", _transcribe)
    monkeypatch.setattr(asr, "whisper_model_cached", lambda *a, **k: True)


def test_the_first_page_picks_bed_or_recording_and_the_transcript_source(qtbot):
    dialog = ImportAudioDialog("talk.wav")
    qtbot.addWidget(dialog)
    assert dialog.choice()["kind"] == "bed"
    assert not dialog.transcript_box.isEnabled()

    dialog.recording_radio.setChecked(True)
    assert dialog.choice() == {"kind": RECORDING, "transcript": WHISPER, "caption_path": "", "refine": False}
    dialog.captions_radio.setChecked(True)
    ok = dialog.buttons.buttons()[0]
    assert not ok.isEnabled()  # a caption file has to be named
    dialog.caption_edit.setText("/x/talk.srt")
    dialog.refine_check.setChecked(True)
    assert ok.isEnabled()
    assert dialog.choice() == {"kind": RECORDING, "transcript": CAPTIONS, "caption_path": "/x/talk.srt",
                               "refine": True}


def test_whisper_import_runs_off_the_gui_thread_and_makes_a_clip_per_sentence(qt_app, tmp_path, monkeypatch):
    document = qt_app.document
    document.text = "Show notes."
    qt_app.editor.load_text(document.text)
    host = _host(qt_app)
    path = _wav(tmp_path)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (path, "")))
    monkeypatch.setattr(qt_app, "_ask_audio_import",
                        lambda p: {"kind": RECORDING, "transcript": WHISPER, "caption_path": "", "refine": False})
    calls, gate = [], threading.Event()
    _heard_by(monkeypatch, HEARD, calls, gate)
    shown = _answer_review(qt_app, monkeypatch, character=host.id)

    qt_app.import_audio_dialog()
    assert qt_app.is_busy() and "Transcribing interview.wav" in qt_app.transport_dock.status_text()
    assert shown == []
    gate.set()
    qt_app.wait_for_recording_import()

    stored, _engine, _exists = calls[0]
    assert stored.startswith(os.path.join(qt_app.project_dir, "audio", "imported"))
    assert not qt_app.is_busy()
    # Only cloning-capable characters are offered, plus "Unknown speaker".
    combo = shown[0].character_combo
    assert [combo.itemData(i) for i in range(combo.count())] == [host.id, UNKNOWN_SPEAKER]

    assert document.text == "Show notes.\n\nHello there.\n\nHow are you?\n\nFine"
    assert qt_app.editor.toPlainText() == document.text
    recordings = [c for c in document.clips if imported.is_recording_clip(c)]
    assert len(recordings) == 3
    assert {c.character_id for c in recordings} == {host.id}
    assert [s.range for s in recordings[0].segments] == [[0.2, 1.0]]
    assert recordings[2].gap_before_s == pytest.approx(3.0 - 1.9)
    source, = document.sources
    assert document.source_path(source) == stored
    assert document.dirty_clips() == []

    qt_app.undo()
    assert document.text == "Show notes." and document.clips == [] and document.sources == {}
    assert qt_app.editor.toPlainText() == document.text


def test_unknown_speaker_makes_a_voiceless_cloning_character(qt_app, tmp_path, monkeypatch):
    document = qt_app.document
    _heard_by(monkeypatch, HEARD[:2])
    shown = _answer_review(qt_app, monkeypatch, character=UNKNOWN_SPEAKER)
    before = [c.id for c in document.characters]

    assert qt_app.import_recording(_wav(tmp_path))
    qt_app.wait_for_recording_import()

    combo = shown[0].character_combo
    assert [combo.itemData(i) for i in range(combo.count())] == [UNKNOWN_SPEAKER]  # the default is Kokoro
    speaker, = [c for c in document.characters if c.id not in before]
    assert speaker.name == "Unknown speaker" and "voice" not in speaker.preset_data
    assert qt_app.can_clone(speaker)
    clip, = document.clips
    assert clip.character_id == speaker.id

    qt_app.undo()
    assert [c.id for c in document.characters] == before


def test_review_edit_keeps_the_corrected_words_times(qt_app, tmp_path, monkeypatch):
    document = qt_app.document
    host = _host(qt_app)
    _heard_by(monkeypatch, [("Helo", 0.2, 0.5), ("their.", 0.6, 1.0)])
    _answer_review(qt_app, monkeypatch, character=host.id,
                   edit=lambda dialog: dialog.set_text(0, "Hello there."))

    qt_app.import_recording(_wav(tmp_path))
    qt_app.wait_for_recording_import()

    assert document.text == "Hello there."
    clip, = document.clips
    words = imported.clip_words(document, clip)
    assert [(document.text[w[0]:w[1]], w[3], w[4]) for w in words] == [("Hello", 0.2, 0.55), ("there", 0.55, 1.0)]
    # The same line as Whisper spelled it would have had the same times.
    assert [w[3:] for w in imported.run_from_asr_words([("Helo", 0.2, 0.5), ("their.", 0.6, 1.0)], "x")[1]] == [
        [0.2, 0.55], [0.55, 1.0]]


def test_review_play_button_plays_the_lines_slice(qt_app, tmp_path, monkeypatch):
    host = _host(qt_app)
    _heard_by(monkeypatch, HEARD)
    fake = MagicMock()
    monkeypatch.setattr(recording_import, "playback", fake)
    played = []

    def _edit(dialog):
        dialog.play_buttons[1].click()
        played.extend(fake.play_range.call_args_list)

    _answer_review(qt_app, monkeypatch, character=host.id, edit=_edit)
    qt_app.import_recording(_wav(tmp_path))
    qt_app.wait_for_recording_import()

    (args, _kwargs), = played
    assert args[0] == qt_app.document.source_path(next(iter(qt_app.document.sources)))
    assert args[1:] == (1.1, 1.9)  # "How are you?"


def test_cancelling_the_review_imports_nothing(qt_app, tmp_path, monkeypatch):
    document = qt_app.document
    document.text = "Notes"
    _heard_by(monkeypatch, HEARD)
    _answer_review(qt_app, monkeypatch, accept=False)

    qt_app.import_recording(_wav(tmp_path))
    qt_app.wait_for_recording_import()

    assert document.text == "Notes" and document.clips == [] and document.sources == {}


CAPTIONED = (
    "WEBVTT\n\n"
    "00:00.500 --> 00:02.000\n<v Alice>Where were you?\n\n"
    "00:02.500 --> 00:03.500\n<v Bob>Out.\n"
)


def test_a_caption_file_gives_proportional_times_and_maps_speakers_without_asr(qt_app, tmp_path, monkeypatch):
    document = qt_app.document
    narrator = document.characters[0]
    captions = tmp_path / "talk.vtt"
    captions.write_text(CAPTIONED, encoding="utf-8")
    calls = []
    _heard_by(monkeypatch, [], calls)
    asked = []

    def _mapping(speakers, characters):
        asked.append(list(speakers))
        return {"Alice": NEW_CHARACTER, "Bob": NARRATOR}

    monkeypatch.setattr(qt_app, "_ask_speaker_mapping", _mapping)
    shown = _answer_review(qt_app, monkeypatch)

    assert qt_app.import_recording(_wav(tmp_path), CAPTIONS, str(captions))

    assert calls == [] and qt_app._recording_thread is None  # no Whisper at all
    assert asked == [["Alice", "Bob"]]
    assert shown[0].character_combo is None  # the mapping chose the characters
    assert document.text == "Where were you?\n\nOut."
    first, second = document.clips
    alice = document.get_character_by_name("Alice")
    assert (first.character_id, second.character_id) == (alice.id, narrator.id)
    source, = document.sources
    # 1.5 s shared 5:4:4 by the words' lengths ("Where", "were", "you?").
    assert [w[3:] for w in imported.clip_words(document, first)] == [(0.5, 1.076923), (1.076923, 1.538462),
                                                                    (1.538462, 2.0)]
    assert [s.range for s in first.segments] == [[0.5, 2.0]]
    assert [s.range for s in second.segments] == [[2.5, 3.5]]

    qt_app.undo()
    assert document.text == "" and document.get_character_by_name("Alice") is None


def test_refine_retimes_each_cue_on_its_slice_and_keeps_the_text(qt_app, tmp_path, monkeypatch):
    document = qt_app.document
    host = _host(qt_app)
    captions = tmp_path / "talk.srt"
    captions.write_text("1\n00:00:01,000 --> 00:00:03,000\nHello there\n", encoding="utf-8")
    calls = []
    # Whisper hears the words late in the cue, and spells one differently;
    # times are relative to the slice it is given.
    _heard_by(monkeypatch, [("hello", 0.8, 1.2), ("their", 1.3, 1.8)], calls)
    _answer_review(qt_app, monkeypatch, character=host.id)

    assert qt_app.import_recording(_wav(tmp_path), CAPTIONS, str(captions), refine=True)
    qt_app.wait_for_recording_import()

    (slice_path, engine, existed), = calls
    assert engine == "whisper" and existed and not os.path.exists(slice_path)
    assert document.text == "Hello there"
    clip, = document.clips
    assert [w[3:] for w in imported.clip_words(document, clip)] == [(1.8, 2.25), (2.25, 2.8)]
