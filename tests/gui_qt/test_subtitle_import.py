"""File > Import Subtitles (phase 5 D2): cues as pinned clips, the speaker
mapping dialog, one undo step, and ripple leaving the cues where they are."""
import os

import numpy as np
import soundfile as sf

from kokoro_gui.daw.models import Character
from kokoro_gui.qt.speaker_mapping_dialog import (
    NARRATOR, NEW_CHARACTER, NO_SPEAKER, SpeakerMappingDialog, resolve_mapping, speaker_rows,
)

FIXTURES = os.path.join(os.path.dirname(__file__), os.pardir, "daw", "fixtures", "subtitles")

TWO_SPEAKERS = (
    "[Events]\n"
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    "Dialogue: 0,0:00:01.00,0:00:02.50,Default,Alice,0,0,0,,Where were you?\n"
    "Dialogue: 0,0:00:03.00,0:00:04.00,Default,Bob,0,0,0,,Out.\n"
    "Dialogue: 0,0:00:05.00,0:00:07.00,Default,,0,0,0,,The door shuts.\n"
)


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def _answer_mapping(qt_app, monkeypatch, answer):
    """Patches the mapping dialog to return `answer` (a dict, or None for
    Cancel) and records what it was asked."""
    asked = []

    def _ask(speakers, characters):
        asked.append((list(speakers), [c.id for c in characters]))
        return answer(speakers, characters) if callable(answer) else answer

    monkeypatch.setattr(qt_app, "_ask_speaker_mapping", _ask)
    return asked


def _snapshot(document):
    return (document.text, [c.id for c in document.clips], [c.id for c in document.characters],
            [t.id for t in document.tracks])


def test_import_maps_two_speakers_and_pins_a_clip_per_cue(qt_app, tmp_path, monkeypatch):
    document = qt_app.document
    document.text = "Scene one."
    qt_app.editor.load_text(document.text)
    narrator = document.characters[0]
    path = _write(tmp_path, "scene.ass", TWO_SPEAKERS)
    asked = _answer_mapping(qt_app, monkeypatch,
                            lambda speakers, characters: {"Alice": NEW_CHARACTER, "Bob": narrator.id,
                                                          NO_SPEAKER: NARRATOR})

    clip_ids = qt_app.import_subtitles(path)

    assert asked == [(["Alice", "Bob", NO_SPEAKER], [c.id for c in document.characters if c.name != "Alice"])]
    assert document.text == "Scene one.\n\nWhere were you?\n\nOut.\n\nThe door shuts."
    assert qt_app.editor.toPlainText() == document.text
    clips = [document.get_clip(i) for i in clip_ids]
    assert [document.clip_text(c) for c in clips] == ["Where were you?", "Out.", "The door shuts."]
    assert [c.timeline_timestamp for c in clips] == [1.0, 3.0, 5.0]
    assert all(c.pinned for c in clips)
    assert [c.source_text for c in clips] == ["Where were you?", "Out.", "The door shuts."]
    assert [c.overrides["target_duration_s"] for c in clips] == [1.5, 1.0, 2.0]

    alice = document.get_character_by_name("Alice")
    assert alice is not None and alice.library_id is None
    assert [c.character_id for c in clips] == [alice.id, narrator.id, narrator.id]
    # Alice's track is drawn now that a clip uses it.
    assert document.get_track(clips[0].track_id).character_id == alice.id
    names = [qt_app.transcript_dock.character_combo.itemText(i)
             for i in range(qt_app.transcript_dock.character_combo.count())]
    assert "Alice" in names

    # The timeline places each cue at its time.
    placed = qt_app.build_arrangement().by_clip_id()
    assert [placed[i].start_s for i in clip_ids] == [1.0, 3.0, 5.0]


def test_srt_without_speakers_skips_the_dialog(qt_app, monkeypatch):
    def _no_dialog(*_a):
        raise AssertionError("no speakers: the dialog isn't shown")

    monkeypatch.setattr(qt_app, "_ask_speaker_mapping", _no_dialog)
    document = qt_app.document
    narrator = document.characters[0]

    clip_ids = qt_app.import_subtitles(os.path.join(FIXTURES, "dialogue.srt"))

    clips = [document.get_clip(i) for i in clip_ids]
    assert document.text == "Where were you last night?\n\nOut.\n\nLate, then."
    assert clips[0].source_text == "Where were you\nlast night?"
    assert {c.character_id for c in clips} == {narrator.id}
    assert [c.timeline_timestamp for c in clips] == [1.0, 4.25, 3723.004]


def test_cancel_leaves_the_document_unchanged(qt_app, tmp_path, monkeypatch):
    document = qt_app.document
    document.text = "Before."
    qt_app.editor.load_text(document.text)
    before = _snapshot(document)
    _answer_mapping(qt_app, monkeypatch, None)

    assert qt_app.import_subtitles(_write(tmp_path, "scene.ass", TWO_SPEAKERS)) == []

    assert _snapshot(document) == before
    assert not document.undo_stack.can_undo()
    assert qt_app.editor.toPlainText() == "Before."


def test_one_undo_removes_the_whole_import(qt_app, tmp_path, monkeypatch):
    document = qt_app.document
    document.text = "Before."
    qt_app.editor.load_text(document.text)
    before = _snapshot(document)
    _answer_mapping(qt_app, monkeypatch, {"Alice": NEW_CHARACTER, "Bob": NEW_CHARACTER, NO_SPEAKER: NARRATOR})

    clip_ids = qt_app.import_subtitles(_write(tmp_path, "scene.ass", TWO_SPEAKERS))
    assert len(clip_ids) == 3 and len(document.characters) == len(before[2]) + 2

    qt_app.undo()

    assert _snapshot(document) == before
    assert qt_app.editor.toPlainText() == "Before."
    combo = qt_app.transcript_dock.character_combo
    assert "Alice" not in [combo.itemText(i) for i in range(combo.count())]

    qt_app.redo()
    assert [c.id for c in document.clips] == clip_ids
    assert qt_app.editor.toPlainText().endswith("The door shuts.")


def test_menu_action_imports_the_picked_file(qt_app, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    picked = os.path.join(FIXTURES, "bom.srt")
    filters = []

    def _pick(*_a, **k):
        filters.append(k.get("filter"))
        return picked, ""

    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(_pick))

    qt_app.import_subtitles_action.trigger()

    assert "*.srt" in filters[0] and "*.vtt" in filters[0] and "*.ass" in filters[0]
    assert qt_app.document.text == "Café au lait, s'il vous plaît."
    assert qt_app.document.clips[0].pinned


def test_a_bad_file_imports_nothing(qt_app, tmp_path):
    before = _snapshot(qt_app.document)

    assert qt_app.import_subtitles(str(tmp_path / "missing.srt")) == []
    assert qt_app.import_subtitles(_write(tmp_path, "empty.vtt", "WEBVTT\n\n")) == []

    assert _snapshot(qt_app.document) == before


def test_import_goes_to_the_focused_subproject(qt_app, tmp_path, monkeypatch):
    root = qt_app.document
    root.text = "Intro. Chapter text. Outro."
    qt_app.editor.load_text(root.text)
    child = qt_app.new_subproject(7, 20, title="Chapter 1")
    qt_app.selection.select_clip(child.clip_id)
    assert qt_app.document is child.document
    root_before = _snapshot(root)
    _answer_mapping(qt_app, monkeypatch, {"Alice": NARRATOR, "Bob": NARRATOR, NO_SPEAKER: NARRATOR})

    clip_ids = qt_app.import_subtitles(_write(tmp_path, "scene.ass", TWO_SPEAKERS))

    assert all(child.document.get_clip(i) is not None for i in clip_ids)
    assert child.document.text.endswith("The door shuts.")
    assert qt_app.editor.toPlainText() == child.document.text
    assert _snapshot(root) == root_before


# -- ripple (phase 3) never moves an imported cue --------------------------------------


def _wav(tmp_path, name, seconds):
    path = tmp_path / name
    sf.write(str(path), np.full(int(24000 * seconds), 0.2, dtype=np.float32), 24000)
    return str(path)


def _generate_with(qt_app, clip, path, seconds, take=0, key="k"):
    import concurrent.futures

    qt_app.engine.worker.run_coro.return_value = concurrent.futures.Future()
    qt_app.transport_dock.set_busy(False)
    qt_app.timeline_dock.on_generate_clip_requested(clip.id)
    qt_app.engine.worker.run_coro.return_value.set_result([
        {"path": path, "text": qt_app.document.clip_text(clip), "duration": seconds, "seg_idx": 0,
         "cache_key": key, "take": take},
    ])


def test_an_imported_cue_survives_a_regenerate_ripple(qt_app, tmp_path, monkeypatch):
    document = qt_app.document
    document.settings["gap_s"] = 0.0
    document.text = "aaa bbb"
    qt_app.editor.load_text(document.text)
    character = document.characters[0]
    a = document.assign_character_to_range(0, 3, character.id)
    b = document.assign_character_to_range(4, 7, character.id)
    _generate_with(qt_app, a, _wav(tmp_path, "a0.wav", 1.0), 1.0)
    b.timeline_timestamp = 2.0  # dragged: ripple moves it
    cue_id = qt_app.import_subtitles(_write(tmp_path, "cue.srt", "1\n00:00:03,000 --> 00:00:04,000\nStay.\n"))[0]

    _generate_with(qt_app, a, _wav(tmp_path, "a1.wav", 1.5), 1.5, take=1, key="k1")

    assert abs(b.timeline_timestamp - 2.5) < 1e-3
    cue = document.get_clip(cue_id)
    assert cue.timeline_timestamp == 3.0 and cue.pinned
    assert qt_app.build_arrangement().by_clip_id()[cue_id].start_s == 3.0


def test_dragging_a_cue_moves_it_and_keeps_it_locked(qt_app, tmp_path):
    """Phase 3's "Lock in time": a drag still moves a pinned clip, and the
    lock stays (the plan's confirm-and-unpin on drag isn't built, see the
    D2 report)."""
    cue_id = qt_app.import_subtitles(_write(tmp_path, "cue.srt", "1\n00:00:03,000 --> 00:00:04,000\nMove me.\n"))[0]

    qt_app.timeline_dock.on_clip_moved(cue_id, 6.0)

    cue = qt_app.document.get_clip(cue_id)
    assert cue.timeline_timestamp == 6.0 and cue.pinned


# -- the dialog on its own (P3 reuses it) -----------------------------------------------


def test_speaker_rows_lists_each_speaker_once_then_the_untagged_row():
    from kokoro_gui.daw.subtitles import Cue

    cues = [Cue(0, 1, "a", "Bob"), Cue(1, 2, "b", None), Cue(2, 3, "c", "Alice"), Cue(3, 4, "d", "Bob")]

    assert speaker_rows(cues) == ["Bob", "Alice", NO_SPEAKER]
    assert speaker_rows([Cue(0, 1, "a")]) == []


def test_dialog_defaults_and_mapping(qtbot):
    default = Character.from_preset_dict("Default", {})
    alice = Character.from_preset_dict("Alice", {})
    dialog = SpeakerMappingDialog(["alice", "Bob", NO_SPEAKER], [default, alice])
    qtbot.addWidget(dialog)

    # A name match picks the character; an unknown speaker a new one; the
    # untagged row the narrator (and it has no "new character" choice).
    assert dialog.mapping() == {"alice": alice.id, "Bob": NEW_CHARACTER, NO_SPEAKER: NARRATOR}
    assert dialog.combo_for(NO_SPEAKER).findData(NEW_CHARACTER) == -1
    assert dialog.combo_for("Bob").itemText(0) == "Narrator (Default)"

    assert dialog.set_choice("Bob", default.id)
    assert dialog.set_choice(NO_SPEAKER, NEW_CHARACTER) is False
    assert dialog.mapping()["Bob"] == default.id


def test_resolve_mapping_makes_new_characters_and_finds_the_narrator():
    default = Character.from_preset_dict("Default", {})
    bob = Character.from_preset_dict("Bob", {})
    made = []

    def _make(name, index):
        made.append(index)
        return Character.from_preset_dict(name, {})

    ids, new = resolve_mapping({"Bob": NEW_CHARACTER, "Eve": NEW_CHARACTER, "Ann": bob.id, NO_SPEAKER: NARRATOR},
                               [bob, default], _make)

    assert [c.name for c in new] == ["Bob 2", "Eve"]  # "Bob" is taken
    assert made == [2, 3]
    assert ids == {"Bob": new[0].id, "Eve": new[1].id, "Ann": bob.id, NO_SPEAKER: default.id}

    # No characters at all: the narrator is a new "Default".
    ids, new = resolve_mapping({NO_SPEAKER: NARRATOR, "X": NARRATOR}, [], _make)
    assert [c.name for c in new] == ["Default"]
    assert ids == {NO_SPEAKER: new[0].id, "X": new[0].id}
