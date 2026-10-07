"""Proof by ASR (plan 21): the worker, the results store in `session.json`,
the Proof dock and the timeline mark. The transcribe call is always mocked."""
import os
import zipfile

import numpy as np
import pytest
import soundfile as sf

import kokoro_gui.qt.app as qt_app_module
from kokoro_gui.daw import proof
from kokoro_gui.daw.dirty import build_segments_from_results
from kokoro_gui.engine import asr
from kokoro_gui.qt import asr_prompt, project as project_io
from kokoro_gui.qt.proofing import SCOPE_ALL, SCOPE_SELECTION, SCOPE_SUBPROJECT


def _clip(qt_app, start, end, pieces=None):
    """A clean clip over `[start, end)`: one real wav per piece in the
    project dir, each segment's text the piece (the whole clip by default)."""
    document = qt_app.document
    clip = document.assign_character_to_range(start, end, document.characters[0].id)
    text = document.clip_text(clip)
    key = document.segment_key_fn(text, clip)
    generated = os.path.join(qt_app.project_dir, "audio", "generated")
    os.makedirs(generated, exist_ok=True)
    results = []
    for index, piece in enumerate(pieces or [text]):
        path = os.path.join(generated, f"{key}_{index}.wav")
        sf.write(path, np.full(2400, 0.2, dtype=np.float32), 24000)
        results.append({"text": piece, "path": path, "duration": 0.1, "cache_key": key,
                        "engine_version": qt_app.backend.engine_version()})
    clip.segments = build_segments_from_results(key, results)
    return clip


def _words(text):
    return [(w, i * 0.3, i * 0.3 + 0.2) for i, w in enumerate(text.split())]


@pytest.fixture
def heard(monkeypatch):
    """`heard.say(segment, text)` sets what the mocked Whisper hears in that
    segment's file; `heard.calls` lists the files it was asked about;
    `heard.on_call(path)` runs on the worker thread at each call."""
    class Heard:
        def __init__(self):
            self.by_path = {}
            self.calls = []
            self.on_call = None

        def say(self, segment, text):
            self.by_path[segment.audio_path] = _words(text)

    state = Heard()

    def transcribe(path, engine="whisper", *args, **kwargs):
        state.calls.append(path)
        if state.on_call is not None:
            state.on_call(path)
        if path not in state.by_path:
            raise RuntimeError("unreadable")
        return list(state.by_path[path])

    monkeypatch.setattr(asr, "transcribe_wav_words", transcribe)
    return state


def _two_clips(qt_app, heard):
    """Two generated clips; Whisper hears the first right and the second with
    a dropped word."""
    qt_app.document.text = "The old mill stood. It was cold."
    qt_app.editor.load_text(qt_app.document.text)
    good = _clip(qt_app, 0, 19)
    bad = _clip(qt_app, 20, 31)
    heard.say(good.segments[0], "the old mill stood")
    heard.say(bad.segments[0], "it cold")
    return good, bad


def _proof_all(qt_app):
    ids, _message = qt_app.clips_for_proof_scope(SCOPE_ALL)
    assert qt_app.run_proof(ids)
    qt_app.wait_for_proof()


def _shown_dock(qt_app, qtbot):
    qt_app.show()
    qtbot.waitExposed(qt_app)
    qt_app.proof_dock.show()
    qt_app.proof_dock.raise_()
    qt_app.flush_updates()
    qt_app.proof_dock.rebuild()
    return qt_app.proof_dock


def _block(qt_app, clip):
    return qt_app.timeline_dock.timeline_view._blocks_by_clip_id[clip.id]


# -- the worker ---------------------------------------------------------------------------------


def test_a_proof_scores_each_clip_and_flags_the_one_heard_wrong(qt_app, heard):
    good, bad = _two_clips(qt_app, heard)

    _proof_all(qt_app)

    assert qt_app.proof_entry(good)["ratio"] == 1.0
    assert qt_app.proof_entry(bad)["ratio"] < proof.DEFAULT_THRESHOLD
    assert qt_app.proof_entry(bad)["issues"][0][0] == "dropped"
    assert not qt_app.is_clip_flagged(good) and qt_app.is_clip_flagged(bad)
    assert qt_app.is_proofing is False


def test_proofing_does_not_mark_the_project_unsaved_or_touch_the_document(qt_app, heard):
    good, bad = _two_clips(qt_app, heard)
    qt_app.flush_updates()
    statuses = [c.status for c in qt_app.document.clips]
    unsaved = qt_app.any_project_dirty()

    _proof_all(qt_app)

    assert qt_app.any_project_dirty() == unsaved
    assert [c.status for c in qt_app.document.clips] == statuses


def test_the_expected_side_is_the_segments_spoken_text(qt_app, heard):
    # The lexicon is already in `Segment.text`, so a rewrite isn't a mismatch.
    qt_app.document.text = "Dr. Smith arrived."
    qt_app.editor.load_text(qt_app.document.text)
    clip = _clip(qt_app, 0, 18, pieces=["Doctor Smith arrived."])
    heard.say(clip.segments[0], "doctor smith arrived")

    _proof_all(qt_app)

    assert qt_app.proof_entry(clip)["ratio"] == 1.0


def test_the_timeline_marks_the_flagged_block(qt_app, heard, qtbot):
    good, bad = _two_clips(qt_app, heard)
    qt_app.show()
    qtbot.waitExposed(qt_app)

    _proof_all(qt_app)

    assert _block(qt_app, bad).flagged is True and _block(qt_app, good).flagged is False


def test_a_regenerate_drops_the_stale_result(qt_app, heard, qtbot):
    good, bad = _two_clips(qt_app, heard)
    _proof_all(qt_app)
    dock = _shown_dock(qt_app, qtbot)
    assert dock.row_clip_ids() == [bad.id]

    # What a regenerate stores: new segments under a new key (a new take).
    bad.segments = build_segments_from_results(
        "take2", [{"text": bad.segments[0].text, "path": bad.segments[0].audio_path, "duration": 0.1,
                   "cache_key": "take2"}])
    qt_app.refresh_timeline()

    assert qt_app.proof_entry(bad) is None and not qt_app.is_clip_flagged(bad)
    assert dock.row_clip_ids() == []
    assert _block(qt_app, bad).flagged is False
    # The next write drops the entry from session.json too.
    qt_app.mark_proof_ok(good.id)
    assert bad.id not in project_io.read_session(qt_app.project_dir).get("proof", {})


def test_a_clip_regenerated_while_the_proof_runs_gets_no_result(qt_app, heard):
    good, bad = _two_clips(qt_app, heard)

    def regenerate(path):
        if path == bad.segments[0].audio_path:
            bad.segments[0].cache_key = "regenerated"

    heard.on_call = regenerate
    _proof_all(qt_app)

    assert qt_app.proof_entry(good) is not None and qt_app.proof_entry(bad) is None


def test_cancel_stops_between_segments(qt_app, heard):
    qt_app.document.text = "One two three four. Five six."
    qt_app.editor.load_text(qt_app.document.text)
    first = _clip(qt_app, 0, 19, pieces=["One two", "three four"])
    second = _clip(qt_app, 20, 29)
    for clip in (first, second):
        for segment in clip.segments:
            heard.say(segment, segment.text)
    # The user presses Cancel while the first file is being heard.
    heard.on_call = lambda _path: qt_app._proof_cancel.set()

    _proof_all(qt_app)

    assert len(heard.calls) == 1
    assert qt_app.proof_entry(first) is None and qt_app.proof_entry(second) is None
    assert qt_app.is_proofing is False


def test_cancel_keeps_the_clips_scored_before_it(qt_app, heard):
    good, bad = _two_clips(qt_app, heard)
    heard.on_call = lambda path: qt_app._proof_cancel.set() if path == bad.segments[0].audio_path else None

    _proof_all(qt_app)

    assert qt_app.proof_entry(good) is not None


def test_a_file_that_cannot_be_transcribed_skips_only_its_clip(qt_app, heard):
    good, bad = _two_clips(qt_app, heard)
    del heard.by_path[good.segments[0].audio_path]

    _proof_all(qt_app)

    assert qt_app.proof_entry(good) is None and qt_app.proof_entry(bad) is not None


def test_heard_words_are_reused_on_a_second_proof(qt_app, heard):
    _two_clips(qt_app, heard)
    _proof_all(qt_app)
    assert len(heard.calls) == 2

    _proof_all(qt_app)

    assert len(heard.calls) == 2


def test_word_alignment_hands_its_heard_words_to_the_proof(qt_app, heard):
    good, _bad = _two_clips(qt_app, heard)
    assert qt_app.schedule_word_alignment([good.id], force=True)
    qt_app.wait_for_word_alignment()
    assert len(heard.calls) == 1

    assert qt_app.run_proof([good.id])
    qt_app.wait_for_proof()

    assert len(heard.calls) == 1
    assert qt_app.proof_entry(good)["ratio"] == 1.0


def test_a_changed_file_is_transcribed_again(qt_app, heard):
    good, _bad = _two_clips(qt_app, heard)
    _proof_all(qt_app)
    path = good.segments[0].audio_path
    sf.write(path, np.full(4800, 0.2, dtype=np.float32), 24000)

    _proof_all(qt_app)

    assert heard.calls.count(path) == 2


def test_declining_the_whisper_download_runs_nothing(qt_app, heard, monkeypatch):
    _two_clips(qt_app, heard)
    monkeypatch.setattr(asr_prompt, "confirm_whisper_download", lambda parent: (asr_prompt.CANCEL, False))
    ids, _ = qt_app.clips_for_proof_scope(SCOPE_ALL)

    assert qt_app.run_proof(ids) is False
    assert heard.calls == [] and qt_app.is_proofing is False


def test_a_proof_with_every_file_already_heard_does_not_ask_about_the_download(qt_app, heard, monkeypatch):
    _two_clips(qt_app, heard)
    _proof_all(qt_app)
    monkeypatch.setattr(asr_prompt, "confirm_whisper_download",
                        lambda parent: pytest.fail("asked although nothing needs the model"))

    _proof_all(qt_app)


def test_cancel_with_nothing_running_does_nothing(qt_app):
    qt_app.cancel_proof()
    assert qt_app.is_proofing is False


# -- scopes -------------------------------------------------------------------------------------


def test_scopes_all_and_selection(qt_app, heard):
    good, bad = _two_clips(qt_app, heard)
    assert set(qt_app.clips_for_proof_scope(SCOPE_ALL)[0]) == {good.id, bad.id}

    ids, message = qt_app.clips_for_proof_scope(SCOPE_SELECTION)
    assert ids == [] and message
    qt_app.selection.select_clip(bad.id)
    assert qt_app.clips_for_proof_scope(SCOPE_SELECTION)[0] == [bad.id]

    # At the root with no subproject block selected there is no subproject.
    qt_app.selection.clear()
    ids, message = qt_app.clips_for_proof_scope(SCOPE_SUBPROJECT)
    assert ids == [] and message


def test_this_subproject_proofs_the_selected_subproject_blocks_clips(qt_app):
    qt_app.document.text = "Intro. Chapter text. Outro."
    qt_app.editor.load_text(qt_app.document.text)
    intro = _clip(qt_app, 0, 6)
    chapter = _clip(qt_app, 7, 20)
    child = qt_app.new_subproject(7, 20, title="Chapter 1")
    qt_app.selection.select_clip(child.clip_id)

    ids, _ = qt_app.clips_for_proof_scope(SCOPE_SUBPROJECT)

    assert ids == [chapter.id] and intro.id not in ids


def test_a_clip_with_missing_audio_is_not_proofable(qt_app, heard):
    good, bad = _two_clips(qt_app, heard)
    os.remove(bad.segments[0].audio_path)
    assert qt_app.proofable(good) and not qt_app.proofable(bad)
    assert qt_app.clips_for_proof_scope(SCOPE_ALL)[0] == [good.id]


# -- the dock ------------------------------------------------------------------------------------


def test_the_dock_is_registered_and_tabbed_with_the_settings_docks(qt_app):
    dock = qt_app.proof_dock
    assert dock.objectName() == "dock_proof" and dock.windowTitle() == "Proof"
    assert dock in qt_app._all_docks()
    assert dock in qt_app.tabifiedDockWidgets(qt_app.settings_dock)


def test_the_list_shows_flagged_clips_worst_first_with_the_first_issue(qt_app, heard, qtbot):
    qt_app.document.text = "One two three four. Five six seven eight. Nine ten eleven twelve."
    qt_app.editor.load_text(qt_app.document.text)
    a = _clip(qt_app, 0, 19)
    b = _clip(qt_app, 20, 41)
    c = _clip(qt_app, 42, 65)
    heard.say(a.segments[0], "one two three")  # one word dropped
    heard.say(b.segments[0], "five")  # most dropped
    heard.say(c.segments[0], "nine ten eleven twelve")
    _proof_all(qt_app)

    dock = _shown_dock(qt_app, qtbot)

    assert dock.row_clip_ids() == [b.id, a.id]
    second = dock.tree.topLevelItem(1)
    assert second.text(2) == "dropped: 'four'"
    assert second.text(1).endswith("%")
    assert dock.footer.text() == "2 flagged of 3 proofed."


def test_the_threshold_decides_what_is_flagged_and_is_a_saved_setting(qt_app, heard, qtbot):
    _good, bad = _two_clips(qt_app, heard)
    _proof_all(qt_app)
    dock = _shown_dock(qt_app, qtbot)
    assert dock.row_clip_ids() == [bad.id] and qt_app.settings["proof_threshold"] == 0.92

    dock.threshold_spin.setValue(0.5)
    dock.rebuild()
    assert dock.row_clip_ids() == []
    assert qt_app.settings["proof_threshold"] == 0.5
    assert _block(qt_app, bad).flagged is False

    dock.threshold_spin.setValue(1.0)
    dock.rebuild()
    assert dock.row_clip_ids() == [bad.id]  # 1.00 flags every clip that isn't word for word


def test_a_junk_threshold_setting_falls_back(qt_app):
    qt_app.settings["proof_threshold"] = "high"
    assert qt_app.proof_threshold() == proof.DEFAULT_THRESHOLD


def test_clicking_a_row_selects_the_clip_and_seeks(qt_app, heard, qtbot, monkeypatch):
    _good, bad = _two_clips(qt_app, heard)
    _proof_all(qt_app)
    dock = _shown_dock(qt_app, qtbot)
    placed = qt_app.build_arrangement(qt_app.level).by_clip_id()[bad.id]
    seeks = []
    monkeypatch.setattr(qt_app.transport, "seek", lambda seconds: seeks.append(seconds))

    dock.tree.itemClicked.emit(dock.row_item(bad.id), 0)

    assert qt_app.selection.selected_clip_id == bad.id
    assert seeks == [placed.start_s] and seeks[0] > 0
    assert qt_app.editor.textCursor().position() == qt_app.document.clip_extent(bad.id)[0]


def test_mark_ok_takes_the_clip_off_the_list(qt_app, heard, qtbot):
    _good, bad = _two_clips(qt_app, heard)
    _proof_all(qt_app)
    dock = _shown_dock(qt_app, qtbot)
    unsaved = qt_app.any_project_dirty()
    dock.tree.setCurrentItem(dock.row_item(bad.id))

    assert dock.mark_current_ok()

    assert dock.row_clip_ids() == [] and not qt_app.is_clip_flagged(bad)
    assert project_io.read_session(qt_app.project_dir)["proof"][bad.id]["ok"] is True
    assert _block(qt_app, bad).flagged is False
    assert qt_app.any_project_dirty() == unsaved


def test_a_regenerate_brings_a_marked_ok_clip_back_to_be_proofed_again(qt_app, heard, qtbot):
    _good, bad = _two_clips(qt_app, heard)
    _proof_all(qt_app)
    qt_app.mark_proof_ok(bad.id)

    bad.segments[0].cache_key = "take2"
    heard.say(bad.segments[0], "it cold")
    _proof_all(qt_app)

    assert qt_app.proof_entry(bad)["ok"] is False and qt_app.is_clip_flagged(bad)


def test_needs_rewrite_sets_the_status_and_undo_restores_it(qt_app, heard, qtbot):
    _good, bad = _two_clips(qt_app, heard)
    _proof_all(qt_app)
    dock = _shown_dock(qt_app, qtbot)
    before = bad.status
    dock.tree.setCurrentItem(dock.row_item(bad.id))

    assert dock.mark_current_needs_rewrite()
    assert qt_app.document.get_clip(bad.id).status == "needs_rewrite"
    assert dock.row_clip_ids() == []

    qt_app.undo()
    qt_app.flush_updates()
    qt_app.refresh_timeline()

    assert qt_app.document.get_clip(bad.id).status == before
    assert dock.row_clip_ids() == [bad.id]


def test_regenerate_sends_the_clip_to_generate(qt_app, heard, qtbot, monkeypatch):
    _good, bad = _two_clips(qt_app, heard)
    _proof_all(qt_app)
    dock = _shown_dock(qt_app, qtbot)
    sent = []
    monkeypatch.setattr(qt_app, "generate_clip", lambda clip_id: sent.append(clip_id))
    dock.tree.setCurrentItem(dock.row_item(bad.id))

    dock.regenerate_current()

    assert sent == [bad.id]


def test_the_buttons_need_a_row(qt_app, qtbot):
    dock = _shown_dock(qt_app, qtbot)
    assert not dock.regenerate_button.isEnabled() and not dock.ok_button.isEnabled()
    assert not dock.rewrite_button.isEnabled()
    assert dock.mark_current_ok() is False and dock.mark_current_needs_rewrite() is False


def test_the_run_button_proofs_the_scope_and_says_why_when_empty(qt_app, heard, qtbot):
    dock = _shown_dock(qt_app, qtbot)
    dock.scope_combo.setCurrentIndex(dock.scope_combo.findData(SCOPE_SELECTION))
    assert dock.run() is False

    _two_clips(qt_app, heard)
    dock.scope_combo.setCurrentIndex(dock.scope_combo.findData(SCOPE_ALL))
    assert dock.run() is True
    assert dock.run_button.text() == "Cancel"
    qt_app.wait_for_proof()
    dock.rebuild()
    assert dock.run_button.text() == "Run proof"
    assert len(dock.row_clip_ids()) == 1


def test_a_hidden_dock_builds_nothing_until_shown(qt_app, heard):
    _two_clips(qt_app, heard)
    dock = qt_app.proof_dock
    dock.hide()
    _proof_all(qt_app)
    assert dock.row_clip_ids() == []

    qt_app.show()
    dock.show()
    dock._on_visibility_changed(True)
    assert len(dock.row_clip_ids()) == 1


# -- session.json ----------------------------------------------------------------------------


def _saved(qt_app, tmp_path):
    qt_app.save_project_as(str(tmp_path / "story"))
    qt_app.wait_for_project_io()
    assert not qt_app.is_project_dirty()


def test_results_survive_a_save_and_a_reopen_through_session_json(qt_app, heard, tmp_path):
    good, bad = _two_clips(qt_app, heard)
    _proof_all(qt_app)
    _saved(qt_app, tmp_path)
    assert bad.id in project_io.read_session(qt_app.project_dir)["proof"]
    project_dir = qt_app.project_dir
    qt_app.close()

    second = qt_app_module.QtTTSApp()
    try:
        second.wait_for_project_io()
        assert second.project_dir == project_dir
        assert second.is_clip_flagged(second.document.get_clip(bad.id))
        assert second.proof_entry(second.document.get_clip(good.id))["ratio"] == 1.0
    finally:
        second.close()


def test_results_are_never_bundled(qt_app, heard, tmp_path):
    _two_clips(qt_app, heard)
    _proof_all(qt_app)
    _saved(qt_app, tmp_path)
    with zipfile.ZipFile(qt_app.project_path) as zf:
        assert "session.json" not in zf.namelist()
        assert b"segment_keys" not in b"".join(zf.read(n) for n in zf.namelist() if n.endswith(".json"))


def test_carry_over_only_applies_to_the_same_source_file(qt_app, heard, tmp_path):
    _two_clips(qt_app, heard)
    _proof_all(qt_app)
    _saved(qt_app, tmp_path)
    project_dir = qt_app.project_dir
    previous = project_io.read_session(project_dir)
    entries = previous["proof"]

    # Open rewrote the session without `proof`: the same file gets it back.
    fresh = {k: v for k, v in previous.items() if k != "proof"}
    project_io.write_session(project_dir, fresh)
    qt_app.carry_proof_over(previous, project_dir)
    assert project_io.read_session(project_dir)["proof"] == entries

    # A different source file does not.
    other = {**fresh, "source_path": str(tmp_path / "somewhere_else.tbaw")}
    project_io.write_session(project_dir, other)
    qt_app.carry_proof_over(previous, project_dir)
    assert "proof" not in project_io.read_session(project_dir)


@pytest.mark.parametrize("junk", ["text", 5, None, [], {"c": 1}, {"c": {"ratio": "x", "segment_keys": []}}])
def test_a_junk_proof_value_opens_without_error(qt_app, heard, tmp_path, junk):
    good, _bad = _two_clips(qt_app, heard)
    _saved(qt_app, tmp_path)
    project_dir = qt_app.project_dir
    qt_app.close()
    session = project_io.read_session(project_dir)
    session["proof"] = junk
    project_io.write_session(project_dir, session)

    second = qt_app_module.QtTTSApp()
    try:
        second.wait_for_project_io()
        assert second.proof_store() == {}
        assert second.is_clip_flagged(second.document.get_clip(good.id)) is False
    finally:
        second.close()


# -- the timeline mark -------------------------------------------------------------------------


def test_a_flagged_block_paints_a_corner_mark_in_the_theme_color(qt_app):
    from PySide6.QtGui import QColor, QImage, QPainter

    from kokoro_gui.qt import theme
    from kokoro_gui.qt.timeline_view import ClipBlockItem

    def paint(flagged):
        block = ClipBlockItem()
        block.set_geometry(0, 0, 200, 60)
        block.set_flagged(flagged)
        image = QImage(200, 60, QImage.Format.Format_ARGB32)
        image.fill(QColor("white"))
        painter = QPainter(image)
        block.paint(painter, None)
        painter.end()
        return image

    token = QColor(theme.current().proof_flag)
    assert paint(True).pixelColor(3, 56).name() == token.name()
    assert paint(False).pixelColor(3, 56).name() != token.name()
    assert theme.LIGHT.proof_flag != theme.DARK.proof_flag
