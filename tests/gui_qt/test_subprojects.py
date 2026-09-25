"""Subprojects (phase 4, grill NP1-NP8): a `.tbaw` placed on another one's
timeline as a nested clip. New Subproject, the embedded child in the
parent's bundle, lazy opening, and Save carrying the child."""
import os
import zipfile

import numpy as np
import soundfile as sf

from kokoro_gui.daw.dirty import build_segments_from_results
from kokoro_gui.daw.models import PLACEHOLDER
from kokoro_gui.qt import project as project_io


def _generated_clip(qt_app, start, end):
    """A clean clip over `[start, end)` with one real wav in the project dir."""
    document = qt_app.document
    character = document.characters[0]
    clip = document.assign_character_to_range(start, end, character.id)
    text = document.clip_text(clip)
    key = document.segment_key_fn(text, clip)
    generated = os.path.join(qt_app.project_dir, "audio", "generated")
    os.makedirs(generated, exist_ok=True)
    path = os.path.join(generated, f"{key}_0.wav")
    sf.write(path, np.full(2400, 0.2, dtype=np.float32), 24000)
    clip.segments = build_segments_from_results(key, [{"text": text, "path": path, "duration": 0.1,
                                                       "cache_key": key,
                                                       "engine_version": qt_app.backend.engine_version()}])
    return clip


def _book(qt_app):
    """"Intro. Chapter text. Outro." with the middle clip moved into an
    embedded subproject."""
    qt_app.document.text = "Intro. Chapter text. Outro."
    qt_app.editor.load_text(qt_app.document.text)
    intro = _generated_clip(qt_app, 0, 6)
    chapter = _generated_clip(qt_app, 7, 20)
    child = qt_app.new_subproject(7, 20, title="Chapter 1")
    return intro, chapter, child


def test_new_subproject_moves_the_selection_into_an_embedded_child(qt_app):
    intro, chapter, child = _book(qt_app)
    parent = qt_app.document

    nested = parent.get_clip(child.clip_id)
    assert nested.is_nested and nested.child == {"kind": "embedded", "id": child.project_id}
    assert parent.text == "Intro. Chapter 1 Outro."  # the placeholder took "Chapter text."
    assert [r.kind for r in parent.runs if r.clip_id == nested.id] == [PLACEHOLDER]
    assert parent.get_clip(chapter.id) is None
    assert parent.get_clip(intro.id) is not None
    assert parent.get_track(nested.track_id).role == "subprojects"
    assert qt_app.editor.toPlainText() == parent.text

    # The child holds the moved clip, its audio copied into the child's dir.
    moved = child.document.get_clip(chapter.id)
    assert child.document.text == "Chapter text."
    assert moved.segments[0].audio_path.startswith(child.project_dir)
    assert os.path.isfile(moved.segments[0].audio_path)
    assert child.document.get_character(moved.character_id) is not None
    assert child.project_settings["title"] == "Chapter 1"
    assert child.parent_id == qt_app.root.project_id
    assert qt_app.children[child.project_id] is child
    assert child.dirty


def test_new_subproject_is_one_undo_step_on_the_parent(qt_app):
    _intro, chapter, child = _book(qt_app)
    qt_app.document.undo_stack.undo()
    assert qt_app.document.text == "Intro. Chapter text. Outro."
    assert qt_app.document.get_clip(chapter.id) is not None
    assert qt_app.document.get_clip(child.clip_id) is None
    qt_app.document.undo_stack.redo()
    assert qt_app.document.get_clip(child.clip_id).is_nested


def test_a_selection_over_a_subproject_is_refused(qt_app):
    _book(qt_app)
    assert qt_app.new_subproject(0, len(qt_app.document.text)) is None


def test_an_empty_subproject_goes_in_without_a_selection(qt_app):
    qt_app.document.text = "Hello."
    child = qt_app.new_subproject(6)
    assert child.document.text == ""
    assert qt_app.document.text == "Hello.Subproject 1"


def test_parent_with_an_embedded_child_round_trips(qt_app, tmp_path):
    _intro, chapter, child = _book(qt_app)
    child_id = child.project_id
    clip_id = child.clip_id
    qt_app.save_project_as(str(tmp_path / "book"))
    qt_app.wait_for_project_io()
    path = qt_app.project_path

    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        entry = zf.getinfo(f"projects/{child_id}.tbaw")
        manifest = __import__("json").loads(zf.read("manifest.json"))
    assert entry.compress_type == zipfile.ZIP_STORED
    assert manifest["requires"] == ["nested"]
    assert manifest["includes"]["projects"] == [child_id]
    assert not child.dirty and not qt_app.any_project_dirty()
    assert any(n.startswith("audio/generated/") for n in names)  # the intro's audio
    # The child's own audio rides inside its bundle, not the parent's.
    assert not any(chapter.segments and os.path.basename(chapter.segments[0].audio_path) in n
                   for n in names if n.startswith("audio/"))

    session = project_io.read_session(child.project_dir)
    assert session["source_path"] == f"{path}#{child_id}"

    # Reopen: the child isn't opened with the parent.
    qt_app.new_project()
    qt_app.open_project(path)
    qt_app.wait_for_project_io()
    assert qt_app.project_path == path
    assert qt_app.children == {}
    nested = qt_app.document.get_clip(clip_id)
    assert nested.is_nested
    assert os.path.isfile(os.path.join(qt_app.project_dir, "projects", f"{child_id}.tbaw"))

    # Opening it: from the embedded bundle, into a dir keyed by its id.
    reopened = qt_app.open_child(nested)
    qt_app.wait_for_project_io()
    reopened = reopened or qt_app.children[child_id]
    assert reopened.document.text == "Chapter text."
    assert os.path.basename(reopened.project_dir).startswith(child_id)
    moved = reopened.document.get_clip(chapter.id)
    assert moved.segments[0].audio_path.startswith(reopened.project_dir)
    assert os.path.isfile(moved.segments[0].audio_path)
    assert not reopened.dirty
    assert project_io.read_session(reopened.project_dir)["source_path"] == f"{path}#{child_id}"


def test_a_child_edit_makes_the_root_dirty_and_saves_into_the_parent(qt_app, tmp_path):
    _intro, _chapter, child = _book(qt_app)
    qt_app.save_project_as(str(tmp_path / "book"))
    qt_app.wait_for_project_io()
    assert not qt_app.any_project_dirty()

    child.document.settings["gap_s"] = 0.5
    qt_app.save_settings()
    assert child.dirty and not qt_app.root.dirty
    assert qt_app.is_project_dirty()
    assert qt_app.windowTitle().startswith("book*")

    qt_app.save_project()
    qt_app.wait_for_project_io()
    assert not child.dirty
    loaded = project_io.load_project(os.path.join(qt_app.project_dir, "projects", f"{child.project_id}.tbaw"))
    assert loaded.document.settings["gap_s"] == 0.5
    with zipfile.ZipFile(qt_app.project_path) as zf:
        inner = zf.read(f"projects/{child.project_id}.tbaw")
    with open(os.path.join(qt_app.project_dir, "projects", f"{child.project_id}.tbaw"), "rb") as f:
        assert f.read() == inner


def test_a_bundle_needing_nested_is_refused_where_it_isnt_supported(qt_app, tmp_path, monkeypatch):
    _book(qt_app)
    qt_app.save_project_as(str(tmp_path / "book"))
    qt_app.wait_for_project_io()
    monkeypatch.setattr(project_io, "SUPPORTED_FEATURES", frozenset({"takes"}))
    import pytest

    with pytest.raises(project_io.ProjectError, match="nested"):
        project_io.inspect_bundle(qt_app.project_path)


def test_discarding_the_root_drops_the_childs_dir(qt_app):
    _intro, _chapter, child = _book(qt_app)
    child_dir = child.project_dir
    qt_app.new_project()  # the fixture answers Discard
    assert not os.path.isdir(child_dir)
    assert qt_app.children == {}


def test_close_with_only_a_dirty_child_prompts_once_naming_it(qt_app, tmp_path, monkeypatch):
    _intro, _chapter, child = _book(qt_app)
    qt_app.save_project_as(str(tmp_path / "book"))
    qt_app.wait_for_project_io()
    child.document.settings["gap_s"] = 0.7
    qt_app.save_settings()
    asked = []
    monkeypatch.setattr(type(qt_app), "_ask_close_choice",
                        lambda self: (asked.append(self._unsaved_changes_text()), "discard")[1])
    qt_app.new_project()
    assert len(asked) == 1
    assert "Chapter 1" in asked[0]


# -- step 4: the selection points the docks at a subproject (NP1) -----------------


def test_selecting_the_nested_block_points_the_docks_at_the_child(qt_app):
    _intro, chapter, child = _book(qt_app)
    root = qt_app.root

    qt_app.selection.select_clip(child.clip_id)

    assert qt_app.focus is child
    assert qt_app.level is root
    assert qt_app.document is child.document
    assert qt_app.selection.project_id == child.project_id
    assert qt_app.editor.toPlainText() == "Chapter text."
    assert qt_app.scope_text() == "Subproject: Chapter 1"
    assert not qt_app.transcript_dock.scope_bar.isHidden()
    assert qt_app.fx_dock.scope_label.text().startswith("Subproject: Chapter 1")
    assert qt_app.settings_dock.subproject_label.text() == "Subproject: Chapter 1"
    assert qt_app.settings_dock.scope_fields.widgets["title"].text() == "Chapter 1"
    # The timeline still shows the parent.
    assert chapter.id not in qt_app.timeline_dock.timeline_view._blocks_by_clip_id
    assert child.clip_id in qt_app.timeline_dock.timeline_view._blocks_by_clip_id

    # Typing edits the child; undo is the child's.
    qt_app.editor.textCursor().insertText("Hi ")
    assert child.document.text.startswith("Hi ")
    assert root.document.text == "Intro. Chapter 1 Outro."

    # Back, or selecting a parent clip, returns the docks to the level.
    qt_app.transcript_dock.scope_back_btn.click()
    assert qt_app.focus is root
    assert qt_app.editor.toPlainText() == root.document.text
    assert qt_app.transcript_dock.scope_bar.isHidden()


def test_selecting_a_parent_clip_moves_the_focus_back(qt_app):
    intro, _chapter, child = _book(qt_app)
    qt_app.selection.select_clip(child.clip_id)
    assert qt_app.focus is child
    qt_app.selection.select_clip(intro.id)
    assert qt_app.focus is qt_app.root


def test_a_selection_inside_the_child_keeps_the_focus_there(qt_app):
    _intro, chapter, child = _book(qt_app)
    qt_app.selection.select_clip(child.clip_id)
    qt_app.selection.select_clip(chapter.id)  # the child's own clip, from its transcript
    assert qt_app.focus is child
    qt_app.selection.clear()
    assert qt_app.focus is child


def test_renaming_a_subproject_rewrites_its_placeholder(qt_app):
    _intro, _chapter, child = _book(qt_app)
    qt_app.selection.select_clip(child.clip_id)
    field = qt_app.settings_dock.scope_fields.widgets["title"]
    field.setText("The Beginning")
    field.editingFinished.emit()
    assert child.project_settings["title"] == "The Beginning"
    assert qt_app.root.document.text == "Intro. The Beginning Outro."
    assert qt_app.scope_text() == "Subproject: The Beginning"


def test_the_placeholder_line_is_read_only(qt_app, qtbot):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QTextCursor

    _intro, _chapter, child = _book(qt_app)
    editor = qt_app.editor
    start, end = qt_app.document.clip_extent(child.clip_id)
    assert editor.edit_touches_placeholder(start + 2, start + 2, inserting=True)
    assert editor.edit_touches_placeholder(start - 1, start + 3)
    assert not editor.edit_touches_placeholder(start, end)  # the whole line may go
    assert not editor.edit_touches_placeholder(0, 3)

    # A backspace over a selection reaching into the placeholder is refused.
    editor._updating_from_model = True
    cursor = QTextCursor(editor.document())
    cursor.setPosition(start - 2)
    cursor.setPosition(start + 3, QTextCursor.MoveMode.KeepAnchor)
    editor.setTextCursor(cursor)
    editor._updating_from_model = False
    qtbot.keyClick(editor, Qt.Key.Key_Backspace)
    assert qt_app.root.document.text == "Intro. Chapter 1 Outro."

    # Typing right after the placeholder doesn't join it.
    qt_app.root.document.replace_text(end, 0, 1, qt_app.root.document.text[:end] + "X" +
                                      qt_app.root.document.text[end:])
    assert qt_app.root.document.clip_text(qt_app.root.document.get_clip(child.clip_id)) == "Chapter 1"


# -- step 5: the child's mixdown and staleness (NP2) ---------------------------------


def _render(qt_app, child):
    assert qt_app.render_subproject(child)
    qt_app.wait_for_project_io()


def test_a_new_child_is_stale_until_rendered(qt_app):
    _intro, chapter, child = _book(qt_app)
    nested = qt_app.document.get_clip(child.clip_id)
    assert child.document.dirty_clips() == []  # the moved clip kept its audio
    assert qt_app.nested_state(nested) == "stale"
    assert nested in qt_app.document.dirty_clips()
    assert qt_app.clip_duration_s(nested) is None

    _render(qt_app, child)

    info = project_io.read_mixdown_info(child.project_dir)
    assert info["file"].startswith(child.project_dir)
    assert info["digest"] == child.digest
    assert info["duration_s"] > 0
    assert qt_app.nested_state(nested) == "ok"
    assert nested not in qt_app.document.dirty_clips()
    assert abs(qt_app.clip_duration_s(nested) - info["duration_s"]) < 1e-6
    samples, rate = qt_app.rendered_clip_samples(nested)
    assert len(samples) > 0 and rate == qt_app.project_sample_rate()
    block = qt_app.timeline_dock.timeline_view._blocks_by_clip_id[nested.id]
    assert block.nested_state == "ok"
    assert not block.estimated


def test_editing_the_child_makes_its_block_stale_again(qt_app):
    _intro, _chapter, child = _book(qt_app)
    _render(qt_app, child)
    nested = qt_app.document.get_clip(child.clip_id)
    child.document.settings["gap_s"] = 0.9
    qt_app.save_settings()
    assert qt_app.nested_state(nested) == "stale"


def test_the_transport_plays_the_childs_mixdown(qt_app, monkeypatch):
    intro, _chapter, child = _book(qt_app)
    _render(qt_app, child)
    loaded = []
    real_load = qt_app.transport.load
    monkeypatch.setattr(qt_app.transport, "load", lambda schedule, **k: (loaded.append(schedule),
                                                                          real_load(schedule, **k)))
    qt_app._rebuild_transport_schedule()
    paths = {c.clip_id: c.path for c in loaded[-1]}
    nested = qt_app.document.get_clip(child.clip_id)
    assert paths[nested.id] == project_io.read_mixdown_info(child.project_dir)["file"]
    assert intro.id in paths


def test_the_parent_applies_only_fx_set_on_the_nested_clip(qt_app):
    _intro, _chapter, child = _book(qt_app)
    nested = qt_app.document.get_clip(child.clip_id)
    qt_app.settings_dock.volume_spin.setValue(1.5)  # a project default, already in the mixdown
    assert qt_app.post_config_for_clip(nested) == {}
    nested.fx_override = {"reverb_enabled": True}
    config = qt_app.post_config_for_clip(nested)
    assert config.get("reverb_enabled") is True and config.get("apply_fx") is True
    assert "volume" not in config


def test_the_gutter_button_generates_and_renders_a_stale_subproject(qt_app):
    _intro, _chapter, child = _book(qt_app)
    nested = qt_app.document.get_clip(child.clip_id)
    qt_app.generate_clip(nested.id)
    qt_app.wait_for_project_io()
    assert qt_app.nested_state(nested) == "ok"


def test_generate_renders_stale_subprojects_after_the_parents_clips(qt_app):
    _intro, _chapter, child = _book(qt_app)
    nested = qt_app.document.get_clip(child.clip_id)
    qt_app.on_generate_clicked()  # the parent's own clips are clean
    qt_app.wait_for_project_io()
    assert qt_app.nested_state(nested) == "ok"


def test_a_child_with_stale_clips_generates_them_before_rendering(qt_app):
    _intro, chapter, child = _book(qt_app)
    moved = child.document.get_clip(chapter.id)
    moved.segments = []  # stale
    nested = qt_app.document.get_clip(child.clip_id)
    assert qt_app.render_subproject(child) is False

    qt_app.generate_subproject(nested)
    # The child's batch went to its engine; the render waits for it.
    assert qt_app.engine.generate_dirty_clips.called
    group = qt_app.engine.generate_dirty_clips.call_args[0][0]
    assert [cid for cid, _t, _c in group] == [chapter.id]
    key = child.document.segment_key_fn("Chapter text.", moved)
    path = os.path.join(child.project_dir, "audio", "generated", f"{key}_0.wav")
    sf.write(path, np.full(2400, 0.1, dtype=np.float32), 24000)
    qt_app.engine.worker.run_coro.return_value.set_result([{
        "clip_id": chapter.id, "success": True, "error": "", "cancelled": False,
        "results": [{"path": path, "text": "Chapter text.", "duration": 0.1, "seg_idx": 0,
                     "cache_key": key, "take": 0, "engine_version": qt_app.backend.engine_version()}],
    }])
    qt_app.wait_for_project_io()
    assert moved.segments and moved.segments[0].audio_path == path
    assert project_io.read_mixdown_info(child.project_dir) is not None


def test_export_mixes_the_childs_mixdown(qt_app, tmp_path):
    from kokoro_gui.qt.docks.export_dialog import run_export

    _intro, _chapter, child = _book(qt_app)
    _render(qt_app, child)
    values = {"out_dir": str(tmp_path / "out"), "filename": "book", "format": "wav", "srt": False,
              "keep_clip_files": False, "channels": 1}
    assert run_export(qt_app, values)
    future = qt_app.engine.worker.run_coro.call_args[0][0]
    import asyncio

    result = asyncio.run(future)
    data, _rate = sf.read(result.audio_path)
    assert result.skipped_clip_ids == []
    assert np.abs(data).max() > 0.1


# -- step 6: entering a subproject, the breadcrumb (NP5, NP6) ------------------------


def test_entering_a_subproject_shows_it_in_the_timeline(qt_app):
    _intro, chapter, child = _book(qt_app)
    qt_app.on_subproject_action(child.clip_id, "enter")

    assert qt_app.level is child and qt_app.focus is child
    view = qt_app.timeline_dock.timeline_view
    assert chapter.id in view._blocks_by_clip_id
    assert child.clip_id not in view._blocks_by_clip_id
    buttons = qt_app.timeline_dock.breadcrumb_buttons
    assert [b.text() for b in buttons] == ["Untitled", "Chapter 1"]
    assert not qt_app.timeline_dock.breadcrumb.isHidden()
    assert qt_app.scope_text() is None

    buttons[0].click()
    assert qt_app.level is qt_app.root and qt_app.focus is qt_app.root
    assert child.clip_id in qt_app.timeline_dock.timeline_view._blocks_by_clip_id


def test_double_clicking_a_nested_block_enters_it(qt_app, qtbot):
    from PySide6.QtCore import Qt

    _intro, _chapter, child = _book(qt_app)
    view = qt_app.timeline_dock.timeline_view
    block = view._blocks_by_clip_id[child.clip_id]
    pos = view.mapFromScene(block.mapToScene(5, 10))
    qtbot.mouseDClick(view.viewport(), Qt.MouseButton.LeftButton, pos=pos)
    assert qt_app.level is child


def test_breadcrumb_hidden_at_a_root_without_subprojects(qt_app):
    qt_app.refresh_timeline()
    assert qt_app.timeline_dock.breadcrumb.isHidden()


# -- step 7: linked children, relink, detach, embed, remove (NP4) -------------------


def _saved_project(qt_app, tmp_path, name, text):
    """A separate .tbaw on disk (made by this app, then put away)."""
    qt_app.document.text = text
    qt_app.editor.load_text(text)
    qt_app.save_project_as(str(tmp_path / name))
    qt_app.wait_for_project_io()
    path, project_id = qt_app.project_path, qt_app.project_id
    qt_app.new_project()
    return path, project_id


def test_add_subproject_links_an_existing_file(qt_app, tmp_path):
    other_path, other_id = _saved_project(qt_app, tmp_path, "episode", "Guest segment.")
    qt_app.document.text = "Show intro."
    qt_app.editor.load_text(qt_app.document.text)
    qt_app.save_project_as(str(tmp_path / "show"))
    qt_app.wait_for_project_io()

    child = qt_app.add_subproject(other_path)

    nested = qt_app.document.get_clip(child.clip_id)
    assert nested.child == {"kind": "linked", "id": other_id, "path": "episode.tbaw"}
    assert child.kind == "linked" and child.path == other_path
    assert child.document.text == "Guest segment."
    assert qt_app.document.clip_text(nested) == "episode"

    # Saving the parent doesn't copy a linked child into it.
    qt_app.save_project()
    qt_app.wait_for_project_io()
    with zipfile.ZipFile(qt_app.project_path) as zf:
        assert not any(n.startswith("projects/") for n in zf.namelist())
    # Adding the same project twice is refused.
    assert qt_app.add_subproject(other_path) is None


def test_a_missing_linked_file_is_missing_and_relinks(qt_app, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    other_path, other_id = _saved_project(qt_app, tmp_path, "episode", "Guest segment.")
    qt_app.save_project_as(str(tmp_path / "show"))
    qt_app.wait_for_project_io()
    child = qt_app.add_subproject(other_path)
    clip_id = child.clip_id
    qt_app.save_project()
    qt_app.wait_for_project_io()
    moved = str(tmp_path / "moved.tbaw")
    os.replace(other_path, moved)

    show = qt_app.project_path
    qt_app.new_project()
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.No))
    qt_app.open_project(show)
    qt_app.wait_for_project_io()
    nested = qt_app.document.get_clip(clip_id)
    assert qt_app.open_child(nested) is None
    assert qt_app.nested_state(nested) == "missing"
    assert qt_app.timeline_dock.timeline_view._blocks_by_clip_id[clip_id].nested_state == "missing"

    # A file with another project id is refused, with the id shown.
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: warned.append(a[2])))
    assert qt_app.relink_subproject(nested, show) is False
    assert qt_app.root.project_id in warned[0]

    assert qt_app.relink_subproject(nested, moved) is True
    assert nested.child["path"] == "moved.tbaw"
    assert qt_app.child_project(nested).document.text == "Guest segment."
    assert qt_app.nested_state(nested) != "missing"


def test_detach_then_embed_round_trip(qt_app, tmp_path):
    _intro, _chapter, child = _book(qt_app)
    qt_app.save_project_as(str(tmp_path / "book"))
    qt_app.wait_for_project_io()
    nested = qt_app.document.get_clip(child.clip_id)

    assert qt_app.detach_subproject(nested, str(tmp_path / "chapter1"))
    assert nested.child == {"kind": "linked", "id": child.project_id, "path": "chapter1.tbaw"}
    assert os.path.isfile(tmp_path / "chapter1.tbaw")
    assert child.kind == "linked"
    qt_app.document.undo_stack.undo()
    assert nested.child["kind"] == "embedded"
    qt_app.document.undo_stack.redo()

    assert qt_app.embed_subproject(nested)
    assert nested.child == {"kind": "embedded", "id": child.project_id}
    assert os.path.isfile(os.path.join(qt_app.project_dir, "projects", f"{child.project_id}.tbaw"))
    qt_app.save_project()
    qt_app.wait_for_project_io()
    with zipfile.ZipFile(qt_app.project_path) as zf:
        assert f"projects/{child.project_id}.tbaw" in zf.namelist()


def test_remove_takes_the_placeholder_out_in_one_undo_step(qt_app):
    _intro, _chapter, child = _book(qt_app)
    nested = qt_app.document.get_clip(child.clip_id)
    assert qt_app.remove_subproject(nested)
    assert qt_app.document.get_clip(child.clip_id) is None
    assert qt_app.document.text == "Intro.  Outro."
    assert child.project_id not in qt_app.children
    qt_app.document.undo_stack.undo()
    assert qt_app.document.get_clip(child.clip_id) is not None


def test_save_as_rebases_linked_paths(qt_app, tmp_path):
    other_path, _other_id = _saved_project(qt_app, tmp_path, "episode", "Guest.")
    qt_app.save_project_as(str(tmp_path / "show"))
    qt_app.wait_for_project_io()
    child = qt_app.add_subproject(other_path)
    nested = qt_app.document.get_clip(child.clip_id)
    os.makedirs(tmp_path / "sub")
    qt_app.save_project_as(str(tmp_path / "sub" / "show2"))
    qt_app.wait_for_project_io()
    assert nested.child["path"] == "../episode.tbaw"
