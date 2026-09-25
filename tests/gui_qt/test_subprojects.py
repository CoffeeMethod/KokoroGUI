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
