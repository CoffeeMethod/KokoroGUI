"""The three sequences in kokoro_gui/qt/project.py that the app and a headless
caller share: `begin_project_dir` (New), `open_action` (the recover, reuse or
extract choice of Open) and `save_project` (a synchronous Save). No Qt."""
import os
import zipfile

import pytest

from kokoro_gui.daw.models import Character, Clip, Document, Run, Segment
from kokoro_gui.qt import project as project_io


def _document(project_dir=None, text="hello"):
    character = Character.from_preset_dict("A", {"voice": "af_heart"})
    clip = Clip(character_id=character.id)
    if project_dir is not None:
        generated = os.path.join(project_dir, "audio", "generated")
        os.makedirs(generated, exist_ok=True)
        path = os.path.join(generated, "abc_0.wav")
        with open(path, "wb") as f:
            f.write(b"RIFF" + b"\0" * 60)
        clip.segments = [Segment(0, text, "abc", path, 1.5)]
    return Document(runs=[Run(text, clip.id, "generated")], clips=[clip], characters=[character], settings={})


def _record_calls(monkeypatch, *names):
    calls = []
    for name in names:
        real = getattr(project_io, name)

        def wrapped(*a, _name=name, _real=real, **k):
            calls.append(_name)
            return _real(*a, **k)

        monkeypatch.setattr(project_io, name, wrapped)
    return calls


def _disk_full(*_args, **_kwargs):
    raise OSError("disk full")


# -- begin_project_dir ---------------------------------------------------------------


@pytest.fixture
def begun(isolated_dirs):
    """`begin_project_dir`, with every lock it hands out released at the end."""
    locks = []

    def _begin(document, settings, **kwargs):
        result = project_io.begin_project_dir(document, settings, **kwargs)
        locks.append(result[2])
        return result

    yield _begin
    for lock in locks:
        lock.release()


def test_begin_creates_a_locked_dir_with_the_document_and_a_fresh_session(begun):
    document = _document()
    project_dir, project_id, lock = begun(document, {"title": "T"})

    assert project_dir == os.path.join(project_io.projects_root(), project_id)
    assert sorted(os.listdir(project_dir)) == ["audio", "document.json", "lock", "project.json", "session.json"]
    assert os.path.isdir(os.path.join(project_dir, "audio", "generated"))
    assert lock.held and lock.project_dir == project_dir
    assert project_io.is_locked(project_dir)
    assert project_io.read_session(project_dir) == {
        "source_path": None, "zip_size": None, "zip_mtime": None,
        "saved_digest": project_io.project_digest(document, {"title": "T"}, project_dir),
        "dirty": False, "asset_index": {},
    }
    loaded = project_io.load_project_dir(project_dir)
    assert loaded.document.text == "hello"
    assert loaded.project_settings == {"title": "T"}


def test_begin_makes_a_new_project_each_time(begun):
    first, second = begun(_document(), {}), begun(_document(), {})
    assert first[0] != second[0] and first[1] != second[1]


def test_begin_locks_before_it_writes(begun, monkeypatch):
    calls = _record_calls(monkeypatch, "create_project_dir", "autosave_to_dir", "write_session")
    seen = []

    def on_locked(project_dir, project_id, lock):
        calls.append("on_locked")
        seen.append((project_dir, project_id, lock, lock.held))
        assert not os.path.exists(os.path.join(project_dir, "document.json"))

    result = begun(_document(), {}, on_locked=on_locked)
    assert calls == ["create_project_dir", "on_locked", "autosave_to_dir", "write_session"]
    assert seen == [(*result, True)]


@pytest.mark.parametrize("step", ["autosave_to_dir", "write_session"])
def test_begin_releases_the_lock_when_a_write_fails(isolated_dirs, monkeypatch, step):
    monkeypatch.setattr(project_io, step, _disk_full)
    # The kept exception's frames hold the lock object, so only an explicit
    # release frees it here.
    with pytest.raises(OSError, match="disk full") as _kept:
        project_io.begin_project_dir(_document(), {})
    (name,) = os.listdir(project_io.projects_root())
    assert not project_io.is_locked(os.path.join(project_io.projects_root(), name))


def test_begin_leaves_the_lock_to_an_owner_that_took_it(isolated_dirs, monkeypatch):
    """The app keeps the dir it was handed when the first write fails."""
    monkeypatch.setattr(project_io, "autosave_to_dir", _disk_full)
    owned = []
    with pytest.raises(OSError, match="disk full"):
        project_io.begin_project_dir(_document(), {}, on_locked=lambda *a: owned.append(a))
    (_project_dir, _project_id, lock), = owned
    assert lock.held
    lock.release()


# -- save_project --------------------------------------------------------------------


def test_save_plans_writes_and_records_in_that_order(isolated_dirs, tmp_path, monkeypatch):
    project_dir, project_id = project_io.create_project_dir()
    document = _document(project_dir)
    path = str(tmp_path / "p.tbaw")
    calls = _record_calls(monkeypatch, "plan_save", "write_bundle", "record_save")

    saved = project_io.save_project(document, path, {"title": "T"}, project_dir, project_id)

    assert calls == ["plan_save", "write_bundle", "record_save"]
    assert saved.zip_size == os.path.getsize(path) and saved.zip_mtime == os.path.getmtime(path)
    assert saved.plan.path == os.path.abspath(path)
    assert saved.plan.manifest["project_id"] == project_id
    with zipfile.ZipFile(path) as zf:
        assert zf.namelist() == ["manifest.json", "document.json", "project.json", "audio/generated/abc_0.wav"]
        assert zf.read("document.json") == saved.plan.document_bytes
        assert zf.read("project.json") == saved.plan.project_bytes
    assert project_io.read_session(project_dir) == {
        "source_path": os.path.abspath(path), "zip_size": saved.zip_size, "zip_mtime": saved.zip_mtime,
        "saved_digest": project_io.project_digest(document, {"title": "T"}, project_dir),
        "dirty": False, "asset_index": saved.asset_index,
    }


def test_save_returns_the_plans_warnings(isolated_dirs, tmp_path):
    project_dir, project_id = project_io.create_project_dir()
    settings = {"video": {"path": "gone.mp4", "offset_s": 0.0}, "bundle": {"include_video": True}}
    saved = project_io.save_project(Document(), str(tmp_path / "p.tbaw"), settings, project_dir, project_id)
    assert any("reference video not found" in w for w in saved.warnings)
    clean = project_io.save_project(Document(), str(tmp_path / "q.tbaw"), {}, project_dir, project_id)
    assert clean.warnings == []


def test_save_keeps_the_previous_manifests_created_date(isolated_dirs, tmp_path):
    project_dir, project_id = project_io.create_project_dir()
    saved = project_io.save_project(_document(), str(tmp_path / "p.tbaw"), {}, project_dir, project_id,
                                    previous_manifest={"created": "2001-02-03T04:05:06"})
    assert saved.plan.manifest["created"] == "2001-02-03T04:05:06"


def test_a_failed_write_records_nothing(isolated_dirs, tmp_path, monkeypatch):
    project_dir, project_id = project_io.create_project_dir()
    project_io.write_session(project_dir, {"dirty": True, "saved_digest": "old"})
    monkeypatch.setattr(project_io, "write_bundle", _disk_full)
    with pytest.raises(OSError, match="disk full"):
        project_io.save_project(_document(), str(tmp_path / "p.tbaw"), {}, project_dir, project_id)
    assert project_io.read_session(project_dir) == {"dirty": True, "saved_digest": "old"}


def test_a_json_save_has_no_plan(isolated_dirs, tmp_path):
    saved = project_io.save_project(_document(), str(tmp_path / "p.json"), {})
    assert saved.plan is None and saved.warnings == []


# -- open_action ---------------------------------------------------------------------


@pytest.fixture
def clean_dir(isolated_dirs, tmp_path):
    """`(info, project_dir)`: a saved bundle and the working copy it came from."""
    project_dir, project_id = project_io.create_project_dir()
    document = _document(project_dir)
    project_io.autosave_to_dir(document, {}, project_dir)
    path = str(tmp_path / "p.tbaw")
    project_io.save_project(document, path, {}, project_dir, project_id)
    return project_io.inspect_bundle(path), project_dir


def _action(info, project_dir, dirty_children=(), **changes):
    session = {**project_io.read_session(project_dir), **changes}
    return project_io.open_action(session, info, project_dir, dirty_children)


def test_a_clean_dir_of_this_very_file_is_reused(clean_dir):
    info, project_dir = clean_dir
    assert _action(info, project_dir) == "reuse"


def test_a_dirty_session_asks(clean_dir):
    info, project_dir = clean_dir
    assert _action(info, project_dir, dirty=True) == "recover"
    # Even when the file changed since: the prompt says so.
    assert _action(info, project_dir, dirty=True, zip_size=1) == "recover"


def test_unsaved_subproject_edits_ask_even_when_the_root_is_clean(clean_dir, tmp_path):
    info, project_dir = clean_dir
    assert _action(info, project_dir, [str(tmp_path / "child")]) == "recover"
    assert _action(info, project_dir, []) == "reuse"


@pytest.mark.parametrize("changes", [{"zip_size": 1}, {"zip_mtime": 1.0}, {"zip_size": None}, {"zip_mtime": "x"}])
def test_a_file_that_changed_since_is_extracted(clean_dir, changes):
    info, project_dir = clean_dir
    assert _action(info, project_dir, **changes) == "extract"


@pytest.mark.parametrize("session", [None, {}])
def test_a_dir_without_a_session_is_extracted(clean_dir, session, tmp_path):
    info, project_dir = clean_dir
    assert project_io.open_action(session, info, project_dir, []) == "extract"
    assert project_io.open_action(session, info, project_dir, [str(tmp_path / "child")]) == "extract"


def test_a_dir_without_a_document_is_extracted(clean_dir, tmp_path):
    info, project_dir = clean_dir
    os.remove(os.path.join(project_dir, "document.json"))
    assert _action(info, project_dir) == "extract"
    assert _action(info, project_dir, dirty=True) == "extract"
    assert _action(info, project_dir, [str(tmp_path / "child")]) == "extract"


def test_a_dir_missing_the_bundled_video_is_extracted(isolated_dirs, tmp_path):
    project_dir, project_id = project_io.create_project_dir()
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video" * 20)
    project_io.trust_video(project_dir, str(video))
    path = str(tmp_path / "p.tbaw")
    settings = {"video": {"path": "clip.mp4", "offset_s": 0.0}, "bundle": {"include_video": True}}
    document = _document()
    project_io.autosave_to_dir(document, settings, project_dir)
    project_io.save_project(document, path, settings, project_dir, project_id)
    info = project_io.inspect_bundle(path)
    assert _action(info, project_dir) == "reuse"  # the video is still where the project says

    video.unlink()
    assert project_io.video_extract_pending(info, project_dir)
    assert _action(info, project_dir) == "extract"
    # Unsaved edits are never wiped for a video.
    assert _action(info, project_dir, dirty=True) == "recover"


def test_open_action_writes_nothing(clean_dir):
    info, project_dir = clean_dir

    def state():
        return {f: os.stat(os.path.join(project_dir, f)).st_mtime_ns for f in os.listdir(project_dir)}

    before = state()
    for changes in ({}, {"dirty": True}, {"zip_size": 1}):
        _action(info, project_dir, **changes)
    assert state() == before
