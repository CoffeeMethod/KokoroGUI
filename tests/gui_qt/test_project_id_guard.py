"""A `.tbaw`'s `project_id` names its working copy under `cache/projects/`
(plan 46 F1). A crafted id must never pick a folder outside that root: not a
relative path, an absolute one, a non-string, or a link inside the root that
points out. Open refuses such a bundle before any lock, extraction or wipe,
so a sentinel file outside the cache stays untouched. The dir's own files
(`session.json`, `lock`) are matched case-insensitively, as Windows and
macOS name them."""
import json
import os
import sys
import zipfile

import pytest

from kokoro_gui.daw.models import Document
from kokoro_gui.qt import project as project_io


def _bundle(tmp_path, project_id, extra=None) -> str:
    """A real saved bundle whose manifest then names `project_id`."""
    path = str(tmp_path / "crafted.tbaw")
    project_io.save_project(Document(), path, {})
    with zipfile.ZipFile(path) as zf:
        entries = {name: zf.read(name) for name in zf.namelist()}
    manifest = json.loads(entries["manifest.json"])
    manifest["project_id"] = project_id
    entries["manifest.json"] = json.dumps(manifest).encode("utf-8")
    entries.update(extra or {})
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return path


def _sentinel(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep", encoding="utf-8")
    return outside


def _untouched(outside):
    return sorted(os.listdir(outside)) == ["keep.txt"] and (outside / "keep.txt").read_text(encoding="utf-8") == "keep"


def _escaping_ids(outside):
    relative = os.path.relpath(str(outside), project_io.projects_root())
    return [relative, str(outside), "../outside", "..", ".", "a/b", "a\\b", "C:x", "", " ", "x.", "con", "NUL",
            "a:b", "x" * 200]


@pytest.mark.parametrize("value", ["0123456789abcdef", "x", "p", "My_project-2"])
def test_a_plain_id_is_accepted(value):
    assert project_io.safe_project_id(value) == value


@pytest.mark.parametrize("value", [5, 1.5, True, ["a"], {"a": 1}, None, b"abc"])
def test_a_non_string_id_is_refused(value):
    assert project_io.safe_project_id(value) is None


def test_a_path_like_id_is_refused(isolated_dirs, tmp_path):
    for value in _escaping_ids(tmp_path / "outside"):
        assert project_io.safe_project_id(value) is None, value


@pytest.mark.parametrize("kind", ["relative", "absolute", "int", "list", "dict"])
def test_inspect_refuses_a_bundle_whose_id_is_not_plain(isolated_dirs, tmp_path, kind):
    outside = _sentinel(tmp_path)
    value = {"relative": os.path.relpath(str(outside), project_io.projects_root()), "absolute": str(outside),
             "int": 5, "list": ["a"], "dict": {"a": 1}}[kind]
    path = _bundle(tmp_path, value)
    with pytest.raises(project_io.ProjectError, match="project id"):
        project_io.inspect_bundle(path)
    assert _untouched(outside)


@pytest.mark.parametrize("value", ["../../outside", None])
def test_choose_project_dir_refuses_an_unsafe_id(isolated_dirs, tmp_path, value):
    with pytest.raises(project_io.ProjectError):
        project_io.choose_project_dir(value, str(tmp_path / "a.tbaw"))


def _link_dir(link, target) -> bool:
    try:
        if sys.platform == "win32":
            import _winapi

            _winapi.CreateJunction(str(target), str(link))
        else:
            os.symlink(str(target), str(link), target_is_directory=True)
    except (OSError, AttributeError, NotImplementedError):
        return False
    return True


def test_a_link_inside_the_cache_that_points_out_is_refused(isolated_dirs, tmp_path):
    outside = _sentinel(tmp_path)
    os.makedirs(project_io.projects_root(), exist_ok=True)
    link = os.path.join(project_io.projects_root(), "linked")
    if not _link_dir(link, outside):
        pytest.skip("this machine can't make a directory link")
    assert not project_io.project_dir_in_cache(link)
    with pytest.raises(project_io.ProjectError, match="outside"):
        project_io.choose_project_dir("linked", str(tmp_path / "a.tbaw"))
    project_io.wipe_project_dir(link)
    project_io.delete_project_dir(link)
    assert _untouched(outside)


def test_wipe_and_delete_leave_a_dir_outside_the_cache_alone(isolated_dirs, tmp_path):
    outside = _sentinel(tmp_path)
    project_io.wipe_project_dir(str(outside))
    project_io.delete_project_dir(str(outside))
    assert _untouched(outside)


@pytest.mark.parametrize("kind", ["relative", "absolute"])
def test_open_never_touches_a_folder_outside_the_cache(qt_app, tmp_path, kind):
    outside = _sentinel(tmp_path)
    value = os.path.relpath(str(outside), project_io.projects_root()) if kind == "relative" else str(outside)
    path = _bundle(tmp_path, value)
    qt_app.open_project(path)
    qt_app.wait_for_project_io()
    assert _untouched(outside)
    assert qt_app.project_path != os.path.abspath(path)


def test_open_never_follows_a_link_out_of_the_cache(qt_app, tmp_path):
    outside = _sentinel(tmp_path)
    link = os.path.join(project_io.projects_root(), "linked")
    if not _link_dir(link, outside):
        pytest.skip("this machine can't make a directory link")
    path = _bundle(tmp_path, "linked")
    qt_app.open_project(path)
    qt_app.wait_for_project_io()
    assert _untouched(outside)
    assert qt_app.project_path != os.path.abspath(path)


@pytest.mark.parametrize("alias", ["SESSION.JSON", "Session.Json", "LOCK", "session.json.", "session.json "])
def test_a_private_name_alias_is_never_extracted(isolated_dirs, tmp_path, alias):
    path = _bundle(tmp_path, "0123456789abcdef", {alias: b'{"source_path": "/elsewhere", "dirty": false}'})
    info = project_io.inspect_bundle(path)
    project_dir = project_io.choose_project_dir(info.project_id, info.path)
    project_io.write_session(project_dir, {"source_path": "mine"})
    project_io.extract_small(info, project_dir)
    assert project_io.read_session(project_dir) == {"source_path": "mine"}
    assert [n for n in os.listdir(project_dir) if n.casefold().startswith(("session", "lock"))] == ["session.json"]
    assert project_io.is_private_name(alias)


def test_write_bundle_refuses_a_private_name_alias(isolated_dirs, tmp_path):
    project_dir, project_id = project_io.create_project_dir()
    project_io.write_session(project_dir, {"source_path": "mine"})
    plan, _warnings = project_io.plan_save(Document(), {}, str(tmp_path / "out.tbaw"), project_dir, project_id,
                                           lambda _i: None, str(tmp_path / "fx"))
    plan.audio_files.append(("SESSION.JSON", os.path.join(project_dir, "session.json")))
    plan.audio_files.append(("Document.json", os.path.join(project_dir, "session.json")))
    project_io.write_bundle(plan, [])
    with zipfile.ZipFile(plan.path) as zf:
        names = zf.namelist()
    assert "SESSION.JSON" not in names and "Document.json" not in names
