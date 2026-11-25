"""What the render, update and speak tests share."""
import json
import os
import zipfile

import pytest

from kokoro_gui.audio import post
from kokoro_gui.headless.errors import HeadlessError
from kokoro_gui.headless.render import render_script
from kokoro_gui.qt import project as project_io

SCRIPT = "The door opens.\n\n[Sam]: Who is there?\n\n[Ann]: Only me, with the post."


def files_under(folder):
    return sorted(os.path.relpath(os.path.join(d, f), folder).replace("\\", "/")
                  for d, _dirs, names in os.walk(folder) for f in names)


def working_dirs():
    root = project_io.projects_root()
    return [os.path.join(root, name) for name in os.listdir(root)] if os.path.isdir(root) else []


def reopen(path):
    """The bundle from scratch: its working folder goes first."""
    info = project_io.inspect_bundle(path)
    project_dir = os.path.join(project_io.projects_root(), info.project_id)
    project_io.delete_project_dir(project_dir)
    os.makedirs(project_dir)
    project_io.extract_small(info, project_dir)
    project_io.extract_audio(info, project_dir)
    return project_io.finish_open(info, project_dir)


def keys_of(path):
    with zipfile.ZipFile(path) as zf:
        data = json.loads(zf.read("document.json"))
    return [s["cache_key"] for clip in data["clips"] for s in clip["segments"]]


def failing(script=SCRIPT, **kwargs):
    if "host" not in kwargs:
        kwargs.setdefault("engine", "fake")
    with pytest.raises(HeadlessError) as caught:
        render_script(script, **kwargs)
    return caught.value


def assert_nothing_left(tmp_path, working_folder):
    """No file or folder at the output, no lock, no samples kept."""
    assert not (tmp_path / "audio_output").exists()
    folders = working_dirs()
    assert len(folders) == (1 if working_folder else 0)
    assert not any(project_io.is_locked(d) for d in folders)
    assert post._RENDER_CACHE == {}
