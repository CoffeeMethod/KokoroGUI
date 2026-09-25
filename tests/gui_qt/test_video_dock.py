"""The reference video (phase 5, TB16): the video dock, File > Load Video...,
the relink prompt and the dock following the transport."""
import os

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog

import kokoro_gui.qt.app  # noqa: F401 - the app module first: the docks package imports it back
from kokoro_gui.qt import project as project_io
from kokoro_gui.qt.docks import video_dock
from kokoro_gui.qt.workspace import ADVANCED, SIMPLE


@pytest.fixture(autouse=True)
def _no_player(monkeypatch):
    """No QtMultimedia unless a test asks for it: a fake .mp4 fed to a real
    backend only produces an error, and a machine without the backend's
    system libraries can't load one at all."""
    monkeypatch.setattr(video_dock, "PLAYER_ENABLED", False)


def _video_file(tmp_path, name="clip.mp4"):
    path = tmp_path / "footage" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0\0\0\x18ftypmp42" + b"\1" * 64)
    return str(path)


def _save_as(qt_app, path):
    qt_app.save_project_as(path)
    qt_app.wait_for_project_io()
    return qt_app.project_path


class FakePlayer:
    def __init__(self, position=0):
        self._position = position
        self.seeks = []
        self.calls = []

    def position(self):
        return self._position

    def setPosition(self, ms):  # noqa: N802 (Qt name)
        self.seeks.append(ms)
        self._position = ms

    def play(self):
        self.calls.append("play")

    def pause(self):
        self.calls.append("pause")


def test_qt_multimedia_imports_or_skips():
    """The CI check: QtMultimedia loads here, or this skips with the reason
    (on Linux a missing system library, e.g. libpulse, fails the import)."""
    try:
        from PySide6 import QtMultimedia, QtMultimediaWidgets  # noqa: F401
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"QtMultimedia unavailable: {e}")


def test_real_player_is_built_muted_when_qt_multimedia_loads(qt_app, tmp_path, monkeypatch):
    monkeypatch.setattr(video_dock, "PLAYER_ENABLED", True)
    monkeypatch.setattr(video_dock, "_multimedia", None)
    classes, reason = video_dock.load_multimedia()
    if classes is None:
        pytest.skip(f"QtMultimedia unavailable: {reason}")
    qt_app.load_video(_video_file(tmp_path))
    dock = qt_app.video_dock
    assert dock.player is not None
    assert dock.player.audioOutput() is None  # no audio output: muted
    assert dock.player.videoOutput() is dock.video_widget
    qt_app.video_dock.set_video(None)
    assert dock.stack.currentWidget() is dock.message_label

def test_video_dock_is_a_tab_in_advanced_and_hidden_in_simple(qt_app):
    dock = qt_app.video_dock
    assert dock.objectName() == "dock_video"
    assert dock in qt_app._all_docks()
    assert qt_app.dockWidgetArea(dock) == Qt.DockWidgetArea.TopDockWidgetArea
    assert dock in qt_app.tabifiedDockWidgets(qt_app.settings_dock)
    assert qt_app.workspaces.active == ADVANCED
    assert not dock.isHidden()

    qt_app.activate_workspace(SIMPLE)
    assert dock.isHidden()
    qt_app.activate_workspace(ADVANCED)
    assert not dock.isHidden()


def test_dock_without_a_video_says_how_to_add_one(qt_app):
    dock = qt_app.video_dock
    assert dock.player is None
    assert dock.stack.currentWidget() is dock.message_label
    assert dock.message_label.text() == video_dock.NO_VIDEO_TEXT
    assert not dock.offset_spin.isEnabled()


def test_without_qt_multimedia_the_dock_degrades_to_a_label(qt_app, tmp_path, monkeypatch):
    monkeypatch.setattr(video_dock, "PLAYER_ENABLED", True)
    monkeypatch.setattr(video_dock, "_multimedia", "libpulse.so.0: cannot open shared object file")
    assert qt_app.load_video(_video_file(tmp_path))
    dock = qt_app.video_dock
    assert dock.player is None
    assert "isn't available" in dock.message_label.text() and "libpulse" in dock.message_label.text()
    assert dock.offset_spin.isEnabled()


def test_load_video_is_absolute_while_untitled_and_relative_after_save_as(qt_app, tmp_path):
    video = _video_file(tmp_path)
    assert qt_app.load_video(video)
    assert qt_app.project_settings["video"] == {"path": os.path.abspath(video), "offset_s": 0.0}
    assert qt_app.video_dock.path == os.path.realpath(video)

    _save_as(qt_app, str(tmp_path / "work" / "proj"))
    assert qt_app.project_settings["video"]["path"] == "../footage/clip.mp4"
    _save_as(qt_app, str(tmp_path / "proj2"))
    assert qt_app.project_settings["video"]["path"] == "footage/clip.mp4"
    assert qt_app.video_path() == os.path.realpath(video)


def test_load_video_refuses_a_missing_file(qt_app, tmp_path):
    assert not qt_app.load_video(str(tmp_path / "nope.mp4"))
    assert "video" not in qt_app.project_settings


def test_offset_box_edits_the_setting_and_survives_a_reload(qt_app, tmp_path):
    video = _video_file(tmp_path)
    qt_app.load_video(video)
    qt_app.video_dock.offset_spin.setValue(2.5)
    assert qt_app.project_settings["video"]["offset_s"] == 2.5
    assert qt_app.video_dock.offset_s == 2.5
    path = _save_as(qt_app, str(tmp_path / "proj"))
    assert project_io.load_project(path).project_settings["video"] == {"path": "footage/clip.mp4", "offset_s": 2.5}
    # Loading another file keeps the offset.
    qt_app.load_video(_video_file(tmp_path, "other.mp4"))
    assert qt_app.project_settings["video"] == {"path": "footage/other.mp4", "offset_s": 2.5}


def test_dock_follows_the_transport_past_the_drift_threshold(qt_app):
    dock = qt_app.video_dock
    player = FakePlayer(position=5_000)
    dock.player = player
    dock.path = "clip.mp4"
    dock.offset_s = 1.0

    dock.follow_position(4.03)  # target 5030: 30 ms off, left alone
    assert player.seeks == []
    dock.follow_position(4.5)  # target 5500
    assert player.seeks == [5_500]

    dock.follow_state("playing")
    dock.follow_state("paused")
    dock.follow_state("stopped")
    assert player.calls == ["play", "pause", "pause"]

    # The transport drives the dock through its signals.
    qt_app.transport.positionChanged.emit(10.0)
    assert player.seeks[-1] == 11_000
    qt_app.transport.stateChanged.emit("playing")
    assert player.calls[-1] == "play"


def test_open_with_the_video_missing_offers_a_relink(qt_app, tmp_path, monkeypatch):
    video = _video_file(tmp_path)
    qt_app.load_video(video)
    path = _save_as(qt_app, str(tmp_path / "proj"))
    os.remove(video)
    replacement = _video_file(tmp_path, "found.mp4")
    offered = []
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: offered.append(a) or (replacement, "")))

    qt_app.new_project()
    qt_app.open_project(path)
    qt_app.wait_for_project_io()

    assert offered and offered[0][1] == "Load video"
    assert qt_app.project_settings["video"]["path"] == "footage/found.mp4"
    assert qt_app.video_dock.path == os.path.realpath(replacement)


def test_open_plays_the_bundled_copy_when_the_video_moved(qt_app, tmp_path, monkeypatch):
    video = _video_file(tmp_path)
    qt_app.load_video(video)
    qt_app.project_settings.setdefault("bundle", {})["include_video"] = True
    path = _save_as(qt_app, str(tmp_path / "proj"))
    os.remove(video)
    asked = []
    monkeypatch.setattr(type(qt_app), "_offer_video_relink", lambda self: asked.append(True))

    qt_app.new_project()
    qt_app.open_project(path)
    qt_app.wait_for_project_io()

    assert asked == []
    played = qt_app.video_dock.path
    assert played is not None and os.path.dirname(played) == os.path.join(qt_app.project_dir, "video")
    assert qt_app.project_settings["video"]["path"] == "footage/clip.mp4"


def test_export_dialog_bundles_the_video_only_when_asked(qt_app):
    from kokoro_gui.qt.docks.export_dialog import ExportDialog, run_export

    dialog = ExportDialog(qt_app)
    assert dialog.bundle_video_check.isChecked() is False
    assert dialog.bundle_values()["include_video"] is False
    dialog.bundle_video_check.setChecked(True)
    run_export(qt_app, dialog.values(), bundle=dialog.bundle_values())  # no clips: nothing scheduled
    assert qt_app.project_settings["bundle"]["include_video"] is True
    assert ExportDialog(qt_app).bundle_video_check.isChecked() is True


def test_a_declined_relink_leaves_the_setting_and_says_what_is_missing(qt_app, tmp_path):
    video = _video_file(tmp_path)
    qt_app.load_video(video)
    path = _save_as(qt_app, str(tmp_path / "proj"))
    os.remove(video)

    qt_app.new_project()
    qt_app.open_project(path)  # the fixture's file dialog returns nothing
    qt_app.wait_for_project_io()

    assert qt_app.project_settings["video"]["path"] == "footage/clip.mp4"
    dock = qt_app.video_dock
    assert dock.path is None
    assert "not found" in dock.message_label.text() and "clip.mp4" in dock.message_label.text()
