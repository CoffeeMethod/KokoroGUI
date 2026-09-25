"""Video dock: the project's reference video (phase 5, TB16), for dubbing
against picture.

A `QVideoWidget` fed by a `QMediaPlayer` with no audio output, so it plays
muted: the transport is the only thing that makes sound. The player follows
the transport (kokoro_gui/qt/video_sync.py): each 30 Hz `positionChanged`
seeks it when it has drifted more than 40 ms, and `stateChanged` plays or
pauses it. It never moves the transport.

QtMultimedia is imported the first time a video is shown, not at startup,
so a machine whose multimedia backend can't load (a missing system
library) still opens the app; the dock then says playback isn't available.
`PLAYER_ENABLED = False` keeps the import from happening at all
(scripts/render_screenshot.py sets it for headless runs).

The offset spin box edits `project_settings["video"]["offset_s"]` through
the app (`QtTTSApp.set_video_offset`): video time is transport time plus
the offset.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QUrl
from PySide6.QtWidgets import (
    QDockWidget, QDoubleSpinBox, QHBoxLayout, QLabel, QStackedWidget, QVBoxLayout, QWidget,
)

from kokoro_gui.qt import video_sync

# False keeps QtMultimedia from loading (headless screenshot runs); the dock
# then shows the "not available" line for any video.
PLAYER_ENABLED = True

NO_VIDEO_TEXT = "No reference video. File > Load Video... adds one."

# `(QMediaPlayer, QVideoWidget)` once loaded, or the reason they can't be.
_multimedia: tuple | str | None = None


def load_multimedia():
    """`((QMediaPlayer, QVideoWidget), None)`, or `(None, reason)` when
    QtMultimedia can't be imported here. Tried once per process."""
    global _multimedia
    if not PLAYER_ENABLED:
        return None, "video playback is turned off for this run"
    if _multimedia is None:
        try:
            from PySide6.QtMultimedia import QMediaPlayer
            from PySide6.QtMultimediaWidgets import QVideoWidget
        except Exception as e:  # noqa: BLE001 - ImportError, or an OSError from a missing system library
            _multimedia = str(e) or e.__class__.__name__
        else:
            _multimedia = (QMediaPlayer, QVideoWidget)
    if isinstance(_multimedia, str):
        return None, _multimedia
    return _multimedia, None


class VideoDock(QDockWidget):
    def __init__(self, app, parent=None):
        super().__init__("Video", parent)
        self.setObjectName("dock_video")
        self.app = app
        self.player = None
        self.video_widget = None
        self.path: str | None = None
        self.offset_s = 0.0
        self.unavailable_reason: str | None = None

        content = QWidget()
        layout = QVBoxLayout(content)
        self.stack = QStackedWidget()
        self.message_label = QLabel(NO_VIDEO_TEXT)
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message_label.setWordWrap(True)
        self.stack.addWidget(self.message_label)
        layout.addWidget(self.stack, 1)

        offset_row = QHBoxLayout()
        offset_row.addWidget(QLabel("Offset:"))
        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setRange(-86400.0, 86400.0)
        self.offset_spin.setDecimals(3)
        self.offset_spin.setSingleStep(0.1)
        self.offset_spin.setSuffix(" s")
        self.offset_spin.setToolTip("Video time at the timeline's 0. Positive skips into the video; "
                                    "negative starts it later.")
        self.offset_spin.setEnabled(False)
        self.offset_spin.valueChanged.connect(self._on_offset_edited)
        offset_row.addWidget(self.offset_spin)
        offset_row.addStretch(1)
        layout.addLayout(offset_row)
        self.setWidget(content)

        transport = getattr(app, "transport", None)
        if transport is not None:
            transport.positionChanged.connect(self.follow_position)
            transport.stateChanged.connect(self.follow_state)

    # -- what's shown -------------------------------------------------------------

    def set_video(self, path: str | None, offset_s: float = 0.0, missing: str | None = None) -> None:
        """Shows `path` (None clears the dock), `offset_s` in the spin box,
        and puts the player where the transport is. `missing` names a video
        the project has but that isn't there, for the message in its place."""
        self.path = path
        self.offset_s = float(offset_s or 0.0)
        self.offset_spin.blockSignals(True)
        self.offset_spin.setValue(self.offset_s)
        self.offset_spin.blockSignals(False)
        self.offset_spin.setEnabled(path is not None or missing is not None)
        if path is None:
            if self.player is not None:
                self.player.stop()
                self.player.setSource(QUrl())
            if missing:
                self._show_message(f"Reference video not found: {missing}. File > Load Video... finds it.")
            else:
                self._show_message(NO_VIDEO_TEXT)
            return
        if not self._ensure_player():
            self._show_message(f"Video playback isn't available here ({self.unavailable_reason}).")
            return
        self.player.setSource(QUrl.fromLocalFile(path))
        self.stack.setCurrentWidget(self.video_widget)
        transport = getattr(self.app, "transport", None)
        if transport is not None:
            self.follow_position(transport.position())
            self.follow_state(transport.state)

    def set_offset(self, offset_s: float) -> None:
        self.offset_s = float(offset_s or 0.0)
        transport = getattr(self.app, "transport", None)
        if transport is not None:
            self.follow_position(transport.position())

    def _show_message(self, text: str) -> None:
        self.message_label.setText(text)
        self.stack.setCurrentWidget(self.message_label)

    def _ensure_player(self) -> bool:
        if self.player is not None:
            return True
        classes, reason = load_multimedia()
        if classes is None:
            self.unavailable_reason = reason
            return False
        media_player_cls, video_widget_cls = classes
        self.video_widget = video_widget_cls()
        self.stack.addWidget(self.video_widget)
        # No setAudioOutput: a player without an audio output makes no
        # sound, which is the muted reference picture TB16 asks for.
        self.player = media_player_cls(self)
        self.player.setVideoOutput(self.video_widget)
        self.player.errorOccurred.connect(self._on_player_error)
        return True

    def _on_player_error(self, _error, message: str) -> None:
        if self.path is not None:
            self._show_message(f"Couldn't play the reference video: {message or 'unknown error'}")

    # -- following the transport ----------------------------------------------------

    def follow_position(self, seconds: float) -> None:
        if self.player is None or self.path is None:
            return
        target = video_sync.seek_target_ms(seconds, self.offset_s, self.player.position())
        if target is not None:
            self.player.setPosition(target)

    def follow_state(self, state: str) -> None:
        if self.player is None or self.path is None:
            return
        if video_sync.player_command(state) == "play":
            self.player.play()
        else:
            self.player.pause()

    def _on_offset_edited(self, value: float) -> None:
        setter = getattr(self.app, "set_video_offset", None)
        if setter is not None:
            setter(value)
        else:
            self.set_offset(value)
