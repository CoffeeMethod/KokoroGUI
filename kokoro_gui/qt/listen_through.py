"""Listen-through mode (plan 27): the app half.

`ListenThroughMixin` gives `QtTTSApp` the playback speed and the flags.

Speed is `Transport.rate`, 0.5x to 2x with the pitch unchanged
(`audio/stretch.py`). It is a view setting, not project data: the
Transport dock's combo and the `[` and `]` keys set it through
`set_playback_rate`, and `settings["playback_rate"]` in `config_qt.json`
remembers it. L plays, and while it plays steps through
`keymap.LISTEN_RATES`.

A flag is a marker with a note (`daw/markers.py`), so it is stored, saved,
drawn on the ruler and listed in the Outline like any marker and needs no
new data. M drops one named "Flag N" at the playhead as one undo step and
opens `FlagNotePopup`, a small window with a note field that takes the
keyboard but leaves the transport running. Enter saves the note as a second
undo step, Esc keeps the flag with no note, a click elsewhere saves what was
typed. N and Shift+N seek to the next and previous flag.

Kept out of app.py, which already holds the keyboard slots it replaces.
"""
from __future__ import annotations

from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QLineEdit

from kokoro_gui.audio import transport as transport_mod
from kokoro_gui.daw import markers as marker_ops
from kokoro_gui.daw.undo import SetFieldCommand
from kokoro_gui.qt import keymap

RATE_SETTING = "playback_rate"


def rate_label(rate: float) -> str:
    """"1x", "1.5x", "0.75x"."""
    return f"{rate:g}x"


def step_rate(current: float, steps, up: bool):
    """The step after (`up`) or before `current` in `steps`, or None when
    `current` is already at that end."""
    current = float(current)
    if up:
        return next((step for step in steps if step > current + 1e-6), None)
    return next((step for step in reversed(steps) if step < current - 1e-6), None)


def nearest_step(rate) -> float:
    """The `Transport` rate step closest to `rate`, the one the combo can show."""
    rate = transport_mod.clamp_rate(rate)
    return min(transport_mod.RATE_STEPS, key=lambda step: abs(step - rate))


class FlagNotePopup(QFrame):
    """The note field a new flag opens. `finished(marker_id, note)` is
    emitted once, with the note to store: the typed text on Enter or when
    the popup loses focus, "" on Esc."""

    def __init__(self, parent, on_finished):
        super().__init__(parent, Qt.WindowType.Popup)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self._on_finished = on_finished
        self._marker_id = None
        self._done = False
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        self.label = QLabel("")
        layout.addWidget(self.label)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText("Note (Enter saves, Esc keeps the flag without one)")
        self.edit.setMinimumWidth(320)
        self.edit.returnPressed.connect(lambda: self._finish(self.edit.text()))
        self.edit.installEventFilter(self)
        layout.addWidget(self.edit)

    def begin(self, marker_id: str, name: str, global_pos: QPoint) -> None:
        if self.isVisible():
            self._finish(self.edit.text())
        self._marker_id = marker_id
        self._done = False
        self.label.setText(name)
        self.edit.clear()
        self.adjustSize()
        self.move(global_pos)
        self.show()
        self.edit.setFocus()

    def eventFilter(self, obj, event):
        if obj is self.edit and event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape:
            self._finish("")
            return True
        return super().eventFilter(obj, event)

    def hideEvent(self, event) -> None:
        # A click outside closes a popup without a key: keep what was typed.
        self._finish(self.edit.text())
        super().hideEvent(event)

    def _finish(self, note: str) -> None:
        if self._done:
            return
        self._done = True
        marker_id = self._marker_id
        self.hide()
        self._on_finished(marker_id, note.strip())


class ListenThroughMixin:
    def _init_listen_through(self) -> None:
        self._flag_popup = FlagNotePopup(self, self._store_flag_note)

    # -- speed -------------------------------------------------------------------

    def set_playback_rate(self, rate, remember: bool = True) -> float:
        """The transport's speed, the Transport dock's combo and
        `settings["playback_rate"]` all at `rate` (clamped). Returns it."""
        rate = self.transport.set_rate(rate)
        if self.transport_dock is not None:
            self.transport_dock.set_rate_choice(rate)
        if remember and self.settings.get(RATE_SETTING) != rate:
            self._set_setting(RATE_SETTING, rate)
        return rate

    def apply_saved_playback_rate(self) -> None:
        """At launch, once the Transport dock exists."""
        self.set_playback_rate(nearest_step(self.settings.get(RATE_SETTING, 1.0)), remember=False)

    def _step_playback_rate(self, steps, up: bool) -> None:
        target = step_rate(self.transport.rate, steps, up)
        if target is None:
            self.set_status(f"Playback speed is already {rate_label(self.transport.rate)}"
                            f"{', the fastest' if up else ', the slowest'}.")
            return
        self.set_status(f"Playback speed {rate_label(self.set_playback_rate(target))}.")

    def rate_up_key(self) -> None:
        self._step_playback_rate(transport_mod.RATE_STEPS, up=True)

    def rate_down_key(self) -> None:
        self._step_playback_rate(transport_mod.RATE_STEPS, up=False)

    def play_key(self) -> None:
        """L: play at the current speed; while it plays, 1.5x, then 2x."""
        if not self.transport.is_playing:
            self.transport.play()
            return
        target = step_rate(self.transport.rate, keymap.LISTEN_RATES, up=True)
        if target is not None:
            self.set_status(f"Playback speed {rate_label(self.set_playback_rate(target))}.")

    # -- flags -------------------------------------------------------------------

    def drop_flag(self) -> None:
        """M: a marker named "Flag N" at the playhead, then its note field."""
        from kokoro_gui.qt.docks.transport_dock import format_clock  # the docks package imports app

        document = self.level.document
        new_list, marker = marker_ops.add_marker(document.settings, self.transport.position(),
                                                 marker_ops.next_flag_name(document.settings))
        self._push_markers(new_list)
        self.set_status(f"{marker['name']} at {format_clock(marker['seconds'])}.")
        self._flag_popup.begin(marker["id"], marker["name"], self._flag_popup_position())

    def _flag_popup_position(self) -> QPoint:
        view = self.timeline_dock.timeline_view
        return view.mapToGlobal(QPoint(12, 12))

    def _store_flag_note(self, marker_id: str, note: str) -> None:
        """The popup's answer. An empty note leaves the flag as dropped."""
        marker = marker_ops.get_marker(self.level.document.settings, marker_id)
        if marker is None or not note or note == marker["note"]:
            return
        self._push_markers(marker_ops.rename_marker(self.level.document.settings, marker_id, marker["name"], note))

    def _push_markers(self, new_list: list) -> None:
        self.level.document.undo_stack.push(
            SetFieldCommand("document", None, "settings", new_list, key=marker_ops.MARKERS_KEY))
        self.schedule_save()
        self.refresh_timeline()

    def _flag_times(self) -> list:
        return [flag["seconds"] for flag in marker_ops.list_flags(self.level.document.settings)]

    def go_to_previous_flag(self) -> None:
        target = keymap.previous_time(self._flag_times(), self.transport.position())
        if target is None:
            self.set_status("No flag before the playhead.", "warning")
        else:
            self.transport.seek(target)

    def go_to_next_flag(self) -> None:
        target = keymap.next_time(self._flag_times(), self.transport.position())
        if target is None:
            self.set_status("No flag after the playhead.", "warning")
        else:
            self.transport.seek(target)
