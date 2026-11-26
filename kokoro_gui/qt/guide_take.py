"""Record a guide take (plan 31, grill PG8).

`store_guide_take` turns a recording into a take on a clip: the samples go
to a temp WAV, `project.import_audio_file` copies it into the project's
`audio/imported/`, and one `AddTakeCommand` adds a one-segment take and
makes it the active one. The segment has no `cache_key` and its
`range` starts after the count-in, so playback and export skip the beeps.
`dirty.is_recorded_take` reads those two facts: a clip whose active take
is recorded is never stale. No `.tbaw` field is new: the take is stored
like any other, and its file is bundled with the imported audio.

`GuideTakePanel` is the non-modal recorder: input device, level meter,
pre-roll, Record and Stop. It drives `audio/recorder.py` and the app's
transport. Tests replace `recorder.default_stream_factory` and
`playback.play_array`, so no test opens a real device.
"""
from __future__ import annotations

import os
import tempfile
from typing import Optional

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout,
)

import playback
from kokoro_gui.audio import recorder as recorder_module
from kokoro_gui.daw import revision
from kokoro_gui.daw.models import Segment
from kokoro_gui.daw.undo import AddTakeCommand
from kokoro_gui.i18n import tr
from kokoro_gui.qt import tips

# A recording that ends this soon after the count-in holds nothing to keep.
MIN_TAKE_S = 0.1


def store_guide_take(app, clip_id: str, samples, sample_rate: int,
                     count_in_s: float = recorder_module.COUNT_IN_S) -> Optional[Segment]:
    """Adds `samples` (mono float32, the count-in first) to clip `clip_id`
    as its new active take, one undo step on the clip's project. Returns
    the take's segment, or None with a status line when the clip is gone,
    the recording is too short or the file couldn't be stored."""
    import soundfile as sf

    from kokoro_gui.qt import project as project_io

    project = app.project_of_clip_id(clip_id)
    clip = project.document.get_clip(clip_id) if project is not None else None
    if clip is None:
        app.set_status(tr("The clip was deleted while recording; the take was not kept."), "warning")
        return None
    data = np.asarray(samples, dtype=np.float32).reshape(-1)
    seconds = len(data) / float(sample_rate)
    if seconds < count_in_s + MIN_TAKE_S:
        app.set_status(tr("The recording stopped before the count-in ended; nothing was kept."), "warning")
        return None
    if not project.project_dir:
        app.set_status(tr("This project has no folder to store the take in."), "warning")
        return None
    fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="guide_take_")
    os.close(fd)
    try:
        sf.write(tmp, data, int(sample_rate), subtype="PCM_16")
        revision.bump_files()  # a file is about to be copied into the project
        path = project_io.import_audio_file(tmp, project.project_dir)
    except (OSError, RuntimeError, project_io.ProjectError) as e:
        app.set_status(tr("Couldn't store the take: {error}", error=e), "error")
        return None
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    end_s = round(seconds, 3)
    start_s = round(float(count_in_s), 3)
    segment = Segment(order_index=0, text=" ".join(project.document.clip_text(clip).split()), cache_key=None,
                      audio_path=path, duration=round(end_s - start_s, 3), raw=True, range=[start_s, end_s])
    project.document.undo_stack.push(AddTakeCommand(clip_id, [segment]))
    if app.editor is not None:
        app.editor.rehighlight()
    app.schedule_save()
    app.refresh_timeline()
    take = int(clip.overrides.get("take", 0) or 0)
    app.set_status(tr("Recorded take {take} ({end_s:.1f}s).", take=take + 1, end_s=end_s - start_s), "info")
    return segment


# -- the panel ------------------------------------------------------------------------

DEFAULT_PREROLL_S = 2.0
LEVEL_TIMER_MS = 33
# How often a take that stopped during a project open or save tries again.
COMMIT_RETRY_MS = 250


def level_percent(linear: float) -> int:
    """A linear peak on the meter's 0..100 scale: -60 dBFS to 0 dBFS."""
    if linear <= 0.0:
        return 0
    db = max(-60.0, 20.0 * float(np.log10(min(1.0, linear))))
    return int(round((db + 60.0) / 60.0 * 100.0))


class GuideTakePanel(QDialog):
    """Block menu > Record guide take..., the R key and the Transport dock's
    Rec button. Non-modal: an input device, a level meter, the pre-roll,
    Record and Stop.

    Record starts the input and plays the count-in (`recorder.count_in`)
    at once, so the beeps are in the file. The transport starts `pre-roll`
    seconds before the clip, timed so the clip's start lands on the beat
    after the last beep; the monitor mode decides what the performer
    hears. Stop pauses the transport and stores the take
    (`store_guide_take`). A stop while a project file is opening or saving
    keeps the take here and stores it when that ends."""

    def __init__(self, app, clip_id: str, parent=None):
        super().__init__(parent if parent is not None else app)
        self.setWindowTitle(tr("Record guide take"))
        self.setModal(False)
        self.app = app
        self.clip_id = clip_id
        self._recorder = None
        self._seek_s = None
        self._delay_ms = 0
        self._played = False
        self._pending = None

        layout = QVBoxLayout(self)
        self.clip_label = QLabel()
        self.clip_label.setWordWrap(True)
        layout.addWidget(self.clip_label)
        form = QFormLayout()
        self.device_combo = QComboBox()
        self.device_combo.setToolTip(tr("The microphone to record from."))
        self.device_combo.addItem(tr("Default input"), None)
        self._rates = {None: recorder_module.DEFAULT_SAMPLE_RATE}
        for index, name, rate in recorder_module.input_devices():
            self.device_combo.addItem(name, index)
            self._rates[index] = rate
        form.addRow(tr("Input:"), self.device_combo)
        self.preroll_spin = QDoubleSpinBox()
        self.preroll_spin.setRange(0.0, recorder_module.COUNT_IN_S)
        self.preroll_spin.setSingleStep(0.5)
        self.preroll_spin.setSuffix(" s")
        self.preroll_spin.setValue(DEFAULT_PREROLL_S)
        self.preroll_spin.setToolTip(tr("How much of the timeline plays before the clip, under the count-in."))
        form.addRow(tr("Pre-roll:"), self.preroll_spin)
        self.level_bar = QProgressBar()
        self.level_bar.setRange(0, 100)
        self.level_bar.setTextVisible(False)
        self.level_bar.setToolTip(tr("Input level, -60 to 0 dBFS"))
        form.addRow(tr("Level:"), self.level_bar)
        layout.addLayout(form)
        buttons = QHBoxLayout()
        self.record_btn = QPushButton(tr("Record"))
        self.record_btn.setToolTip(tr("Start the count-in, then record."))
        self.stop_btn = QPushButton(tr("Stop"))
        self.stop_btn.setToolTip(tr("Stop and keep the take."))
        self.stop_btn.setEnabled(False)
        buttons.addWidget(self.record_btn)
        buttons.addWidget(self.stop_btn)
        layout.addLayout(buttons)
        self.status_label = QLabel(tr("Three beeps, then your line. The beeps are cut from the take."))
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self._level_timer = QTimer(self)
        self._level_timer.setInterval(LEVEL_TIMER_MS)
        self._level_timer.timeout.connect(self.update_level)
        self._transport_timer = QTimer(self)
        self._transport_timer.setSingleShot(True)
        self._transport_timer.timeout.connect(self.start_transport)
        self._commit_timer = QTimer(self)
        self._commit_timer.setSingleShot(True)
        self._commit_timer.timeout.connect(self.commit_pending)
        self.record_btn.clicked.connect(self.record)
        self.stop_btn.clicked.connect(self.stop)
        self.set_clip(clip_id)
        tips.label_rows(self)

    def set_clip(self, clip_id: str) -> None:
        """Points the panel at another clip; ignored while recording."""
        if self.is_recording():
            return
        self.clip_id = clip_id
        project = self.app.project_of_clip_id(clip_id)
        clip = project.document.get_clip(clip_id) if project is not None else None
        text = " ".join(project.document.clip_text(clip).split()) if clip is not None else ""
        self.clip_label.setText(text if len(text) <= 80 else text[:79] + "…")

    def is_recording(self) -> bool:
        return self._recorder is not None and self._recorder.recording

    def has_pending_take(self) -> bool:
        return self._pending is not None

    def transport_delay_ms(self) -> int:
        """When the transport starts after Record: the count-in less the pre-roll."""
        return self._delay_ms

    def _say(self, message: str) -> None:
        self.status_label.setText(message)

    def record(self) -> bool:
        if self.is_recording():
            return False
        if self.app.is_io_busy():
            self._say(tr("Wait for the project to finish opening or saving."))
            return False
        if self._pending is not None:
            self._say(tr("The last take is still waiting to be stored."))
            return False
        if self.app.project_of_clip_id(self.clip_id) is None:
            self._say(tr("The clip is gone."))
            return False
        if not recorder_module.available():
            self._say(tr("No audio input: sounddevice or PortAudio is missing."))
            return False
        device = self.device_combo.currentData()
        rate = int(self._rates.get(device, recorder_module.DEFAULT_SAMPLE_RATE))
        self._recorder = recorder_module.Recorder(sample_rate=rate, device=device)
        try:
            self._recorder.start()
        except recorder_module.RecorderError as e:
            self._recorder = None
            self._say(str(e))
            return False
        playback.play_array(recorder_module.count_in(rate), rate)
        placed = self.app.build_arrangement().by_clip_id().get(self.clip_id)
        self._played = False
        if placed is not None:
            lead = min(float(self.preroll_spin.value()), max(0.0, placed.start_s), recorder_module.COUNT_IN_S)
            self._seek_s = placed.start_s - lead
            self._delay_ms = int(round((recorder_module.COUNT_IN_S - lead) * 1000))
            self._transport_timer.start(self._delay_ms)
        else:
            self._seek_s = None
            self._delay_ms = 0
        self._level_timer.start()
        self._set_controls(recording=True)
        self._say(tr("Recording..."))
        return True

    def _set_controls(self, recording: bool) -> None:
        self.record_btn.setEnabled(not recording)
        self.stop_btn.setEnabled(recording)
        self.device_combo.setEnabled(not recording)
        self.preroll_spin.setEnabled(not recording)

    def start_transport(self) -> None:
        """The pre-roll: plays the timeline from `pre-roll` seconds before the clip."""
        if not self.is_recording() or self._seek_s is None or self._played:
            return
        self.app.transport.seek(self._seek_s)
        self.app.transport.play()
        self._played = True

    def update_level(self) -> None:
        self.level_bar.setValue(level_percent(self._recorder.level()) if self.is_recording() else 0)

    def stop(self) -> None:
        if not self.is_recording():
            return
        self._transport_timer.stop()
        self._level_timer.stop()
        rate = self._recorder.sample_rate
        samples = self._recorder.stop()
        self._recorder = None
        if self._played:
            self.app.transport.pause()
            self._played = False
        self.level_bar.setValue(0)
        self._set_controls(recording=False)
        self._pending = (samples, rate)
        self.commit_pending()

    def commit_pending(self) -> None:
        """Stores the stopped take, or waits while a project file is
        opening or saving."""
        if self._pending is None:
            return
        if self.app.is_io_busy():
            self._say(tr("The take is stored when the project finishes saving."))
            self._commit_timer.start(COMMIT_RETRY_MS)
            return
        samples, rate = self._pending
        self._pending = None
        segment = store_guide_take(self.app, self.clip_id, samples, rate)
        self._say(tr("Kept a {duration:.1f}s take. Record again for another.", duration=segment.duration)
                  if segment is not None else tr("Nothing was kept."))

    def closeEvent(self, event) -> None:
        self.stop()
        super().closeEvent(event)
