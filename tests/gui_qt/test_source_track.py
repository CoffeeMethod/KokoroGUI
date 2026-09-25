"""The original dialogue as a per-clip reference (phase 5 D5): File > Import
Source Track, the transport's alt schedule built from each clip's reference
range, the Dub / Original / Both toggle, and the Settings tab's fields."""
import numpy as np
import soundfile as sf
from PySide6.QtWidgets import QApplication

from kokoro_gui.qt import project as project_io

RATE = 24000
CUES = ("1\n00:00:01,000 --> 00:00:02,000\nFirst.\n\n"
        "2\n00:00:03,000 --> 00:00:03,500\nSecond.\n")


def _source_wav(tmp_path, seconds=5.0):
    """A ramp, so a slice's first sample says where in the file it starts."""
    samples = (np.arange(int(RATE * seconds)) / (RATE * seconds)).astype(np.float32)
    path = str(tmp_path / "original.wav")
    sf.write(path, samples, RATE, subtype="FLOAT")
    return path


def _with_cues_and_track(qt_app, tmp_path):
    srt = tmp_path / "cues.srt"
    srt.write_text(CUES, encoding="utf-8")
    clip_ids = qt_app.import_subtitles(str(srt))
    assert qt_app.import_source_track(_source_wav(tmp_path))
    qt_app._rebuild_transport_schedule()
    return clip_ids


def test_without_a_source_track_the_toggle_is_disabled_on_dub(qt_app):
    dock = qt_app.transport_dock

    assert dock.monitor() == "dub"
    assert not any(b.isEnabled() for b in dock.monitor_buttons.values())
    assert qt_app.transport.monitor == "dub"
    assert qt_app.transport.loaded_alt_clips() == []


def test_the_alt_schedule_slices_the_source_track_under_each_cue(qt_app, tmp_path):
    first, second = _with_cues_and_track(qt_app, tmp_path)

    alt = sorted(qt_app.transport.loaded_alt_clips(), key=lambda c: c.start_frame)
    assert [c.clip_id for c in alt] == [first, second]
    assert [c.start_frame for c in alt] == [RATE, 3 * RATE]
    assert [len(c.samples) for c in alt] == [RATE, RATE // 2]
    assert abs(alt[0].samples[0] - 1.0 / 5.0) < 1e-3  # second 1 of a 5 s ramp
    assert all(b.isEnabled() for b in qt_app.transport_dock.monitor_buttons.values())


def test_the_offset_shifts_every_slice_into_the_file(qt_app, tmp_path):
    first, _second = _with_cues_and_track(qt_app, tmp_path)
    qt_app.selection.clear()
    offset = qt_app.settings_dock.scope_fields.widgets["source_offset"]
    assert offset.isEnabled()

    offset.setValue(0.5)
    offset.editingFinished.emit()
    qt_app._rebuild_transport_schedule()

    assert qt_app.document.settings["source_track"]["offset_s"] == 0.5
    alt = {c.clip_id: c for c in qt_app.transport.loaded_alt_clips()}
    assert alt[first].start_frame == RATE
    assert abs(alt[first].samples[0] - 1.5 / 5.0) < 1e-3


def test_the_toggle_sets_the_transport_monitor_and_is_kept_in_the_session(qt_app, tmp_path):
    _with_cues_and_track(qt_app, tmp_path)
    dock = qt_app.transport_dock
    qt_app.transport.seek(1.25)

    dock.monitor_buttons["original"].click()

    assert qt_app.transport.monitor == "original"
    assert abs(qt_app.transport.position() - 1.25) < 1e-6
    assert project_io.read_session(qt_app.root.project_dir)["monitor"] == "original"
    dock.monitor_buttons["both"].click()
    assert qt_app.transport.monitor == "both" and dock.monitor() == "both"


def test_removing_the_source_track_falls_back_to_dub_and_undo_brings_the_choice_back(qt_app, tmp_path):
    _with_cues_and_track(qt_app, tmp_path)
    qt_app.transport_dock.monitor_buttons["original"].click()
    qt_app.selection.clear()

    qt_app.settings_dock.scope_fields.widgets["source_remove"].click()
    QApplication.processEvents()
    qt_app._rebuild_transport_schedule()

    assert "source_track" not in qt_app.document.settings
    assert not qt_app.settings_dock.scope_fields.widgets["source_remove"].isEnabled()
    assert qt_app.transport.monitor == "dub" and qt_app.transport_dock.monitor() == "dub"
    assert qt_app.transport.loaded_alt_clips() == []
    qt_app.undo()
    qt_app._rebuild_transport_schedule()
    assert qt_app.transport.monitor == "original" and qt_app.transport_dock.monitor() == "original"


def test_a_clips_reference_range_is_set_by_hand_and_cleared(qt_app, tmp_path):
    first, _second = _with_cues_and_track(qt_app, tmp_path)
    clip = qt_app.document.get_clip(first)
    qt_app.selection.select_clip(first)
    field = qt_app.settings_dock.scope_fields.widgets["reference_range"]
    assert field.text() == "1.00 - 2.00"

    field.setText("2 - 2.25")
    field.editingFinished.emit()
    assert clip.overrides["reference_range"] == [2.0, 2.25]
    qt_app._rebuild_transport_schedule()
    alt = {c.clip_id: c for c in qt_app.transport.loaded_alt_clips()}
    assert len(alt[first].samples) == RATE // 4

    field.setText("nonsense")
    field.editingFinished.emit()
    assert field.text() == "2.00 - 2.25" and clip.overrides["reference_range"] == [2.0, 2.25]

    field.setText("")
    field.editingFinished.emit()
    assert "reference_range" not in clip.overrides
    qt_app.document.undo_stack.undo()
    assert clip.overrides["reference_range"] == [2.0, 2.25]


def test_the_monitor_choice_comes_back_from_the_session_when_the_document_loads(qt_app, tmp_path):
    """Like the loop region: `session.json` in the root project dir, read
    back when a document is switched in (a recovered session keeps it)."""
    _with_cues_and_track(qt_app, tmp_path)
    qt_app.transport_dock.monitor_buttons["both"].click()
    qt_app.set_monitor_mode("dub", remember=False)
    assert qt_app.transport.monitor == "dub"

    qt_app._switch_document(qt_app.document, None, qt_app.project_settings)

    assert qt_app.transport.monitor == "both" and qt_app.transport_dock.monitor() == "both"


def test_file_menu_import_source_track_asks_for_an_audio_file(qt_app, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    src = _source_wav(tmp_path)
    filters = []

    def _pick(*_args, **kwargs):
        filters.append(kwargs.get("filter", ""))
        return src, ""

    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(_pick))
    qt_app.import_source_track_action.trigger()

    assert "*.wav" in filters[0]
    assert qt_app.document.settings["source_track"]["path"].startswith("audio/imported/")
    assert qt_app.import_source_track_action in qt_app.file_menu.actions()


def test_new_subproject_copies_the_source_track_into_the_child(qt_app, tmp_path):
    clip_ids = _with_cues_and_track(qt_app, tmp_path)
    parent_dir = qt_app.project_dir
    extent = qt_app.document.clip_extent(clip_ids[0])

    child = qt_app.new_subproject(*extent, title="Scene")

    path = project_io.source_track_path(child.document, child.project_dir)
    assert path is not None and path.startswith(child.project_dir)
    assert not path.startswith(parent_dir + "/")
    assert child.document.settings["source_track"]["path"].startswith("audio/imported/")
