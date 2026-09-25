"""FX are read-time post-processing (kokoro_gui/audio/post.py +
kokoro_gui/qt/fx_resolve.py): changing them re-renders what the transport
plays and never dirties the clip. These tests seed a "generated" clip with
a real raw wav and check the transport/arrangement side, with StubEngine
so nothing is synthesized."""
import numpy as np
import soundfile as sf

from kokoro_gui.daw.dirty import compute_expected_cache_hash
from kokoro_gui.daw.models import Segment
from kokoro_gui.daw.undo import SetClipFxCommand

RATE = 24000


def _generated_clip(qt_app, tmp_path, text="hello world", seconds=0.2, amplitude=0.25, name="seg"):
    qt_app.document.text = text
    character = qt_app.document.characters[0]
    clip = qt_app.document.assign_character_to_range(0, len(text), character.id)
    path = tmp_path / f"{name}.wav"
    sf.write(str(path), np.full(int(RATE * seconds), amplitude, dtype=np.float32), RATE)
    config = qt_app._assemble_clip_config(clip)
    expected_hash = compute_expected_cache_hash(qt_app.document.clip_text(clip), config)
    clip.segments = [Segment(order_index=0, text=text, cache_key=expected_hash,
                             audio_path=str(path), duration=seconds)]
    return clip


def _transport_samples(qt_app):
    qt_app._rebuild_transport_schedule()
    return [c.samples for c in qt_app.transport.loaded_clips()]


def test_clip_fx_override_is_audible_without_dirtying_the_clip(qt_app, tmp_path):
    clip = _generated_clip(qt_app, tmp_path)
    assert qt_app.document.dirty_clips() == []
    before = _transport_samples(qt_app)[0]

    qt_app.document.undo_stack.push(SetClipFxCommand(clip.id, {"gain_enabled": True, "gain_db": 6.0}))

    assert qt_app.document.dirty_clips() == []  # FX never require generation
    after = _transport_samples(qt_app)[0]
    assert clip.segments[0].audio_path.endswith("seg.wav")  # segment untouched
    assert np.max(np.abs(after)) > np.max(np.abs(before)) * 1.5
    # The raw file on disk is what it was.
    on_disk, _ = sf.read(clip.segments[0].audio_path, dtype="float32")
    assert np.array_equal(on_disk, before)


def test_project_scope_fx_reach_a_clip_with_no_character_preset(qt_app, tmp_path):
    clip = _generated_clip(qt_app, tmp_path)
    before = _transport_samples(qt_app)[0]

    qt_app.fx_dock._value_widgets["gain_db"].setValue(6.0)
    qt_app.fx_dock._enabled_checks["gain_enabled"].setChecked(True)

    config = qt_app._assemble_clip_config(clip)
    assert config["gain_enabled"] is True and config["gain_db"] == 6.0
    after = _transport_samples(qt_app)[0]
    assert np.max(np.abs(after)) > np.max(np.abs(before)) * 1.5
    assert qt_app.document.dirty_clips() == []


def test_project_scope_fx_edit_schedules_a_timeline_refresh(qt_app, tmp_path, qtbot):
    _generated_clip(qt_app, tmp_path)
    calls = []
    qt_app.refresh_timeline = lambda: calls.append(True)

    qt_app.fx_dock._value_widgets["gain_db"].setValue(3.0)

    assert qt_app.fx_dock._project_timer.isActive()
    qtbot.waitUntil(lambda: bool(calls), timeout=2000)


def test_clip_volume_edit_is_post_processing(qt_app, tmp_path):
    """The Settings tab's volume in clip scope (a `clip.overrides` write;
    the seeded Default character carries its own volume, so project scope
    wouldn't reach this clip - Q7 layering, not an FX matter)."""
    clip = _generated_clip(qt_app, tmp_path)
    before = _transport_samples(qt_app)[0]

    qt_app.selection.select_clip(clip.id)
    qt_app.settings_dock.volume_spin.setValue(0.5)

    assert clip.overrides["volume"] == 0.5
    after = _transport_samples(qt_app)[0]
    assert np.allclose(after, before * 0.5, atol=1e-6)
    assert qt_app.document.dirty_clips() == []


def test_master_apply_fx_off_silences_the_chain_but_not_volume(qt_app, tmp_path):
    clip = _generated_clip(qt_app, tmp_path)
    before = _transport_samples(qt_app)[0]
    qt_app.document.undo_stack.push(SetClipFxCommand(clip.id, {"gain_enabled": True, "gain_db": 6.0}))

    qt_app.settings_dock.apply_fx_check.setChecked(False)

    assert qt_app._assemble_clip_config(clip)["apply_fx"] is False
    assert np.array_equal(_transport_samples(qt_app)[0], before)


def test_clip_fx_override_turns_fx_on_even_if_the_character_preset_says_off(qt_app, tmp_path):
    clip = _generated_clip(qt_app, tmp_path)
    qt_app.document.characters[0].preset_data["apply_fx"] = False
    assert qt_app._assemble_clip_config(clip)["apply_fx"] is False

    qt_app.document.undo_stack.push(SetClipFxCommand(clip.id, {"gain_enabled": True, "gain_db": 6.0}))
    assert qt_app._assemble_clip_config(clip)["apply_fx"] is True

    clip.overrides["apply_fx"] = False  # an explicit clip-level off still wins
    assert qt_app._assemble_clip_config(clip)["apply_fx"] is False


def test_arrangement_measures_the_rendered_length(qt_app, tmp_path):
    text = "hello world"
    qt_app.document.text = text
    character = qt_app.document.characters[0]
    clip = qt_app.document.assign_character_to_range(0, len(text), character.id)
    path = tmp_path / "padded.wav"
    silence = np.zeros(RATE // 2, dtype=np.float32)
    tone = np.full(RATE // 2, 0.3, dtype=np.float32)
    sf.write(str(path), np.concatenate([silence, tone, silence]), RATE)
    config = qt_app._assemble_clip_config(clip)
    clip.segments = [Segment(order_index=0, text=text, audio_path=str(path), duration=1.5,
                             cache_key=compute_expected_cache_hash(text, config))]

    placed = qt_app.build_arrangement().by_clip_id()[clip.id]
    assert placed.duration_s == 1.5 and not placed.estimated

    qt_app.selection.select_clip(clip.id)
    qt_app.settings_dock.trim_check.setChecked(True)
    assert clip.overrides["trim"] is True
    placed = qt_app.build_arrangement().by_clip_id()[clip.id]
    assert placed.duration_s == 0.5
    assert qt_app.document.dirty_clips() == []


def test_segments_with_a_range_are_measured_scheduled_and_drawn_as_slices(qt_app, tmp_path):
    text = "hello world"
    qt_app.document.text = text
    character = qt_app.document.characters[0]
    clip = qt_app.document.assign_character_to_range(0, len(text), character.id)
    path = tmp_path / "recording.wav"
    ramp = (np.arange(RATE * 2) / (RATE * 2.0)).astype(np.float32)
    sf.write(str(path), ramp, RATE, subtype="FLOAT")
    # A trim setting doesn't reach a slice (post._slice_config).
    clip.overrides["trim"] = True
    clip.segments = [Segment(order_index=0, audio_path=str(path), range=[1.0, 1.5]),
                     Segment(order_index=1, audio_path=str(path), range=[0.25, 0.5])]

    placed = qt_app.build_arrangement().by_clip_id()[clip.id]
    assert placed.duration_s == 0.75 and not placed.estimated

    qt_app._rebuild_transport_schedule()
    loaded = sorted(qt_app.transport.loaded_clips(), key=lambda c: c.start_frame)
    assert [len(c.samples) for c in loaded] == [RATE // 2, RATE // 4]
    assert loaded[1].start_frame - loaded[0].start_frame == RATE // 2
    assert np.allclose(loaded[0].samples, ramp[RATE:RATE + RATE // 2], atol=1e-6)

    samples, rate = qt_app.rendered_clip_samples(clip)
    assert rate == RATE and len(samples) == RATE * 3 // 4


def test_legacy_baked_segment_is_dirty_and_a_fresh_one_is_not(qt_app, tmp_path):
    clip = _generated_clip(qt_app, tmp_path)
    assert qt_app.document.dirty_clips() == []
    clip.segments[0].raw = False
    assert [c.id for c in qt_app.document.dirty_clips()] == [clip.id]


def test_fx_dock_clip_scope_shows_what_the_config_plays(qt_app, tmp_path):
    clip = _generated_clip(qt_app, tmp_path)
    qt_app.fx_dock._value_widgets["gain_db"].setValue(2.0)  # project layer
    qt_app.document.undo_stack.push(SetClipFxCommand(clip.id, {"eq_bass": 3.0}))

    qt_app.selection.select_clip(clip.id)

    assert qt_app.fx_dock.mode == "clip"
    config = qt_app._assemble_clip_config(clip)
    assert qt_app.fx_dock._value_widgets["gain_db"].value() == config["gain_db"] == 2.0
    assert qt_app.fx_dock._value_widgets["eq_bass"].value() == config["eq_bass"] == 3.0
