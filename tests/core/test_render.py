"""kokoro_gui/core/render.py: a clip's rendered length and samples, and the
export mix, from plain values."""
import os

import numpy as np
import pytest
import soundfile as sf

from kokoro_gui.audio import loudness, post
from kokoro_gui.core import render as core_render
from kokoro_gui.daw.arrangement import compute_arrangement
from kokoro_gui.daw.mixdown import ChapterPlan, mixdown, plan_chapters
from kokoro_gui.daw.models import Clip, Document, Run, Segment

RATE = 24000
VALUES = {"format": "wav", "srt": False, "keep_clip_files": False}


@pytest.fixture(autouse=True)
def fresh_memo():
    post.clear_render_cache()
    yield
    post.clear_render_cache()


def _wav(folder, name, value, seconds=0.5):
    path = str(folder / name)
    sf.write(path, np.full(int(RATE * seconds), value, dtype=np.float32), RATE, subtype="FLOAT")
    return path


def _segment(path, order_index=0, duration=0.5, **extra):
    return Segment(order_index=order_index, text="Hello.", cache_key="k", audio_path=path, duration=duration, **extra)


def _document(folder):
    clips = [Clip(segments=[_segment(_wav(folder, "a.wav", 0.25))]),
             Clip(segments=[_segment(_wav(folder, "b.wav", 0.5))])]
    return Document(runs=[Run("One. ", clips[0].id), Run("Two.", clips[1].id)], clips=clips)


# --- clip_duration_s ---------------------------------------------------------

def _no_reads(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("read the file")

    monkeypatch.setattr(post, "render", fail)
    monkeypatch.setattr(post, "_read_mono", fail)


def test_a_trimmed_duration_comes_from_the_hint_without_a_read(monkeypatch):
    _no_reads(monkeypatch)
    clip = Clip(segments=[_segment("/gone/a.wav", duration=2.0, onset_s=0.3, tail_s=0.2)])
    assert core_render.clip_duration_s(clip, {"trim_silence": True}, RATE) == pytest.approx(1.5)
    assert core_render.clip_duration_s(clip, {}, RATE) == pytest.approx(2.0)


def test_pitch_and_stretch_change_the_duration(monkeypatch):
    _no_reads(monkeypatch)
    clip = Clip(segments=[_segment("/gone/a.wav", duration=2.0, onset_s=0.0, tail_s=0.0)])
    assert core_render.clip_duration_s(clip, {"pitch": 12.0}, RATE) == pytest.approx(1.0)
    assert core_render.clip_duration_s(clip, {"time_stretch": 1.25}, RATE) == pytest.approx(1.6)


def test_segments_add_up_and_a_range_is_its_own_length(monkeypatch):
    _no_reads(monkeypatch)
    clip = Clip(segments=[_segment("/gone/a.wav", 0, 2.0, onset_s=0.0, tail_s=0.0),
                          _segment("/gone/b.wav", 1, 9.0, range=[1.0, 1.5])])
    assert core_render.clip_duration_s(clip, {"trim_silence": True}, RATE) == pytest.approx(2.5)


def test_an_unreadable_file_counts_its_raw_duration():
    clip = Clip(segments=[_segment("/gone/a.wav", duration=0.75)])
    assert core_render.clip_duration_s(clip, {"trim_silence": True}, RATE) == 0.75


def test_no_audio_has_no_duration_and_no_samples():
    clip = Clip(segments=[_segment(None)])
    assert core_render.clip_duration_s(clip, {}, RATE) is None
    assert core_render.rendered_clip_samples(clip, {}, RATE) is None
    assert core_render.clip_duration_s(Clip(), {}, RATE) is None


def test_the_callers_segments_are_used(monkeypatch):
    monkeypatch.setattr(core_render, "playable_segments", lambda clip: pytest.fail("looked up again"))
    segments = [_segment("/gone/a.wav", duration=1.0, onset_s=0.0, tail_s=0.0)]
    assert core_render.clip_duration_s(Clip(), {}, RATE, segments) == 1.0


# --- rendered_clip_samples ---------------------------------------------------

def test_samples_are_the_segments_in_order_post_processed(tmp_path):
    clip = Clip(segments=[_segment(_wav(tmp_path, "b.wav", 0.5), 1), _segment(_wav(tmp_path, "a.wav", 0.25), 0)])
    config = {"volume": 0.5}
    samples, rate = core_render.rendered_clip_samples(clip, config, RATE)
    assert rate == RATE
    expected = [post.render(str(tmp_path / name), config, RATE) for name in ("a.wav", "b.wav")]
    np.testing.assert_array_equal(samples, np.concatenate(expected))
    assert not np.array_equal(samples, core_render.rendered_clip_samples(clip, {}, RATE)[0])


def test_an_unreadable_segment_is_left_out(tmp_path):
    clip = Clip(segments=[_segment(str(tmp_path / "gone.wav"), 0), _segment(_wav(tmp_path, "a.wav", 0.25), 1)])
    samples, _rate = core_render.rendered_clip_samples(clip, {}, RATE)
    assert len(samples) == RATE // 2
    assert core_render.rendered_clip_samples(Clip(segments=clip.segments[:1]), {}, RATE) is None


def test_the_post_config_is_asked_only_for_a_clip_with_audio():
    assert core_render.clip_render_requests(Clip(), lambda clip: pytest.fail("asked")) is None


# --- export ------------------------------------------------------------------

def test_mix_inputs_resolves_every_callback_up_front(tmp_path):
    document = _document(tmp_path)
    nested = Clip(source="nested")
    document.clips.append(nested)
    arrangement = compute_arrangement(document, clip_duration=lambda clip: 0.5)
    asked, asked_nested = [], []
    inputs = core_render.mix_inputs(arrangement, lambda clip: asked.append(clip.id) or {"volume": 0.5},
                                    lambda clip: asked_nested.append(clip.id) or "child.wav", RATE, "dummy")
    placed = [p.clip.id for p in arrangement.placed]
    assert asked == placed
    assert asked_nested == [c for c in placed if c == nested.id]
    for clip in document.clips[:2]:
        assert inputs["post_config_for_clip"](clip) == {"volume": 0.5}
        assert inputs["nested_audio_path"](clip) is None
    assert asked == placed
    assert (inputs["sample_rate"], inputs["engine_id"]) == (RATE, "dummy")


def test_export_mix_writes_what_mixdown_writes(tmp_path):
    document = _document(tmp_path)
    configs = {document.clips[0].id: {"volume": 0.5}, document.clips[1].id: {}}
    arrangement = compute_arrangement(document, clip_duration=lambda clip: 0.5)
    inputs = core_render.mix_inputs(arrangement, lambda clip: configs[clip.id], None, RATE, "dummy")
    progress = []
    options = core_render.export_options(VALUES, inputs, progress=lambda f, d: progress.append(f))
    result = core_render.export_mix(document, str(tmp_path / "core.wav"), options)
    direct = mixdown(document, str(tmp_path / "direct.wav"), sample_rate=RATE, arrangement=arrangement,
                     engine_id="dummy", post_config_for_clip=lambda clip: configs[clip.id])
    with open(result.audio_path, "rb") as ours, open(direct.audio_path, "rb") as theirs:
        assert ours.read() == theirs.read()
    samples, _rate = sf.read(result.audio_path)
    assert np.abs(samples).max() > 0.0
    assert progress


def test_export_mix_renders_only_the_range(tmp_path):
    document = _document(tmp_path)
    arrangement = compute_arrangement(document, clip_duration=lambda clip: 0.5)
    options = core_render.export_options(VALUES, core_render.mix_inputs(arrangement, lambda clip: {}, None, RATE, "dummy"))
    result = core_render.export_mix(document, str(tmp_path / "part.wav"), options, range_s=(0.0, 0.25))
    assert sf.info(result.audio_path).duration == pytest.approx(0.25, abs=0.01)


def test_export_mix_with_a_plan_writes_one_file_per_chapter(tmp_path):
    document = _document(tmp_path)
    document.settings["markers"] = [{"id": "m1", "seconds": 0.0, "name": "First"},
                                    {"id": "m2", "seconds": 0.5, "name": "Second"}]
    arrangement = compute_arrangement(document, clip_duration=lambda clip: 0.5)
    plan = plan_chapters(document, arrangement, "markers")
    assert isinstance(plan, ChapterPlan) and plan.chapters
    options = core_render.export_options(VALUES, core_render.mix_inputs(arrangement, lambda clip: {}, None, RATE, "dummy"))
    out_dir = tmp_path / "chapters"
    out_dir.mkdir()
    result = core_render.export_mix(document, str(tmp_path / "unused.wav"), options, plan=plan, out_dir=str(out_dir))
    assert sorted(os.listdir(out_dir)) == sorted(f"{c.name}.wav" for c in plan.chapters)
    assert len(result.files) == len(plan.chapters)
    assert not (tmp_path / "unused.wav").exists()


def test_export_options_default_everything_but_the_required_three():
    options = core_render.export_options(dict(VALUES, format="m4b", m4b_bitrate_kbps=64, bitrate_kbps=192),
                                         {"sample_rate": RATE}, tags={"title": "T"}, checks=("peak",))
    assert options["bitrate_kbps"] == 64
    assert (options["channels"], options["srt_granularity"], options["loudness"]) == (2, "clip", None)
    assert (options["tags"], options["checks"], options["sample_rate"]) == ({"title": "T"}, ("peak",), RATE)
    with pytest.raises(KeyError):
        core_render.export_options({"format": "wav"}, {})


def test_loudness_options(monkeypatch):
    monkeypatch.setattr(loudness, "available", lambda: True)
    assert core_render.loudness_options({}) is None
    assert core_render.loudness_options({"normalize_loudness": True, "target_lufs": -20.0}) == {
        "mode": "lufs", "target_lufs": -20.0, "ceiling_dbtp": core_render.DEFAULT_CEILING_DBTP}
    assert core_render.loudness_options({"normalize_loudness": True, "normalize_mode": "rms"}) == {
        "mode": "rms", "target_rms_dbfs": core_render.DEFAULT_RMS_DBFS,
        "limiter_dbfs": core_render.DEFAULT_LIMITER_DBFS}
    monkeypatch.setattr(loudness, "available", lambda: False)
    assert core_render.loudness_options({"normalize_loudness": True}) is None
