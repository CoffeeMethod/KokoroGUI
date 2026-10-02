"""Tests for kokoro_gui/daw/mixdown.py - offline export of a clip document
(section 6 of Claude/PLAN_ui_shell_redesign.md). Writes real wav files
into tmp_path; no engine, no Qt."""
import datetime
import os

import numpy as np
import pytest
import soundfile as sf

from kokoro_gui.daw import markers as marker_ops
from kokoro_gui.daw.arrangement import compute_arrangement
from kokoro_gui.daw.mixdown import (
    _chapter_title, _cut_long, expand_name, mixdown, mixdown_chapters, name_context, plan_chapters, render_mix,
    unused_path, write_audio, write_srt,
)
from kokoro_gui.daw.models import Character, Clip, Document, Run, Segment, Track


def _doc(text, tagged, **kwargs):
    runs, cursor = [], 0
    for start, end, clip in sorted(tagged, key=lambda t: t[0]):
        if start > cursor:
            runs.append(Run(text=text[cursor:start]))
        runs.append(Run(text=text[start:end], clip_id=clip.id, kind=clip.source))
        cursor = end
    if cursor < len(text):
        runs.append(Run(text=text[cursor:]))
    # These tests are about order and length; gaps have their own tests.
    kwargs.setdefault("settings", {"gap_s": 0.0, "paragraph_gap_s": 0.0})
    return Document(runs=runs, clips=[c for _s, _e, c in tagged], **kwargs)


def _wav(path, value, seconds, rate=8000):
    sf.write(str(path), np.full(int(rate * seconds), value, dtype=np.float32), rate)
    return str(path)


def _two_generated_clips(tmp_path):
    alice = Character.from_preset_dict("Alice", {})
    bob = Character.from_preset_dict("Bo b/ok", {})
    track_a = Track(name="A", character_id=alice.id, order_index=0)
    track_b = Track(name="B", character_id=bob.id, order_index=1)
    a = Clip(character_id=alice.id, track_id=track_a.id,
             segments=[Segment(order_index=0, duration=1.0, audio_path=_wav(tmp_path / "a.wav", 0.25, 1.0))])
    b = Clip(character_id=bob.id, track_id=track_b.id,
             segments=[Segment(order_index=0, duration=0.5, audio_path=_wav(tmp_path / "b.wav", 0.5, 0.5))])
    doc = _doc("Hello there. General Kenobi.", [(0, 12, a), (13, 28, b)],
               characters=[alice, bob], tracks=[track_a, track_b])
    return doc, a, b


def test_mixdown_lays_clips_end_to_end_and_writes_one_file(tmp_path):
    doc, _a, _b = _two_generated_clips(tmp_path)
    out = tmp_path / "out" / "mix.wav"

    result = mixdown(doc, str(out), fmt="wav", sample_rate=8000)

    data, rate = sf.read(str(out), dtype="float32")
    assert rate == 8000
    assert len(data) == 12000  # 1.0s + 0.5s
    assert np.allclose(data[:8000], 0.25, atol=1e-3)
    assert np.allclose(data[8000:], 0.5, atol=1e-3)
    assert result.audio_path == str(out)
    assert result.duration_s == 1.5
    assert result.skipped_clip_ids == []


def test_mixdown_sums_overlapping_pinned_clips(tmp_path):
    doc, a, b = _two_generated_clips(tmp_path)
    b.timeline_timestamp = 0.5  # overlaps the second half of a

    mixdown(doc, str(tmp_path / "mix.wav"), fmt="wav", sample_rate=8000)

    data, _ = sf.read(str(tmp_path / "mix.wav"), dtype="float32")
    assert len(data) == 8000
    assert np.allclose(data[:4000], 0.25, atol=1e-3)
    assert np.allclose(data[4000:], 0.75, atol=1e-3)


def test_mixdown_keeps_per_clip_files_named_by_index_and_character(tmp_path):
    doc, _a, _b = _two_generated_clips(tmp_path)

    result = mixdown(doc, str(tmp_path / "story.wav"), fmt="wav", sample_rate=8000, keep_clip_files=True)

    names = [p.replace(str(tmp_path), "").strip("\\/") for p in result.clip_files]
    assert names == ["story_001_Alice.wav", "story_002_Bo_b_ok.wav"]  # UI14, sanitized
    for path in result.clip_files:
        assert sf.info(path).frames > 0


def test_mixdown_writes_srt_from_arrangement(tmp_path):
    doc, _a, _b = _two_generated_clips(tmp_path)

    result = mixdown(doc, str(tmp_path / "mix.wav"), fmt="wav", sample_rate=8000, include_srt=True)

    text = open(result.srt_path, encoding="utf-8").read()
    assert "1\n00:00:00,000 --> 00:00:01,000\nHello there.\n" in text
    assert "2\n00:00:01,000 --> 00:00:01,500\nGeneral Kenobi.\n" in text


def test_ungenerated_clips_are_silence_and_reported_as_skipped(tmp_path):
    doc, a, b = _two_generated_clips(tmp_path)
    b.segments = []  # dirty: never generated

    # An explicit arrangement pins the estimate rate (the default asks
    # generation_stats.json, whose history varies per machine).
    arrangement = compute_arrangement(doc, chars_per_second=15.0)
    result = mixdown(doc, str(tmp_path / "mix.wav"), fmt="wav", sample_rate=8000, include_srt=True,
                     arrangement=arrangement)

    data, _ = sf.read(str(tmp_path / "mix.wav"), dtype="float32")
    assert result.skipped_clip_ids == [b.id]
    # a is 1.0s; b is estimated (15 chars / 15 cps = 1.0s) so the file is
    # padded with silence to the arrangement's total.
    assert len(data) == 16000
    assert np.allclose(data[8000:], 0.0)
    srt = open(result.srt_path, encoding="utf-8").read()
    assert "General Kenobi" not in srt


def test_write_srt_skips_estimated_and_empty_clips(tmp_path):
    doc, a, b = _two_generated_clips(tmp_path)
    arrangement = compute_arrangement(doc, chars_per_second=15.0)
    b.segments = []
    arrangement = compute_arrangement(doc, chars_per_second=15.0)

    write_srt(doc, arrangement, str(tmp_path / "x.srt"))

    assert open(tmp_path / "x.srt", encoding="utf-8").read().count("-->") == 1


def test_mixdown_applies_post_config_per_clip(tmp_path):
    """Export reads the raw segment files through the same read-time
    post-processing the transport plays (kokoro_gui/audio/post.py)."""
    doc, a, _b = _two_generated_clips(tmp_path)
    out = tmp_path / "out" / "mix.wav"

    def post_for(clip):
        return {"volume": 2.0, "apply_fx": False} if clip.id == a.id else None

    mixdown(doc, str(out), fmt="wav", sample_rate=8000, post_config_for_clip=post_for)

    data, _rate = sf.read(str(out), dtype="float32")
    assert np.allclose(data[:8000], 0.5, atol=1e-3)  # a: 0.25 * 2
    assert np.allclose(data[8000:], 0.5, atol=1e-3)  # b: untouched


def test_mixdown_reads_only_a_segments_range(tmp_path):
    """A segment with `range` exports that slice of its file, placed at its
    length (`end - start`), not the file's."""
    ramp = (np.arange(8000) / 8000.0).astype(np.float32)
    path = str(tmp_path / "ramp.wav")
    sf.write(path, ramp, 8000, subtype="FLOAT")
    clip = Clip(segments=[Segment(order_index=0, audio_path=path, range=[0.5, 0.75]),
                          Segment(order_index=1, audio_path=path, range=[0.0, 0.25])])
    doc = _doc("Sliced.", [(0, 7, clip)])

    result = mixdown(doc, str(tmp_path / "mix.wav"), fmt="wav", sample_rate=8000, channels=1)

    data, _rate = sf.read(str(tmp_path / "mix.wav"), dtype="float32")
    assert result.duration_s == 0.5
    assert np.allclose(data, np.concatenate([ramp[4000:6000], ramp[:2000]]), atol=1e-4)


# -- phase 2: stereo, track controls, range, cue sheet, word SRT -----------------


def test_mixdown_is_stereo_by_default_and_mono_averages(tmp_path):
    doc, _a, _b = _two_generated_clips(tmp_path)
    doc.tracks[0].pan = -1.0

    mixdown(doc, str(tmp_path / "st.wav"), fmt="wav", sample_rate=8000)
    mixdown(doc, str(tmp_path / "mono.wav"), fmt="wav", sample_rate=8000, channels=1)

    stereo, _ = sf.read(str(tmp_path / "st.wav"), dtype="float32")
    mono, _ = sf.read(str(tmp_path / "mono.wav"), dtype="float32")
    assert stereo.shape == (12000, 2)
    assert np.allclose(stereo[:8000, 1], 0.0, atol=1e-3)  # a panned hard left
    assert np.allclose(stereo[:8000, 0], 0.25 * 2 ** 0.5, atol=1e-3)
    assert mono.ndim == 1
    assert np.allclose(mono[8000:], 0.5, atol=1e-3)


def test_mixdown_honours_mute_and_solo(tmp_path):
    doc, a, b = _two_generated_clips(tmp_path)
    doc.tracks[1].mute = True
    mixdown(doc, str(tmp_path / "m.wav"), fmt="wav", sample_rate=8000)
    data, _ = sf.read(str(tmp_path / "m.wav"), dtype="float32")
    assert np.allclose(data[8000:], 0.0)

    doc.tracks[1].mute = False
    doc.tracks[1].solo = True
    mixdown(doc, str(tmp_path / "s.wav"), fmt="wav", sample_rate=8000)
    data, _ = sf.read(str(tmp_path / "s.wav"), dtype="float32")
    assert np.allclose(data[:8000], 0.0)
    assert np.allclose(data[8000:], 0.5, atol=1e-3)


def test_mixdown_applies_clip_fades(tmp_path):
    doc, a, _b = _two_generated_clips(tmp_path)
    a.fade_in_s = 0.5
    mixdown(doc, str(tmp_path / "f.wav"), fmt="wav", sample_rate=8000, channels=1)
    data, _ = sf.read(str(tmp_path / "f.wav"), dtype="float32")
    assert abs(data[2000] - 0.125) < 2e-3  # halfway up a 0.5s fade on a 0.25 clip
    assert abs(data[6000] - 0.25) < 2e-3


def test_mixdown_range_renders_only_the_region(tmp_path):
    doc, _a, _b = _two_generated_clips(tmp_path)
    result = mixdown(doc, str(tmp_path / "r.wav"), fmt="wav", sample_rate=8000, channels=1,
                     range_s=(0.5, 1.25), include_srt=True)
    data, _ = sf.read(str(tmp_path / "r.wav"), dtype="float32")
    assert len(data) == 6000
    assert np.allclose(data[:4000], 0.25, atol=1e-3)
    assert np.allclose(data[4000:], 0.5, atol=1e-3)
    assert result.duration_s == 0.75
    assert "00:00:00,500 --> 00:00:01,000\nGeneral Kenobi." in open(result.srt_path, encoding="utf-8").read()


def test_cue_sheet_rows_use_timecode_when_enabled(tmp_path):
    import csv

    doc, a, b = _two_generated_clips(tmp_path)
    a.status, a.note, a.source_text = "approved", "nice", "Bonjour."
    doc.settings["timecode"] = {"enabled": True, "frame_rate": 25.0}
    result = mixdown(doc, str(tmp_path / "c.wav"), fmt="wav", sample_rate=8000, include_cue_sheet=True)

    with open(result.cue_sheet_path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["start", "end", "character", "source_text", "text", "status", "note"]
    assert rows[1] == ["00:00:00:00", "00:00:01:00", "Alice", "Bonjour.", "Hello there.", "approved", "nice"]
    assert rows[2][:3] == ["00:00:01:00", "00:00:01:12", "Bo b/ok"]
    assert rows[2][5] == "todo"


def test_word_level_srt_uses_stored_word_times(tmp_path):
    doc, a, _b = _two_generated_clips(tmp_path)
    a.segments[0].words = [["Hello", 0.0, 0.4], ["there.", 0.5, 0.9]]
    arrangement = compute_arrangement(doc, chars_per_second=15.0)

    write_srt(doc, arrangement, str(tmp_path / "w.srt"), granularity="word")

    text = open(tmp_path / "w.srt", encoding="utf-8").read()
    assert "1\n00:00:00,000 --> 00:00:00,400\nHello\n" in text
    assert "2\n00:00:00,500 --> 00:00:00,900\nthere.\n" in text
    assert "Kenobi" not in text  # b has no stored words


def test_a_soloed_track_without_clips_silences_nothing(tmp_path):
    """Grill PR4: an unused track isn't drawn, so its solo can't be turned
    off; it must not mute the tracks that are there."""
    from kokoro_gui.daw.models import Track

    doc, _a, _b = _two_generated_clips(tmp_path)
    doc.tracks.append(Track(name="unused", solo=True, order_index=9))
    mixdown(doc, str(tmp_path / "u.wav"), fmt="wav", sample_rate=8000)
    data, _ = sf.read(str(tmp_path / "u.wav"), dtype="float32")
    assert not np.allclose(data[:8000], 0.0)
    assert not np.allclose(data[8000:], 0.0)


def _recording_doc(tmp_path, values, words, rate=8000):
    """An imported recording clip over one source file whose samples are
    `values`, with `words` as the run's words over the text "a b c"."""
    path = tmp_path / "rec.wav"
    sf.write(str(path), np.asarray(values, dtype=np.float32), rate)
    clip = Clip(source="imported")
    doc = Document(runs=[Run(text="a b c", clip_id=clip.id, kind="imported", words=words)], clips=[clip],
                   settings={"gap_s": 0.0, "paragraph_gap_s": 0.0,
                             "sources": {"rec": {"path": str(path), "sample_rate": rate, "duration_s": 3.0}}})
    doc.refresh_imported_segments()
    return doc, clip


def test_an_imported_clip_with_a_deleted_word_closes_up_with_a_crossfade(tmp_path):
    """Phase 5 P3: the ranges left after a delete play end to end, and the
    join crossfades instead of cutting. On a steady tone the crossfade sums
    back to the tone, where two plain fades would dip to silence."""
    doc, clip = _recording_doc(tmp_path, np.full(8000 * 3, 0.5),
                               [[0, 1, "rec", 0.0, 1.0], [2, 3, "rec", 1.0, 2.0], [4, 5, "rec", 2.0, 3.0]])
    doc.replace_text(1, 2, 0, "a c")  # " b": the middle word
    assert [s.range for s in clip.segments] == [[0.0, 1.0], [2.0, 3.0]]

    mixdown(doc, str(tmp_path / "r.wav"), fmt="wav", sample_rate=8000)

    data, _ = sf.read(str(tmp_path / "r.wav"), dtype="float32")
    assert len(data) == 16000  # 1 s + 1 s: the gap closed up
    assert np.allclose(data, 0.5, atol=1e-3)


def test_an_imported_clip_crossfade_blends_the_two_sides_of_the_cut(tmp_path):
    values = np.concatenate([np.full(8000, 0.2), np.full(8000, 0.9), np.full(8000, -0.4)])
    doc, _clip = _recording_doc(tmp_path, values, [[0, 1, "rec", 0.0, 1.0], [4, 5, "rec", 2.0, 3.0]])
    mixdown(doc, str(tmp_path / "r.wav"), fmt="wav", sample_rate=8000)
    data = sf.read(str(tmp_path / "r.wav"), dtype="float32")[0][:, 0]
    xfade = 40  # 5 ms at 8 kHz
    assert len(data) == 16000
    assert np.allclose(data[:8000], 0.2, atol=1e-3)
    assert np.allclose(data[8000 + xfade:], -0.4, atol=1e-3)
    # Inside the join: the deleted word's first 5 ms fading out while the
    # next range fades in.
    ramp = np.arange(xfade) / xfade
    assert np.allclose(data[8000:8000 + xfade], 0.9 * (1 - ramp) - 0.4 * ramp, atol=2e-3)


def test_srt_and_cue_sheet_leave_out_inline_markup(tmp_path):
    from kokoro_gui.daw.mixdown import write_cue_sheet

    alice = Character.from_preset_dict("Alice", {})
    track = Track(name="A", character_id=alice.id, order_index=0)
    clip = Clip(character_id=alice.id, track_id=track.id,
                segments=[Segment(order_index=0, duration=1.0, audio_path=_wav(tmp_path / "a.wav", 0.25, 1.0))])
    text = "[Alice:Radio]: Hello there."
    doc = _doc(text, [(0, len(text), clip)], characters=[alice], tracks=[track])
    arrangement = compute_arrangement(doc)

    srt = open(write_srt(doc, arrangement, str(tmp_path / "a.srt")), encoding="utf-8").read()
    cues = open(write_cue_sheet(doc, arrangement, str(tmp_path / "a.csv")), encoding="utf-8").read()
    assert "\nHello there.\n" in srt and "[Alice" not in srt
    assert "Hello there." in cues and "[Alice" not in cues


# -- render_mix and the loudness option (plan 12) ----------------------------


def test_render_mix_returns_the_mix_without_writing_anything(tmp_path):
    doc, _a, _b = _two_generated_clips(tmp_path)

    mix = render_mix(doc, 8000)

    assert mix.samples.shape == (12000, 2)
    assert mix.samples.dtype == np.float32
    assert np.allclose(mix.samples[:8000], 0.25, atol=1e-3)
    assert mix.duration_s == 1.5
    assert mix.skipped == []
    assert [index for index, _placed, _samples in mix.per_clip] == [0, 1]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.wav", "b.wav"]
    assert render_mix(doc, 8000, channels=1).samples.shape == (12000,)


def test_mixdown_without_loudness_writes_what_render_mix_returns(tmp_path):
    doc, _a, _b = _two_generated_clips(tmp_path)

    result = mixdown(doc, str(tmp_path / "out.wav"), fmt="wav", sample_rate=8000)
    write_audio(str(tmp_path / "ref.wav"), render_mix(doc, 8000).samples, 8000, "wav")

    assert (tmp_path / "out.wav").read_bytes() == (tmp_path / "ref.wav").read_bytes()
    assert result.loudness_before is None and result.loudness_after is None
    assert result.loudness_limited is False


def _tone_doc(tmp_path, amplitude, seconds=4.0, rate=8000):
    t = np.arange(int(rate * seconds)) / rate
    path = tmp_path / "tone.wav"
    sf.write(str(path), (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32), rate)
    alice = Character.from_preset_dict("Alice", {})
    track = Track(name="A", character_id=alice.id, order_index=0)
    clip = Clip(character_id=alice.id, track_id=track.id,
                segments=[Segment(order_index=0, duration=seconds, audio_path=str(path))])
    return _doc("Hello there.", [(0, 12, clip)], characters=[alice], tracks=[track])


def test_mixdown_normalizes_to_the_target_and_reports_before_and_after(tmp_path):
    pytest.importorskip("pyloudnorm")
    import pyloudnorm

    doc = _tone_doc(tmp_path, 0.05)
    out = tmp_path / "norm.wav"

    result = mixdown(doc, str(out), fmt="wav", sample_rate=8000,
                     loudness={"target_lufs": -16.0, "ceiling_dbtp": -1.0})

    data, rate = sf.read(str(out), dtype="float64")
    assert pyloudnorm.Meter(rate).integrated_loudness(data) == pytest.approx(-16.0, abs=0.5)
    assert result.loudness_before.integrated_lufs < -20.0
    assert result.loudness_after.integrated_lufs == pytest.approx(-16.0, abs=0.5)
    assert result.loudness_after.true_peak_dbtp <= -1.0 + 0.01
    assert result.loudness_limited is False


def test_mixdown_normalize_stops_at_the_ceiling_and_says_so(tmp_path):
    pytest.importorskip("pyloudnorm")
    doc = _tone_doc(tmp_path, 0.9)  # a -0.9 dBFS peak (about -0.5 LUFS in stereo) cannot reach -5 LUFS under -6 dBTP

    result = mixdown(doc, str(tmp_path / "limited.wav"), fmt="wav", sample_rate=8000,
                     loudness={"target_lufs": -5.0, "ceiling_dbtp": -6.0})

    assert result.loudness_limited is True
    assert result.loudness_after.true_peak_dbtp == pytest.approx(-6.0, abs=0.1)
    assert result.loudness_after.integrated_lufs < -5.0


def test_mixdown_normalize_leaves_per_clip_files_alone(tmp_path):
    pytest.importorskip("pyloudnorm")
    doc = _tone_doc(tmp_path, 0.05)

    result = mixdown(doc, str(tmp_path / "n.wav"), fmt="wav", sample_rate=8000, keep_clip_files=True,
                     loudness={"target_lufs": -16.0, "ceiling_dbtp": -1.0})

    clip, _ = sf.read(result.clip_files[0], dtype="float32")
    assert np.max(np.abs(clip)) == pytest.approx(0.05, abs=1e-3)


# -- export options: mp3 bitrate, output rate, file-name template -----------------------


def _noise_doc(tmp_path, seconds=2.0, rate=48000):
    rng = np.random.default_rng(7)
    path = tmp_path / "noise.wav"
    sf.write(str(path), (rng.standard_normal(int(rate * seconds)) * 0.1).astype(np.float32), rate)
    alice = Character.from_preset_dict("Alice", {})
    track = Track(name="A", character_id=alice.id, order_index=0)
    clip = Clip(character_id=alice.id, track_id=track.id,
                segments=[Segment(order_index=0, duration=seconds, audio_path=str(path))])
    return _doc("Hello there.", [(0, 12, clip)], characters=[alice], tracks=[track])


def test_mp3_bitrate_sets_the_file_size(tmp_path):
    doc = _noise_doc(tmp_path)

    sizes = {}
    for kbps in (128, 320):
        out = tmp_path / f"out{kbps}.mp3"
        mixdown(doc, str(out), fmt="mp3", sample_rate=48000, bitrate_kbps=kbps)
        sizes[kbps] = out.stat().st_size

    assert sizes[320] > sizes[128] * 2
    assert sizes[128] * 8 / 2.0 / 1000 == pytest.approx(128, rel=0.1)  # constant bitrate, not a quality target


def test_out_rate_writes_the_mixdown_at_that_rate_and_length(tmp_path):
    doc, _a, _b = _two_generated_clips(tmp_path)

    result = mixdown(doc, str(tmp_path / "r.wav"), fmt="wav", sample_rate=8000, out_rate=48000,
                     keep_clip_files=True, include_srt=True)

    info = sf.info(str(tmp_path / "r.wav"))
    assert info.samplerate == 48000 and info.frames == 72000
    assert result.duration_s == pytest.approx(1.5)
    assert sf.info(result.clip_files[0]).samplerate == 8000  # per-clip files keep the project rate
    data, _ = sf.read(str(tmp_path / "r.wav"))
    assert np.allclose(data[8000:40000], 0.25, atol=0.01)  # the level survives the resample


def test_out_rate_equal_to_the_mix_rate_writes_the_same_bytes(tmp_path):
    doc, _a, _b = _two_generated_clips(tmp_path)

    mixdown(doc, str(tmp_path / "plain.wav"), fmt="wav", sample_rate=8000)
    mixdown(doc, str(tmp_path / "same.wav"), fmt="wav", sample_rate=8000, out_rate=8000)

    assert (tmp_path / "plain.wav").read_bytes() == (tmp_path / "same.wav").read_bytes()


def test_loudness_is_measured_at_the_output_rate(tmp_path):
    pytest.importorskip("pyloudnorm")
    doc = _tone_doc(tmp_path, 0.05)

    result = mixdown(doc, str(tmp_path / "n.wav"), fmt="wav", sample_rate=8000, out_rate=16000,
                     loudness={"target_lufs": -16.0, "ceiling_dbtp": -1.0})

    assert sf.info(str(tmp_path / "n.wav")).samplerate == 16000
    assert result.loudness_after.duration_s == pytest.approx(4.0, abs=0.01)


def test_expand_name_fills_the_tokens():
    context = name_context("My Book", range_label="Intro to Ch1", now=datetime.datetime(2026, 10, 2, 9, 5, 7))

    assert expand_name("{project}-{date}", context) == "My Book-2026-10-02"
    assert expand_name("{project}_{time}_{range}", context) == "My Book_090507_Intro to Ch1"
    assert expand_name("{project}", name_context("")) == "Untitled"
    assert expand_name("{range}", name_context("x")) == "full"


def test_expand_name_keeps_an_unknown_token_as_typed():
    assert expand_name("{project}-{chapter}", name_context("Book")) == "Book-{chapter}"


def test_expand_name_cannot_leave_the_folder_or_hold_bad_characters():
    context = name_context("a/b:c*")

    assert expand_name("../../evil", context) == "evil"
    assert expand_name("..", context) == "output"
    assert expand_name('x<y>:"z"|?*', context) == "xyz"
    assert expand_name("{project}", context) == "abc"  # a value can't add a folder either
    assert expand_name("", context) == "output"
    assert expand_name("name. ", context) == "name"
    assert expand_name("a\x00b\x1fc", context) == "abc"


def test_unused_path_adds_a_number(tmp_path):
    target = tmp_path / "mix.wav"
    assert unused_path(str(target)) == str(target)
    target.write_bytes(b"x")
    assert unused_path(str(target)) == str(tmp_path / "mix (2).wav")
    (tmp_path / "mix (2).wav").write_bytes(b"x")
    assert unused_path(str(target)) == str(tmp_path / "mix (3).wav")


# -- RMS mode, the limiter, head and tail silence ------------------------------------------------


def _peaky_doc(tmp_path, seconds=6.0, rate=8000):
    """Noise at about -26 dBFS RMS with a few spikes, so the RMS target needs the limiter."""
    rng = np.random.default_rng(11)
    x = (rng.standard_normal(int(rate * seconds)) * 0.05).astype(np.float32)
    x[[2000, 20000, 33000]] = 0.5
    path = tmp_path / "peaky.wav"
    sf.write(str(path), x, rate)
    alice = Character.from_preset_dict("Alice", {})
    track = Track(name="A", character_id=alice.id, order_index=0)
    clip = Clip(character_id=alice.id, track_id=track.id,
                segments=[Segment(order_index=0, duration=seconds, audio_path=str(path))])
    return _doc("Hello there.", [(0, 12, clip)], characters=[alice], tracks=[track])


def test_rms_mode_reaches_the_target_and_holds_the_limiter_ceiling(tmp_path):
    pytest.importorskip("pyloudnorm")
    doc = _peaky_doc(tmp_path)

    result = mixdown(doc, str(tmp_path / "rms.wav"), fmt="wav", sample_rate=8000, channels=1,
                     loudness={"mode": "rms", "target_rms_dbfs": -20.0, "limiter_dbfs": -3.5})

    data, _ = sf.read(str(tmp_path / "rms.wav"), dtype="float64")
    assert result.loudness_before.rms_dbfs == pytest.approx(-26.0, abs=1.0)
    assert 20 * np.log10(np.sqrt(np.mean(data ** 2))) == pytest.approx(-20.0, abs=0.5)
    assert 20 * np.log10(np.abs(data).max()) <= -3.5 + 0.1
    assert result.loudness_after.sample_peak_dbfs <= -3.5 + 0.1
    assert result.loudness_limited is False


def test_rms_mode_says_when_the_limiter_kept_it_short(tmp_path):
    pytest.importorskip("pyloudnorm")
    doc = _peaky_doc(tmp_path)

    result = mixdown(doc, str(tmp_path / "short.wav"), fmt="wav", sample_rate=8000, channels=1,
                     loudness={"mode": "rms", "target_rms_dbfs": -8.0, "limiter_dbfs": -3.5})

    assert result.loudness_limited is True
    assert result.loudness_after.sample_peak_dbfs <= -3.5 + 0.1


def test_a_loudness_dict_without_a_mode_is_still_lufs(tmp_path):
    pytest.importorskip("pyloudnorm")
    doc = _tone_doc(tmp_path, 0.05)

    result = mixdown(doc, str(tmp_path / "n.wav"), fmt="wav", sample_rate=8000,
                     loudness={"mode": "lufs", "target_lufs": -16.0, "ceiling_dbtp": -1.0})

    assert result.loudness_after.integrated_lufs == pytest.approx(-16.0, abs=0.5)


def test_head_and_tail_pad_the_file_and_move_the_subtitles(tmp_path):
    doc, _a, _b = _two_generated_clips(tmp_path)

    result = mixdown(doc, str(tmp_path / "p.wav"), fmt="wav", sample_rate=8000, head_s=0.75, tail_s=2.0,
                     include_srt=True, include_cue_sheet=True)

    data, _ = sf.read(str(tmp_path / "p.wav"), dtype="float32")
    assert len(data) == int(8000 * (0.75 + 1.5 + 2.0))
    assert not data[:6000].any() and not data[-16000:].any()
    assert np.allclose(data[6000:14000], 0.25, atol=1e-3)
    assert result.duration_s == pytest.approx(4.25)
    assert "00:00:00,750 --> 00:00:01,750" in (tmp_path / "p.srt").read_text()
    assert (tmp_path / "p.csv").read_text().splitlines()[1].startswith("0.750,1.750")


def test_no_head_or_tail_writes_the_same_bytes(tmp_path):
    doc, _a, _b = _two_generated_clips(tmp_path)

    mixdown(doc, str(tmp_path / "plain.wav"), fmt="wav", sample_rate=8000)
    mixdown(doc, str(tmp_path / "zero.wav"), fmt="wav", sample_rate=8000, head_s=0.0, tail_s=0.0)

    assert (tmp_path / "plain.wav").read_bytes() == (tmp_path / "zero.wav").read_bytes()


def test_the_report_measures_the_padded_file(tmp_path):
    pytest.importorskip("pyloudnorm")
    doc = _tone_doc(tmp_path, 0.05)

    result = mixdown(doc, str(tmp_path / "n.wav"), fmt="wav", sample_rate=8000, head_s=1.0, tail_s=1.0,
                     loudness={"target_lufs": -16.0, "ceiling_dbtp": -1.0})

    assert result.loudness_after.duration_s == pytest.approx(6.0, abs=0.01)
    assert result.files[0].report is result.loudness_after


def test_checks_run_on_the_written_file_and_the_failures_are_listed(tmp_path):
    pytest.importorskip("pyloudnorm")
    from kokoro_gui.daw.export_presets import Check

    doc = _tone_doc(tmp_path, 0.05)
    checks = (Check("Peak", "sample_peak_dbfs", high=-60.0), Check("Length", "duration_min", high=1.0))

    result = mixdown(doc, str(tmp_path / "c.wav"), fmt="wav", sample_rate=8000, checks=checks)

    (entry,) = result.files
    assert entry.path == str(tmp_path / "c.wav") and entry.report is not None
    assert entry.failed == ["Peak -26.0 dBFS, needs at most -60 dBFS"]


# -- split exports: one file per subproject or marker range -------------------------------------

RATE = 8000


def _book(tmp_path, titles=("Chapter 1", "Chapter: 2/B")):
    """Intro (1 s of 0.1), one nested clip per title (2 s of 0.2, then 1 s of 0.3...), Outro (0.5 s).
    Returns `(doc, arrangement, nested_audio_path)`."""
    narrator = Character.from_preset_dict("Narrator", {})
    doc = Document.from_plain_text("Intro.\n\nOutro.", characters=[narrator])
    doc.settings.update({"gap_s": 0.0, "paragraph_gap_s": 0.0})
    intro = doc.assign_character_to_range(0, 6, narrator.id)
    outro = doc.assign_character_to_range(8, 14, narrator.id)
    intro.segments = [Segment(order_index=0, duration=1.0, audio_path=_wav(tmp_path / "intro.wav", 0.1, 1.0, RATE))]
    outro.segments = [Segment(order_index=0, duration=0.5, audio_path=_wav(tmp_path / "outro.wav", 0.4, 0.5, RATE))]
    durations, paths, position = {intro.id: 1.0, outro.id: 0.5}, {}, 8
    for number, title in enumerate(titles, start=1):
        chapter = doc.insert_nested_clip(position, {"kind": "embedded", "id": f"c{number}"}, title)
        position = doc.clip_extent(chapter.id)[1]
        seconds = 3.0 - number
        durations[chapter.id] = seconds
        paths[chapter.id] = _wav(tmp_path / f"c{number}.wav", 0.1 * (number + 1), seconds, RATE)
    arrangement = compute_arrangement(doc, clip_duration=lambda c: durations.get(c.id), chars_per_second=15.0)
    return doc, arrangement, lambda clip: paths.get(clip.id)


def test_plan_chapters_by_subproject_names_and_spans_them(tmp_path):
    doc, arrangement, _nested = _book(tmp_path)

    plan = plan_chapters(doc, arrangement, "subprojects")

    assert [c.name for c in plan.chapters] == ["01 - Chapter 1", "02 - Chapter 2B"]
    assert [c.range_s for c in plan.chapters] == [(1.0, 3.0), (3.0, 4.0)]
    assert plan.warnings == ["2 clips outside any subproject weren't exported"]


def test_plan_chapters_by_marker_pairs_names_by_the_first_marker(tmp_path):
    doc, arrangement, _nested = _book(tmp_path)
    for seconds, name in ((0.0, "Opening"), (1.0, ""), (4.0, "End")):
        doc.settings["markers"], _m = marker_ops.add_marker(doc.settings, seconds, name)

    plan = plan_chapters(doc, arrangement, "markers")

    assert [c.name for c in plan.chapters] == ["01 - Opening", "02 - M2"]
    assert [c.range_s for c in plan.chapters] == [(0.0, 1.0), (1.0, 4.0)]
    assert plan.warnings == ["1 clip outside any marker range weren't exported"]


def test_plan_chapters_with_nothing_to_split_is_empty_and_an_unknown_mode_is_refused(tmp_path):
    doc, arrangement, _a = _two_generated_clips_arranged(tmp_path)

    assert plan_chapters(doc, arrangement, "subprojects").chapters == []
    assert plan_chapters(doc, arrangement, "markers").chapters == []
    with pytest.raises(ValueError):
        plan_chapters(doc, arrangement, "pages")


def _two_generated_clips_arranged(tmp_path):
    doc, a, b = _two_generated_clips(tmp_path)
    return doc, compute_arrangement(doc), None


def test_a_long_chapter_is_cut_at_the_clip_boundary_nearest_before_the_limit():
    # limit 10 s -> cuts land at or before 10 - min(60, 10/120) = 9.9167 s past each start
    boundaries = [0.0, 3.0, 9.0, 9.5, 12.0, 20.0, 30.0]
    assert _cut_long(0.0, 25.0, boundaries, 10.0) == [(0.0, 9.5), (9.5, 12.0), (12.0, 20.0), (20.0, 25.0)]
    assert _cut_long(0.0, 8.0, boundaries, 10.0) == [(0.0, 8.0)]
    assert _cut_long(0.0, 25.0, boundaries, None) == [(0.0, 25.0)]
    # no boundary in reach: cut at the limit
    assert _cut_long(0.0, 25.0, [0.0, 30.0], 10.0)[0][1] == pytest.approx(10.0 - 10.0 / 120)


def test_plan_chapters_names_the_parts_of_a_long_chapter(tmp_path):
    doc, arrangement, _nested = _book(tmp_path)

    plan = plan_chapters(doc, arrangement, "subprojects", max_file_s=1.5)

    names = [c.name for c in plan.chapters]
    assert names[0].startswith("01 - Chapter 1 part ") and names[0].endswith("part 1")
    assert names[1].endswith("part 2") and names[-1] == "02 - Chapter 2B"
    assert plan.chapters[0].range_s[0] == 1.0 and plan.chapters[1].range_s[1] == 3.0


def test_titles_are_made_safe_for_a_filename():
    assert _chapter_title('A "bad" <name>: x/y') == "A bad name xy"
    assert _chapter_title("  ..  ") == "Untitled"
    assert len(_chapter_title("x" * 300)) == 100


def test_mixdown_chapters_writes_one_file_per_subproject_with_their_lengths(tmp_path):
    doc, arrangement, nested = _book(tmp_path)
    plan = plan_chapters(doc, arrangement, "subprojects")

    result = mixdown_chapters(doc, str(tmp_path / "out"), plan, fmt="wav", sample_rate=RATE,
                              arrangement=arrangement, nested_audio_path=nested, include_srt=True)

    first, second = tmp_path / "out" / "01 - Chapter 1.wav", tmp_path / "out" / "02 - Chapter 2B.wav"
    assert [f.path for f in result.files] == [str(first), str(second)]
    assert result.audio_path == str(first)
    a, _ = sf.read(str(first), dtype="float32")
    b, _ = sf.read(str(second), dtype="float32")
    assert len(a) == 2 * RATE and np.allclose(a, 0.2, atol=1e-3)
    assert len(b) == 1 * RATE and np.allclose(b, 0.3, atol=1e-3)
    assert result.duration_s == pytest.approx(3.0)
    assert result.warnings == ["2 clips outside any subproject weren't exported"]
    assert (tmp_path / "out" / "01 - Chapter 1.srt").exists() and (tmp_path / "out" / "02 - Chapter 2B.srt").exists()
    assert [f.title for f in result.files] == ["01 - Chapter 1", "02 - Chapter 2B"]


def test_mixdown_chapters_by_marker_range_reads_the_clips_inside(tmp_path):
    doc, arrangement, nested = _book(tmp_path)
    for seconds, name in ((0.0, "Intro"), (1.0, "Rest"), (4.5, "End")):
        doc.settings["markers"], _m = marker_ops.add_marker(doc.settings, seconds, name)
    plan = plan_chapters(doc, arrangement, "markers")

    result = mixdown_chapters(doc, str(tmp_path / "m"), plan, fmt="wav", sample_rate=RATE,
                              arrangement=arrangement, nested_audio_path=nested)

    intro, _ = sf.read(str(tmp_path / "m" / "01 - Intro.wav"), dtype="float32")
    rest, _ = sf.read(str(tmp_path / "m" / "02 - Rest.wav"), dtype="float32")
    assert len(intro) == RATE and np.allclose(intro, 0.1, atol=1e-3)
    assert len(rest) == int(3.5 * RATE) and np.allclose(rest[-4000:], 0.4, atol=1e-3)
    assert result.warnings == []


def test_mixdown_chapters_numbered_keeps_files_that_exist(tmp_path):
    doc, arrangement, nested = _book(tmp_path)
    plan = plan_chapters(doc, arrangement, "subprojects")
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "01 - Chapter 1.wav").write_bytes(b"old")

    result = mixdown_chapters(doc, str(tmp_path / "out"), plan, fmt="wav", sample_rate=RATE, numbered=True,
                              arrangement=arrangement, nested_audio_path=nested)

    assert (tmp_path / "out" / "01 - Chapter 1.wav").read_bytes() == b"old"
    assert result.files[0].path == str(tmp_path / "out" / "01 - Chapter 1 (2).wav")


def test_mixdown_chapters_checks_and_pads_each_file(tmp_path):
    pytest.importorskip("pyloudnorm")
    from kokoro_gui.daw.export_presets import Check

    doc, arrangement, nested = _book(tmp_path)
    plan = plan_chapters(doc, arrangement, "subprojects")

    result = mixdown_chapters(doc, str(tmp_path / "out"), plan, fmt="wav", sample_rate=RATE, head_s=0.5, tail_s=0.5,
                              arrangement=arrangement, nested_audio_path=nested,
                              checks=(Check("Length", "duration_min", high=2.5 / 60),))

    assert [round(f.duration_s, 2) for f in result.files] == [3.0, 2.0]
    assert len(result.files[0].failed) == 1 and result.files[1].failed == []
    assert all(f.report is not None for f in result.files)


def test_mixdown_chapters_reports_a_subproject_without_a_mixdown_as_skipped(tmp_path):
    doc, arrangement, nested = _book(tmp_path)
    missing = doc.nested_clips()[1]
    plan = plan_chapters(doc, arrangement, "subprojects")

    result = mixdown_chapters(doc, str(tmp_path / "out"), plan, fmt="wav", sample_rate=RATE, arrangement=arrangement,
                              nested_audio_path=lambda clip: None if clip.id == missing.id else nested(clip))

    assert result.skipped_clip_ids == [missing.id]


# -- stems (plan 15) ---------------------------------------------------------


def _stem_doc(tmp_path, bed=True, rate=8000):
    """Alice (0.25, 1 s) and Bob (0.5, 1 s, pinned to overlap her second half)
    on two tracks, and, with `bed`, a 4 s music bed (0.4) on a ducked track."""
    alice = Character.from_preset_dict("Alice", {})
    bob = Character.from_preset_dict("Bo b/ok", {})
    track_a = Track(name="Alice track", character_id=alice.id, order_index=0)
    track_b = Track(name="Bob track", character_id=bob.id, order_index=1)
    a = Clip(character_id=alice.id, track_id=track_a.id,
             segments=[Segment(order_index=0, duration=1.0, audio_path=_wav(tmp_path / "a.wav", 0.25, 1.0, rate))])
    b = Clip(character_id=bob.id, track_id=track_b.id, timeline_timestamp=0.5,
             segments=[Segment(order_index=0, duration=1.0, audio_path=_wav(tmp_path / "b.wav", 0.5, 1.0, rate))])
    tagged = [(0, 12, a), (13, 28, b)]
    tracks = [track_a, track_b]
    text = "Hello there. General Kenobi."
    if bed:
        track_m = Track(name="Music", order_index=2, role="music", duck=True)
        music = Clip(source="imported", original_audio_path=_wav(tmp_path / "bed.wav", 0.4, 4.0, rate),
                     track_id=track_m.id, timeline_timestamp=0.0)
        text += " Bed"
        tagged.append((29, 32, music))
        tracks.append(track_m)
    doc = _doc(text, tagged, characters=[alice, bob], tracks=tracks)
    if bed:
        doc.runs[-1].kind = "placeholder"
    return doc, a, b


def _sum(stems):
    return np.sum([s.samples for s in stems], axis=0)


def test_track_stems_are_the_same_length_and_add_up_to_the_mix(tmp_path):
    doc, _a, _b = _stem_doc(tmp_path, bed=False)

    mix = render_mix(doc, 8000, stems="track")

    assert [s.name for s in mix.stems] == ["Alice track", "Bob track"]
    assert all(s.samples.shape == mix.samples.shape for s in mix.stems)
    assert np.allclose(_sum(mix.stems), mix.samples, atol=1e-6)
    assert np.allclose(mix.stems[0].samples[:4000], 0.25, atol=1e-3)
    assert np.allclose(mix.stems[1].samples[:4000], 0.0)  # Bob starts at 0.5 s: the stem is padded to line up


def test_character_stems_follow_the_characters_and_keep_beds_apart(tmp_path):
    doc, a, _b = _stem_doc(tmp_path)
    a.character_id = None  # no character any more

    mix = render_mix(doc, 8000, stems="character")

    assert [s.name for s in mix.stems] == ["Bo b/ok", "Unassigned", "Music"]


def test_a_muted_or_soloed_out_track_has_no_stem_and_a_dead_clip_none_either(tmp_path):
    doc, a, b = _stem_doc(tmp_path, bed=False)
    doc.tracks[1].mute = True
    assert [s.name for s in render_mix(doc, 8000, stems="track").stems] == ["Alice track"]
    doc.tracks[1].mute = False
    doc.tracks[0].solo = True
    assert [s.name for s in render_mix(doc, 8000, stems="character").stems] == ["Alice"]
    doc.tracks[0].solo = False
    b.segments = []  # never generated: silent, so no stem
    assert [s.name for s in render_mix(doc, 8000, stems="track").stems] == ["Alice track"]


def test_the_dialogue_stem_leaves_out_the_bed(tmp_path):
    (tmp_path / "plain").mkdir()
    doc, _a, _b = _stem_doc(tmp_path)
    plain, _a, _b = _stem_doc(tmp_path / "plain", bed=False)

    mix = render_mix(doc, 8000, dialogue_stem=True)
    speech_only = render_mix(plain, 8000)

    assert [s.name for s in mix.stems] == ["Dialogue"]
    assert len(mix.stems[0].samples) == len(mix.samples) == 32000  # the bed runs to 4 s
    assert np.allclose(mix.stems[0].samples[:12000], speech_only.samples, atol=1e-6)
    assert np.allclose(mix.stems[0].samples[12000:], 0.0)


def test_stems_add_up_to_the_mix_with_a_ducked_bed(tmp_path):
    """The bed stem ducks under speech that isn't in it: the sidechain is the
    full mix's, so the stems still add up to the mix."""
    doc, _a, _b = _stem_doc(tmp_path)

    mix = render_mix(doc, 8000, stems="track", dialogue_stem=True)

    by_name = {s.name: s.samples for s in mix.stems}
    music = by_name["Music"]
    assert music[4000, 0] < 0.4 * 0.5  # down under the speech
    assert music[31000, 0] == pytest.approx(0.4, rel=0.02)  # and back up once it stops
    assert np.allclose(by_name["Alice track"] + by_name["Bob track"] + music, mix.samples, atol=1e-6)
    assert np.allclose(by_name["Dialogue"] + music, mix.samples, atol=1e-6)


def test_mono_stems_are_mono(tmp_path):
    doc, _a, _b = _stem_doc(tmp_path, bed=False)
    mix = render_mix(doc, 8000, channels=1, stems="track")
    assert mix.samples.ndim == 1 and all(s.samples.ndim == 1 for s in mix.stems)
    assert np.allclose(_sum(mix.stems), mix.samples, atol=1e-6)


def test_an_unknown_stem_mode_is_refused(tmp_path):
    doc, _a, _b = _stem_doc(tmp_path, bed=False)
    with pytest.raises(ValueError):
        render_mix(doc, 8000, stems="instrument")


def test_mixdown_writes_a_file_per_stem_beside_the_mix(tmp_path):
    doc, _a, _b = _stem_doc(tmp_path)
    out = tmp_path / "out" / "story.wav"

    result = mixdown(doc, str(out), fmt="wav", sample_rate=8000, stems="character", dialogue_stem=True)

    names = [os.path.basename(p) for p in result.stem_files]
    assert names == ["story_Alice.wav", "story_Bo_b_ok.wav", "story_Music.wav", "story_Dialogue.wav"]
    lengths = {sf.info(p).frames for p in result.stem_files} | {sf.info(str(out)).frames}
    assert lengths == {32000}
    assert result.audio_path == str(out)
    assert sorted(os.listdir(out.parent)) == sorted(["story.wav", *names])


def test_a_stem_with_the_name_of_another_gets_a_number(tmp_path):
    doc, _a, _b = _stem_doc(tmp_path, bed=False)
    doc.tracks[1].name = "Alice track"
    result = mixdown(doc, str(tmp_path / "s.wav"), fmt="wav", sample_rate=8000, stems="track")
    assert [os.path.basename(p) for p in result.stem_files] == ["s_Alice_track.wav", "s_Alice_track_2.wav"]


def test_stem_names_carry_the_start_timecode_only_when_it_is_on(tmp_path):
    doc, _a, _b = _stem_doc(tmp_path, bed=False)
    off = mixdown(doc, str(tmp_path / "off" / "s.wav"), fmt="wav", sample_rate=8000, stems="character")
    assert [os.path.basename(p) for p in off.stem_files] == ["s_Alice.wav", "s_Bo_b_ok.wav"]

    doc.settings["timecode"] = {"enabled": True, "frame_rate": 25.0, "start": "01:00:00:00"}
    on = mixdown(doc, str(tmp_path / "on" / "s.wav"), fmt="wav", sample_rate=8000, stems="character")
    assert [os.path.basename(p) for p in on.stem_files] == ["s_Alice_01000000.wav", "s_Bo_b_ok_01000000.wav"]
    # The audio still starts at 0 in each file.
    assert np.allclose(sf.read(on.stem_files[0])[0][:100], 0.25, atol=1e-3)

    doc.settings["timecode"] = {"enabled": True, "frame_rate": 25.0, "start": "01:00:00:00"}
    ranged = mixdown(doc, str(tmp_path / "range" / "s.wav"), fmt="wav", sample_rate=8000, stems="character",
                     range_s=(0.4, 1.5))
    assert os.path.basename(ranged.stem_files[0]) == "s_Alice_01000010.wav"  # the range's start, 10 frames in


def test_stems_are_resampled_padded_and_normalized_with_the_mix(tmp_path):
    pytest.importorskip("pyloudnorm")
    doc, _a, _b = _stem_doc(tmp_path, bed=False, rate=8000)

    result = mixdown(doc, str(tmp_path / "n.wav"), fmt="wav", sample_rate=8000, out_rate=16000, head_s=0.5,
                     tail_s=0.25, loudness={"mode": "lufs", "target_lufs": -23.0, "ceiling_dbtp": -1.0},
                     stems="track")

    mix, rate = sf.read(str(tmp_path / "n.wav"), dtype="float32")
    stems = [sf.read(p, dtype="float32")[0] for p in result.stem_files]
    assert rate == 16000 and len(mix) == int(16000 * 2.25)
    assert all(len(s) == len(mix) for s in stems)
    assert np.allclose(sum(stems), mix, atol=1e-3)  # 16 bit files: a step is 3e-5 each
    assert np.allclose(mix[:8000], 0.0)


def test_rms_mode_limits_the_stems_with_the_mixs_gain(tmp_path):
    pytest.importorskip("pyloudnorm")
    doc = _peaky_doc(tmp_path)
    # A second character on a second track, so there are two stems to add up.
    bob = Character.from_preset_dict("Bob", {})
    track = Track(name="B", character_id=bob.id, order_index=1)
    quiet = Clip(character_id=bob.id, track_id=track.id, timeline_timestamp=0.0,
                 segments=[Segment(order_index=0, duration=2.0, audio_path=_wav(tmp_path / "q.wav", 0.02, 2.0))])
    doc.clips.append(quiet)
    doc.characters.append(bob)
    doc.tracks.append(track)
    doc.runs.append(Run(text="Hi", clip_id=quiet.id, kind=quiet.source))

    result = mixdown(doc, str(tmp_path / "r.wav"), fmt="wav", sample_rate=8000, channels=1,
                     loudness={"mode": "rms", "target_rms_dbfs": -20.0, "limiter_dbfs": -3.5}, stems="track")

    mix = sf.read(str(tmp_path / "r.wav"), dtype="float32")[0]
    stems = [sf.read(p, dtype="float32")[0] for p in result.stem_files]
    assert len(stems) == 2
    assert np.max(np.abs(mix)) <= 10 ** (-3.5 / 20) + 1e-3
    assert np.allclose(sum(stems), mix, atol=1e-3)


def test_stems_survive_a_split_export(tmp_path):
    doc, arrangement, nested = _book(tmp_path)
    plan = plan_chapters(doc, arrangement, "subprojects")

    result = mixdown_chapters(doc, str(tmp_path / "out"), plan, fmt="wav", sample_rate=RATE, arrangement=arrangement,
                              nested_audio_path=nested, stems="track", dialogue_stem=True)

    names = sorted(os.path.basename(p) for p in result.stem_files)
    assert names == sorted(f"{chapter.name}_{stem}.wav" for chapter in plan.chapters
                           for stem in ("Unassigned", "Dialogue"))
