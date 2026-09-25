"""Tests for kokoro_gui/daw/mixdown.py - offline export of a clip document
(section 6 of Claude/PLAN_ui_shell_redesign.md). Writes real wav files
into tmp_path; no engine, no Qt."""
import numpy as np
import soundfile as sf

from kokoro_gui.daw.arrangement import compute_arrangement
from kokoro_gui.daw.mixdown import mixdown, write_srt
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
