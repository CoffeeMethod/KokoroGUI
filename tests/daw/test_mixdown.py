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
