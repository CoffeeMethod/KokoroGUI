"""kokoro_gui/daw/dirty.py's `apply_results`: landing one generate's results
on a clip (segments, takes, status)."""
from kokoro_gui.daw.dirty import apply_results
from kokoro_gui.daw.models import Clip, Segment


def _result(path, key="k", take=None, **extra):
    result = {"text": "Hello.", "path": path, "duration": 1.0, "cache_key": key, **extra}
    if take is not None:
        result["take"] = take
    return result


def _segment(path, key="k", **extra):
    return Segment(order_index=0, text="Hello.", cache_key=key, audio_path=path, duration=1.0, **extra)


def test_first_generate_fills_segments_and_status():
    clip = Clip()
    apply_results(clip, [_result("/g/a.wav", key="a", engine_version="1.0")])
    assert [(s.audio_path, s.cache_key, s.engine_version) for s in clip.segments] == [("/g/a.wav", "a", "1.0")]
    assert clip.status == "generated"
    assert clip.takes == {}
    assert "take" not in clip.overrides


def test_a_take_bump_parks_the_old_take():
    old = [_segment("/g/old.wav", key="old")]
    clip = Clip(segments=old)
    apply_results(clip, [_result("/g/new.wav", key="new", take=1)])
    assert clip.overrides["take"] == 1
    assert clip.takes == {0: old}
    assert clip.segments[0].audio_path == "/g/new.wav"


def test_rerendering_a_parked_take_unparks_it():
    parked = [_segment("/g/t1.wav", key="t1")]
    active = [_segment("/g/t0.wav", key="t0")]
    clip = Clip(segments=active, takes={1: parked})
    apply_results(clip, [_result("/g/t1b.wav", key="t1b", take=1)])
    assert clip.overrides["take"] == 1
    assert clip.takes == {0: active}


def test_back_to_take_zero_drops_the_override():
    clip = Clip(segments=[_segment("/g/t1.wav")], overrides={"take": 1})
    apply_results(clip, [_result("/g/t0.wav", take=0)])
    assert "take" not in clip.overrides
    assert 1 in clip.takes


def test_results_without_a_take_keep_the_current_one():
    clip = Clip(segments=[_segment("/g/t2.wav")], overrides={"take": 2})
    apply_results(clip, [_result("/g/t2b.wav")])
    assert clip.overrides["take"] == 2
    assert clip.takes == {}


def test_timing_carries_over_on_a_cache_hit():
    words = [["Hello", 0.0, 0.4]]
    clip = Clip(segments=[_segment("/g/a.wav", key="a", words=words, onset_s=0.1, tail_s=0.2)])
    apply_results(clip, [_result("/g/a.wav", key="a")])
    segment = clip.segments[0]
    assert (segment.words, segment.onset_s, segment.tail_s) == (words, 0.1, 0.2)


def test_timing_carries_from_a_parked_take():
    clip = Clip(segments=[_segment("/g/b.wav", key="b")],
                takes={1: [_segment("/g/a.wav", key="a", onset_s=0.3)]}, overrides={})
    apply_results(clip, [_result("/g/a.wav", key="a", take=1)])
    assert clip.segments[0].onset_s == 0.3


def test_fallback_key_fills_a_result_without_one():
    clip = Clip()
    apply_results(clip, [{"text": "Hi.", "path": "/g/x.wav", "duration": 0.5}], fallback_key="fallback")
    assert clip.segments[0].cache_key == "fallback"


def test_a_recorded_take_at_the_landing_index_is_parked():
    recorded = [Segment(order_index=0, text="Hello.", cache_key=None,
                        audio_path="/p/audio/imported/rec.wav", duration=1.0)]
    clip = Clip(segments=recorded)
    apply_results(clip, [_result("/p/audio/generated/g.wav", key="g", take=0)])
    assert clip.segments[0].audio_path.endswith("g.wav")
    assert recorded in clip.takes.values()
