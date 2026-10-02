"""Split and join a clip at a point (plan 10): `split_join.offset_at` maps a
playhead time to a text offset, `Document.split_clip` / `join_clips` do the
cut and the merge, and `SplitClipCommand` / `JoinClipsCommand` undo them."""
import dataclasses
import re

from kokoro_gui.daw import split_join
from kokoro_gui.daw.arrangement import PlacedClip
from kokoro_gui.daw.models import Clip, Segment

TEXT = "alpha beta gamma delta"
START = 10  # the clip's document offset


def _spans(text):
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def _placed(text=TEXT, timed=True, start_s=5.0, per_word=1.0, gaps=0.0):
    """A clip placed at `start_s`; each word lasts `per_word` seconds, a
    silence of `gaps` between words. Returns (placed, word_offsets)."""
    spans = _spans(text)
    words, t = [], 0.0
    for i, (a, b) in enumerate(spans):
        words.append([text[a:b], t, t + per_word])
        t += per_word + gaps
    segment = Segment(order_index=0, text=text, audio_path="x.wav", duration=t, words=words if timed else [])
    clip = Clip(segments=[segment])
    placed = PlacedClip(clip=clip, start_s=start_s, duration_s=t, estimated=not timed)

    def word_offsets(seg, index):
        a, b = spans[index]
        return START + a, START + b

    return placed, word_offsets


def test_playhead_mid_word_cuts_at_that_words_start():
    placed, offsets = _placed()
    # "gamma" is the third word, 2.0 to 3.0 s into the clip.
    assert split_join.offset_at(placed, 5.0 + 2.5, TEXT, START, offsets) == START + TEXT.index("gamma")


def test_playhead_in_a_gap_cuts_at_the_next_word():
    placed, offsets = _placed(per_word=1.0, gaps=0.5)
    # "beta" spans 1.5 to 2.5 s, "gamma" 3.0 to 4.0 s; 2.75 s is between them.
    assert split_join.offset_at(placed, 5.0 + 2.75, TEXT, START, offsets) == START + TEXT.index("gamma")


def test_playhead_on_the_first_word_cuts_before_the_second():
    placed, offsets = _placed()
    assert split_join.offset_at(placed, 5.0 + 0.2, TEXT, START, offsets) == START + TEXT.index("beta")


def test_playhead_before_the_audio_starts_cuts_before_the_second_word():
    placed, offsets = _placed()
    assert split_join.offset_at(placed, 4.0, TEXT, START, offsets) == START + TEXT.index("beta")


def test_playhead_after_the_last_word_has_no_cut():
    placed, offsets = _placed(per_word=1.0, gaps=0.5)
    assert split_join.offset_at(placed, 5.0 + 99.0, TEXT, START, offsets) is None


def test_leading_whitespace_does_not_make_the_first_word_a_cut():
    text = "  alpha beta"
    placed, offsets = _placed(text)
    assert split_join.offset_at(placed, 5.0 + 0.2, text, START, offsets) == START + text.index("beta")


def test_untimed_clip_cuts_proportionally_and_snaps_to_a_word_start():
    placed, offsets = _placed(timed=False)
    # 55 percent of 22 characters is offset 12, the second letter of
    # "gamma" (11 to 16); the next word start is "delta" at 17.
    placed = dataclasses.replace(placed, duration_s=10.0)
    assert split_join.offset_at(placed, 5.0 + 5.5, TEXT, START, offsets) == START + TEXT.index("delta")


def test_untimed_cut_on_a_word_start_stays_there():
    placed, offsets = _placed(timed=False)
    placed = dataclasses.replace(placed, duration_s=22.0)
    assert split_join.offset_at(placed, 5.0 + 11.0, TEXT, START, offsets) == START + TEXT.index("gamma")


def test_untimed_cut_near_the_start_still_leaves_a_first_half():
    placed, offsets = _placed(timed=False)
    placed = dataclasses.replace(placed, duration_s=10.0)
    assert split_join.offset_at(placed, 5.0, TEXT, START, offsets) == START + TEXT.index("beta")


def test_untimed_cut_inside_the_last_word_has_no_cut():
    placed, offsets = _placed(timed=False)
    placed = dataclasses.replace(placed, duration_s=10.0)
    assert split_join.offset_at(placed, 5.0 + 9.9, TEXT, START, offsets) is None


def test_one_word_clip_has_no_cut():
    placed, offsets = _placed("alpha")
    assert split_join.offset_at(placed, 5.0 + 0.5, "alpha", START, offsets) is None
    untimed, offsets = _placed("alpha", timed=False)
    assert split_join.offset_at(untimed, 5.0 + 0.5, "alpha", START, offsets) is None


def test_words_the_text_cannot_place_fall_back_to_the_proportional_cut():
    placed, _ = _placed()
    placed = dataclasses.replace(placed, duration_s=10.0)
    assert split_join.offset_at(placed, 5.0 + 5.5, TEXT, START, lambda seg, i: None) \
        == START + TEXT.index("delta")


def test_a_word_offset_inside_a_word_snaps_to_the_next_word_start():
    placed, _ = _placed()
    # A lexicon rewrite can map a spoken word to a span that starts mid-word.
    assert split_join.offset_at(placed, 5.0 + 0.2, TEXT, START, lambda seg, i: (START + 2, START + 4)) \
        == START + TEXT.index("beta")
