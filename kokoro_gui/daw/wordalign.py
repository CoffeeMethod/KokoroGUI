"""Word times for a segment whose engine gave none (Audio8, and Kokoro in
languages other than English), from an ASR pass over the segment's file.

`align(text, asr_words)` matches the ASR words to the segment text's words
with `difflib.SequenceMatcher` over normalised tokens (lower case, letters
and digits only). A matched text word takes the ASR word's times. An
unmatched text word (the ASR dropped it, or heard "twenty" for "20")
interpolates between its matched neighbours; before the first or after the
last match it gets a zero-length span at that neighbour's edge. The result
is `Segment.words`: `[[text_word, start_s, end_s], ...]`, one row per text
word in order.

Qt-free and model-free: the caller supplies the ASR words
(`kokoro_gui.engine.asr.transcribe_wav_words`).
"""
from __future__ import annotations

import difflib
import re

_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)


def _norm(word: str) -> str:
    return _NON_WORD.sub("", word.lower())


def align(text: str, asr_words: list) -> list:
    text_words = text.split()
    if not text_words:
        return []
    asr = []
    for item in asr_words or []:
        try:
            asr.append((str(item[0]), float(item[1]), float(item[2])))
        except (TypeError, ValueError, IndexError):
            continue
    if not asr:
        return []

    times: list = [None] * len(text_words)
    matcher = difflib.SequenceMatcher(a=[_norm(w) for w in text_words], b=[_norm(w[0]) for w in asr],
                                      autojunk=False)
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            _word, start, end = asr[block.b + k]
            times[block.a + k] = (start, end)

    matched = [i for i, t in enumerate(times) if t is not None]
    if not matched:
        # Nothing lines up: spread the text over the ASR's overall span.
        start, end = asr[0][1], asr[-1][2]
        step = (end - start) / len(text_words)
        return [[w, round(start + i * step, 4), round(start + (i + 1) * step, 4)] for i, w in enumerate(text_words)]

    out = []
    for i, word in enumerate(text_words):
        if times[i] is not None:
            start, end = times[i]
        else:
            before = max((j for j in matched if j < i), default=None)
            after = min((j for j in matched if j > i), default=None)
            if before is None:
                start = end = times[after][0]
            elif after is None:
                start = end = times[before][1]
            else:
                # Share the silence between the two matches evenly among
                # the unmatched words that sit in it.
                gap_start, gap_end = times[before][1], times[after][0]
                count = after - before - 1
                step = max(0.0, gap_end - gap_start) / count
                start = gap_start + (i - before - 1) * step
                end = start + step
        out.append([word, round(start, 4), round(end, 4)])
    return out
