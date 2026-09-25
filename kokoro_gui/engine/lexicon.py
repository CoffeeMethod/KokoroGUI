"""Lexicon (find/replace) substitution, applied to text before synthesis.

`apply_lexicon` is the one implementation. Every generation path calls it
(whole document, JIT, preview, and each clip through
`ConversionMixin.generate_clip_audio`), and the dirty check
(kokoro_gui/daw/dirty.py) calls it before hashing, so a segment key is over
the text the engine spoke and a lexicon edit stales exactly the clips whose
text it rewrites."""
import json
import re


def apply_lexicon(text, lexicon, cache=None, with_spans=False):
    """Applies a dict of replacements to `text`: case-insensitive literal
    find, the replacement passed to `re.sub` as given. `cache` maps a source string to
    its compiled pattern; pass a dict that outlives the call to skip
    recompiling.

    `with_spans=True` returns `(text, spans)` instead, `spans` a list of
    `(orig_start, orig_end, new_start, new_end)` covering the result in
    order: an unchanged stretch maps character for character, a
    replacement maps as a whole to the original text it replaced. The
    transcript maps a spoken word's offset back through it
    (`original_offset`)."""
    if not lexicon:
        return (text, [(0, len(text), 0, len(text))] if text else []) if with_spans else text
    if cache is None:
        cache = {}
    # Per character of the current text: the original span it came from,
    # and which replacement produced it (None for untouched text).
    origin = [(i, i + 1, None) for i in range(len(text))] if with_spans else None
    replacements = 0

    for src, dest in lexicon.items():
        if not src:
            continue
        try:
            pattern = cache.get(src)
            if pattern is None:
                # Escape the search term to treat it as literal text
                pattern = cache[src] = re.compile(re.escape(src), re.IGNORECASE)
            if not with_spans:
                text = pattern.sub(dest, text)
                continue
            parts, new_origin, last = [], [], 0
            for match in pattern.finditer(text):
                parts.append(text[last:match.start()])
                new_origin.extend(origin[last:match.start()])
                replacement = match.expand(dest)
                replacements += 1
                span = (origin[match.start()][0], origin[match.end() - 1][1], replacements)
                parts.append(replacement)
                new_origin.extend([span] * len(replacement))
                last = match.end()
            parts.append(text[last:])
            new_origin.extend(origin[last:])
            text, origin = "".join(parts), new_origin
        except Exception as e:
            print(f"Lexicon error for '{src}': {e}")

    if with_spans:
        return text, _collapse_spans(origin)
    return text


def _collapse_spans(origin) -> list:
    """Per-character origins grouped into spans: a run of untouched
    characters with consecutive originals, or one replacement's output."""
    spans = []
    previous = None
    for new_index, (o_start, o_end, rid) in enumerate(origin):
        if previous is not None:
            _p_start, p_end, p_rid = previous
            s_start, s_end, n_start, _n_end = spans[-1]
            same_replacement = rid is not None and rid == p_rid
            continues_text = rid is None and p_rid is None and o_start == p_end
            if same_replacement or continues_text:
                spans[-1] = (s_start, max(s_end, o_end), n_start, new_index + 1)
                previous = (o_start, o_end, rid)
                continue
        spans.append((o_start, o_end, new_index, new_index + 1))
        previous = (o_start, o_end, rid)
    return spans


def original_offset(spans, new_offset: int) -> int:
    """The offset in the original text that `new_offset` in the rewritten
    text came from: exact inside an unchanged stretch, the replaced text's
    start inside a replacement."""
    for o_start, o_end, n_start, n_end in spans:
        if n_start <= new_offset < n_end:
            # Equal lengths: untouched text, or a same-length replacement,
            # which maps character for character just as well.
            if o_end - o_start == n_end - n_start:
                return o_start + (new_offset - n_start)
            return o_start
    if spans:
        return spans[-1][1]
    return new_offset


def original_span(spans, new_start: int, new_end: int) -> tuple:
    """`(start, end)` in the original text for `[new_start, new_end)` in the
    rewritten text. A range touching a replacement widens to the whole text
    that replacement came from."""
    start = original_offset(spans, new_start)
    last = max(new_start, new_end - 1)
    for o_start, o_end, n_start, n_end in spans:
        if n_start <= last < n_end:
            if o_end - o_start == n_end - n_start:
                return start, o_start + (last - n_start) + 1
            return start, o_end
    return start, max(start + 1, original_offset(spans, new_end))


def lexicon_signature(lexicon) -> str:
    """A stable string for `lexicon`, for memo keys: sorted JSON."""
    return json.dumps(lexicon or {}, sort_keys=True, ensure_ascii=False)


class LexiconMixin:
    def apply_lexicon(self, text, lexicon):
        """`apply_lexicon` with this engine's compiled-pattern cache."""
        return apply_lexicon(text, lexicon, self._lexicon_cache)
