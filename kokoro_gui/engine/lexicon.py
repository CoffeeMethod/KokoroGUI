"""Lexicon (find/replace) substitution, applied to text before synthesis.

`apply_lexicon` is the one implementation. Every generation path calls it
(whole document, JIT, preview, and each clip through
`ConversionMixin.generate_clip_audio`), and the dirty check
(kokoro_gui/daw/dirty.py) calls it before hashing, so a segment key is over
the text the engine spoke and a lexicon edit stales exactly the clips whose
text it rewrites. A clip's text goes through `spoken`, which strips its
inline tags and pause markers first.

A lexicon is a list of rules applied in order, `{"find", "replace", "mode",
"case"}` (see `normalize_rules`). The old shape, a `{find: replace}` dict, is
still accepted everywhere and means what its migrated list means."""
import json
import re

from kokoro_gui.engine.text_extraction import strip_markup

# How a rule's `find` matches: as text anywhere ("literal", what every rule
# did before modes), as a whole word ("word"), or as a regular expression.
MODES = ("literal", "word", "regex")


def _rule_tuples(obj):
    """`(find, replace, mode, case)` for each usable rule in `obj`, in order:
    a legacy `{find: replace}` dict (literal, case-insensitive: exactly what
    the dict did) or a list of rule dicts. An entry without a non-empty
    string `find`, or with a non-string `replace`, is dropped (the old code
    skipped those too). A missing or unknown `mode` is "literal", a missing
    `case` False. Anything else gives nothing."""
    if isinstance(obj, dict):
        for find, replace in obj.items():
            if isinstance(find, str) and find and isinstance(replace, str):
                yield find, replace, "literal", False
    elif isinstance(obj, (list, tuple)):
        for entry in obj:
            if not isinstance(entry, dict):
                continue
            find, replace = entry.get("find"), entry.get("replace", "")
            if not isinstance(find, str) or not find or not isinstance(replace, str):
                continue
            mode = entry.get("mode")
            yield find, replace, mode if mode in MODES else "literal", entry.get("case") is True


def normalize_rules(obj) -> list:
    """`obj` as a list of fresh rule dicts `{"find", "replace", "mode",
    "case"}`. `case` True means match case; False (the default, and what
    every legacy rule is) ignores it. A legacy dict becomes literal rules in
    its order, so the spoken text is unchanged by the migration."""
    return [{"find": find, "replace": replace, "mode": mode, "case": case}
            for find, replace, mode, case in _rule_tuples(obj)]


def _pattern_source(find: str, mode: str) -> str:
    """The regex source for `find` in `mode`. A whole word is neither
    preceded nor followed by a word character; the check applies only on a
    side where `find` itself has a word character, so "Dr." still matches
    in "Dr. Who"."""
    if mode == "regex":
        return find
    source = re.escape(find)
    if mode == "word":
        if re.match(r"\w", find[0]):
            source = r"(?<!\w)" + source
        if re.match(r"\w", find[-1]):
            source += r"(?!\w)"
    return source


def compile_rule(find: str, mode: str = "literal", case: bool = False, cache=None):
    """The compiled pattern of a rule; raises `re.error` for a regex that
    doesn't compile. `cache`, a dict that outlives the call, remembers the
    pattern, and a failure as None so a bad rule is printed once."""
    key = (mode, bool(case), find)
    if cache is not None and key in cache:
        pattern = cache[key]
        if pattern is None:
            raise re.error("the pattern did not compile")
        return pattern
    try:
        pattern = re.compile(_pattern_source(find, mode), 0 if case else re.IGNORECASE)
    except re.error:
        if cache is not None:
            cache[key] = None
        raise
    if cache is not None:
        cache[key] = pattern
    return pattern


try:
    from re import _parser as _template_parser  # Python 3.11+
except ImportError:  # pragma: no cover - 3.10
    import sre_parse as _template_parser


def check_rule(find: str, replace: str, mode: str = "literal", case: bool = False):
    """Why a rule would be skipped, as a message, or None when it is fine: a
    regex that doesn't compile, or a replacement that names a group the
    pattern lacks (`\\2`, `\\g<name>`)."""
    try:
        pattern = compile_rule(find, mode, case)
    except re.error as e:
        return str(e)
    try:
        _template_parser.parse_template(replace, pattern)
    except (re.error, IndexError) as e:  # IndexError: an unknown group name
        return str(e)
    except Exception:  # noqa: BLE001 - the parser is private: no answer is no error
        return None
    return None


def apply_lexicon(text, lexicon, cache=None, with_spans=False, origin=None):
    """Applies the rules of `lexicon` (a list of rules, or the old
    `{find: replace}` dict) to `text`, in order. The replacement goes to
    `re.sub` as given, so `\\1` and `\\g<0>` work and a literal backslash is
    `\\\\`. `cache` maps a rule to its compiled pattern; pass a dict that
    outlives the call to skip recompiling. A rule that fails is printed and
    skipped.

    `with_spans=True` returns `(text, spans)` instead, `spans` a list of
    `(orig_start, orig_end, new_start, new_end)` covering the result in
    order: an unchanged stretch maps character for character, a
    replacement maps as a whole to the original text it replaced. The
    transcript maps a spoken word's offset back through it
    (`original_offset`). `origin` (implies `with_spans`) is where each
    character of `text` came from in some earlier text, as
    `text_extraction.strip_markup(..., with_origin=True)` gives it; the
    spans are then into that earlier text."""
    if origin is not None:
        with_spans = True
    if not lexicon:
        if origin is not None:
            return text, _collapse_spans(origin)
        return (text, [(0, len(text), 0, len(text))] if text else []) if with_spans else text
    if cache is None:
        cache = {}
    # Per character of the current text: the original span it came from,
    # and which replacement produced it (None for untouched text).
    if origin is not None:
        origin = list(origin)
    elif with_spans:
        origin = [(i, i + 1, None) for i in range(len(text))]
    replacements = 0

    for src, dest, mode, case in _rule_tuples(lexicon):
        try:
            pattern = compile_rule(src, mode, case, cache)
            if not with_spans:
                text = pattern.sub(dest, text)
                continue
            parts, new_origin, last = [], [], 0
            for match in pattern.finditer(text):
                parts.append(text[last:match.start()])
                new_origin.extend(origin[last:match.start()])
                replacement = match.expand(dest)
                replacements += 1
                if match.start() < match.end():
                    span = (origin[match.start()][0], origin[match.end() - 1][1], replacements)
                else:
                    # An empty match (a pattern like `\b` or `x*`) replaces
                    # no text: what it inserts sits at one point of the original.
                    at = (origin[match.start()][0] if match.start() < len(origin)
                          else origin[-1][1] if origin else 0)
                    span = (at, at, replacements)
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


def spoken(text, lexicon, cache=None, with_spans=False):
    """What a clip speaks: `text` without its markup
    (`text_extraction.strip_markup`), then the lexicon. With `with_spans`
    the spans map back to `text` itself, markup included."""
    if not with_spans:
        return apply_lexicon(strip_markup(text), lexicon, cache)
    stripped, origin = strip_markup(text, with_origin=True)
    return apply_lexicon(stripped, lexicon, cache, origin=origin)


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
    """A stable string for `lexicon`, for memo keys: sorted JSON of its
    normalized rules, so a legacy dict and its migrated list share one."""
    return json.dumps(normalize_rules(lexicon), sort_keys=True, ensure_ascii=False)


class LexiconMixin:
    def apply_lexicon(self, text, lexicon):
        """`apply_lexicon` with this engine's compiled-pattern cache."""
        return apply_lexicon(text, lexicon, self._lexicon_cache)
