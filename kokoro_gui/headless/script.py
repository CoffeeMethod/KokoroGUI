"""A script becomes a `Document` (plan section 4, "Scripts").

The script is plain text, optionally tagged `[Speaker]:` or `[Speaker:FX]:`,
with `[pause:x]` markers. `build_document` removes both kinds of markup,
makes one clip per paragraph per speaker block, and gives each speaker a
character whose settings sit in `preset_data`. It loads no model and reads no
config: the caller passes the backend and the cast.

Tags come from `find_character_fx_spans` and pauses from
`auto_split.find_pause_markers`, the same functions Auto-split uses, so a
script means what the GUI says it means. Two pauses in a row keep the last
one, as `auto_split._carve_pauses` does.
"""
from __future__ import annotations

import math
import re
from typing import Collection, Mapping, Optional

from kokoro_gui.core import engines as core_engines
from kokoro_gui.daw.auto_split import find_pause_markers
from kokoro_gui.daw.share import _PRESET_RANGES, _PRESET_STRING_CHARS
from kokoro_gui.daw.models import DEFAULT_HIGHLIGHT_PALETTE, Character, Document
from kokoro_gui.engine.text_extraction import find_character_fx_spans, tag_overrides
from kokoro_gui.headless.clean import normalise_text

DEFAULT_SPEAKER = "Narrator"
# Keys of a cast entry that go into the character's `preset_data`.
CAST_KEYS = ("voice", "speed", "pitch", "fx_preset")

_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")


def _short(value) -> str:
    return repr(value)[:40]


def _bad(name: str, key: str, value, why: str) -> ValueError:
    return ValueError(f"cast[{name[:40]!r}]: {key} {why} (got {_short(value)})")


def _check_text(name: str, key: str, value) -> str:
    if not isinstance(value, str) or not value:
        raise _bad(name, key, value, "must be a non-empty string")
    if len(value) > _PRESET_STRING_CHARS:
        raise _bad(name, key, value, f"must be at most {_PRESET_STRING_CHARS} characters")
    if normalise_text(value)[0] != value or "\n" in value or "\t" in value:
        raise _bad(name, key, value, "must not hold control characters")
    return value


def _check_number(name: str, key: str, value):
    low, high = _PRESET_RANGES[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _bad(name, key, value, "must be a number")
    try:
        number = float(value)
    except OverflowError:
        number = math.inf
    if not math.isfinite(number):
        raise _bad(name, key, value, "must be a finite number")
    if not low <= number <= high:
        raise _bad(name, key, value, f"must be between {low:g} and {high:g}")
    return value


def _validated_cast(cast) -> dict:
    """`{lowercase name: preset_data dict}` for `cast`, or ValueError naming
    the speaker and key. Names match like `get_character_by_name`: ignoring
    case and surrounding whitespace. It does not check that a voice exists
    (the caller has the engine's voice list)."""
    if cast is None:
        return {}
    if not isinstance(cast, Mapping):
        raise ValueError(f"cast must be a mapping of speaker to voice or settings, got {type(cast).__name__}")
    out: dict = {}
    for name, entry in cast.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"cast: a speaker name must be a non-empty string (got {_short(name)})")
        if isinstance(entry, str):
            preset = {"voice": _check_text(name, "voice", entry)}
        elif isinstance(entry, Mapping):
            preset = {}
            for key in CAST_KEYS:
                value = entry.get(key)
                if value is None:
                    continue
                preset[key] = _check_number(name, key, value) if key in _PRESET_RANGES else _check_text(name, key, value)
        else:
            raise ValueError(f"cast[{name[:40]!r}]: must be a voice name or a mapping, got {type(entry).__name__}")
        out.setdefault(name.strip().lower(), preset)
    return out


def _paragraphs(body: str) -> list:
    """The non-empty paragraphs of `body`, split on blank lines, each
    stripped."""
    return [p.strip() for p in _PARAGRAPH_BREAK.split(body) if p.strip()]


def _pieces(body: str):
    """Yields `("text", str)` and `("pause", seconds)` for `body`, in order."""
    cursor = 0
    for start, end, seconds in find_pause_markers(body):
        yield "text", body[cursor:start]
        yield "pause", seconds
        cursor = end
    yield "text", body[cursor:]


def _blocks(script: str) -> list:
    """`[(speaker or None, fx or None, options, body)]` in text order. Text
    before the first tag has no speaker."""
    spans = find_character_fx_spans(script)
    if not spans:
        return [(None, None, {}, script)]
    blocks = []
    if spans[0].start:
        blocks.append((None, None, {}, script[:spans[0].start]))
    for span in spans:
        # The tag's bracket holds no "]", so its end is the first "]:" after its start.
        tag_end = script.index("]:", span.start) + 2
        blocks.append((span.speaker_name.strip() or None, span.fx_name, span.options, script[tag_end:span.end]))
    return blocks


def build_document(script: str, backend, cast: Optional[Mapping] = None, *,
                   engine_settings: Optional[dict] = None,
                   known_fx: Optional[Collection[str]] = None) -> tuple:
    """`(document, warnings)` for `script`.

    - `backend`: the engine adapter. Its id goes on every character, and
      `core.engines.default_voice(backend, engine_settings)` gives a speaker
      the cast doesn't name its voice.
    - `cast`: `{speaker: voice}` or `{speaker: {voice, speed, pitch,
      fx_preset}}`, names matched ignoring case. Untagged text goes to
      "Narrator", or to `cast["Narrator"]` when the cast has it.
    - `known_fx`: the FX preset names that exist. When given, a tag's FX
      name outside it adds a warning and sets nothing. Without it every tag
      FX name is set as written.

    Tags and `[pause:x]` markers are not in `document.text`. Each paragraph
    of a speaker block is one clip, joined by blank lines. A pause is the
    next clip's `gap_before_s`; a trailing pause has no clip and adds a
    warning. `Document.settings` has `gap_s` and `paragraph_gap_s` at 0, and
    the document gets one track per character.
    """
    cast = _validated_cast(cast)
    engine_settings = dict(engine_settings or {})
    warnings: list = []
    characters: dict = {}  # lowercase name -> Character, in first-use order

    def character_for(name: str) -> Character:
        key = name.lower()
        found = characters.get(key)
        if found is not None:
            return found
        entry = cast.get(name.strip().lower())
        preset = dict(entry or {})
        if "voice" not in preset:
            voice = core_engines.default_voice(backend, engine_settings)
            if voice:
                preset["voice"] = voice
            if entry is None and name != DEFAULT_SPEAKER:
                warnings.append(f"Speaker {name!r} is not in the cast; using the default voice.")
        color = DEFAULT_HIGHLIGHT_PALETTE[len(characters) % len(DEFAULT_HIGHLIGHT_PALETTE)]
        found = Character.from_preset_dict(name, preset, highlight_color=color, backend_id=backend.id)
        characters[key] = found
        return found

    script, dropped = normalise_text(script)
    if dropped:
        warnings.append(f"Dropped {dropped} unsupported character(s) from the script "
                        "(control characters, lone surrogates, U+FFFC, BOM).")

    # One entry per clip: (text, character, overrides, gap before).
    plan: list = []
    pending = None
    for speaker, fx, options, body in _blocks(script):
        name = speaker or DEFAULT_SPEAKER
        if fx and known_fx is not None and fx not in known_fx:
            warnings.append(f"Speaker {name!r}: FX preset {fx!r} does not exist; ignored.")
            fx = None
        tag_set, problems = tag_overrides(options)
        warnings.extend(p for p in problems if p not in warnings)
        first_of_block = True
        for kind, value in _pieces(body):
            if kind == "pause":
                if math.isfinite(value):
                    pending = value
                else:
                    warnings.append("A [pause:x] too large to use was ignored.")
                continue
            for paragraph in _paragraphs(value):
                overrides = {"fx_preset": fx} if fx else {}
                if first_of_block:
                    overrides.update(tag_set)
                    first_of_block = False
                plan.append((paragraph, character_for(name), overrides, pending))
                pending = None
    if pending is not None:
        warnings.append("A [pause:x] at the end of the script has no clip after it; ignored.")

    document = Document(characters=list(characters.values()),
                        settings={"gap_s": 0.0, "paragraph_gap_s": 0.0})
    document.set_plain_text("\n\n".join(item[0] for item in plan))
    cursor = 0
    for text, character, overrides, gap in plan:
        clip = document.assign_character_to_range(cursor, cursor + len(text), character.id)
        if overrides:
            clip.overrides.update(overrides)
        if gap is not None:
            clip.gap_before_s = gap
        cursor += len(text) + 2
    return document, warnings
