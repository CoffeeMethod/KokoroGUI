"""Markdown to speakable text (plan section 6, H7 "clean").

Agent text is untrusted, so nothing here backtracks. Every regex either
matches a fixed shape or has a character class that excludes its own
opener, so a failed match at one position can't rescan what the next
position scans (`[^\\[\\]\\n]*` after `\\[`, never `[^\\]]*`). Trailing space and
trailing `#` are trimmed with `rstrip`, not a regex. `tests/headless/test_clean.py`
runs long hostile input against a time bound.

`[Speaker]:` tags and `[pause:x]` markers pass through untouched, so the
result can go straight into `script.build_document`.
"""
from __future__ import annotations

import html
import re
from collections import deque

# A private-use character stands in for each backslash-escaped character
# while the markup is removed.
_ESCAPABLE = "\\`*_{}[]()#+-.!|>~<&"
_ESCAPE_BASE = 0xE000

# What no text we hand on may hold: C0 and C1 controls except newline and tab,
# a lone surrogate (it can't be encoded as UTF-8), U+FFFC (the editor's
# object-replacement character) and a BOM. CR, U+2028 and U+2029 become
# newlines before this runs.
_UNSAFE = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f￼﻿\ud800-\udfff]")
# The stand-ins for escaped characters. The same characters in the input are dropped.
_STANDIN = re.compile(f"[{chr(_ESCAPE_BASE)}-{chr(_ESCAPE_BASE + len(_ESCAPABLE) - 1)}]")
_ENTITY = re.compile(r"&(?:[A-Za-z][A-Za-z0-9]{1,31}|#[0-9]{1,8}|#[xX][0-9A-Fa-f]{1,8});")

_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|>~<&])")
_AUTOLINK = re.compile(r"<((?:https?://|mailto:)[^<>\s]{1,2000})>")
_HTML_TAG = re.compile(r"</?[A-Za-z][^<>\n]{0,2000}>|<![A-Za-z][^<>\n]{0,2000}>")
_IMAGE = re.compile(r"!\[([^\[\]\n]*)\]\([^()\[\]\s]*(?:\s+\"[^\"\n]*\")?\)")
_LINK = re.compile(r"\[([^\[\]\n]*)\]\([^()\[\]\s]*(?:\s+\"[^\"\n]*\")?\)")
_REF_LINK = re.compile(r"\[(?!pause:\d)([^\[\]\n]+)\]\[[^\[\]\n]*\]")
_FOOTNOTE = re.compile(r"\[\^[^\[\]\s]{1,100}\]")
_REF_DEFINITION = re.compile(r"^ {0,3}\[[^\[\]\n]{1,200}\]:[ \t]*(?:https?://|www\.|<https?://)")
_URL = re.compile(r"(?:https?://|www\.)[^\s<>]{1,2000}")
_STAR_RUN = re.compile(r"\*+")
_UNDERSCORE_RUN = re.compile(r"_+")
_STRIKE = re.compile(r"~~")
_BACKTICK_RUN = re.compile(r"`+")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_LIST_MARKER = re.compile(r"^(?:[-*+]|\d{1,9}[.)])[ \t]+")
_TASK_MARKER = re.compile(r"^\[[ xX]\][ \t]+")
_SEPARATOR_CELL = re.compile(r"^:?-{1,200}:?$")
_SPACES = re.compile(r"[ \t]+")
_BLANK_RUN = re.compile(r"\n{3,}")
_TRAILING_URL_PUNCTUATION = ".,;:!?)]}\"'"


def normalise_text(text: str) -> tuple:
    """`(text, dropped)`: `text` with CR, CRLF, U+2028 and U+2029 as `\n` and
    the characters in `_UNSAFE` removed. `dropped` counts the removed ones."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").replace(" ", "\n").replace(" ", "\n")
    return _UNSAFE.subn("", text)


def _entity(match: re.Match) -> str:
    """The character an entity names. A bracket becomes a parenthesis, so
    `&#91;Sam&#93;:` never turns into a speaker tag."""
    char = _STANDIN.sub("", html.unescape(match.group(0)))
    return {"[": "(", "]": ")"}.get(char, char)


def clean_markdown(text: str, *, code_blocks: str = "drop") -> str:
    """`text` as something a voice can read.

    Removes heading marks, emphasis, strikethrough, link and image syntax
    (the link text or the image's alt text stays), inline code marks (the
    code stays), list, task and block quote prefixes, HTML tags and
    comments, table rules (a row's cells are joined with commas), reference
    definitions and footnote marks. A bare URL becomes its host. Fenced
    code is dropped, or kept without its fences with `code_blocks="keep"`.
    Headings, list items and table rows end in a period when they have no
    end punctuation, so the voice pauses there. HTML entities are decoded.
    """
    if code_blocks not in ("drop", "keep"):
        raise ValueError("code_blocks must be 'drop' or 'keep'")
    text, _dropped = normalise_text(text)
    text = _STANDIN.sub("", text)
    text = _strip_comments(text)
    text = _ESCAPE.sub(lambda m: chr(_ESCAPE_BASE + _ESCAPABLE.index(m.group(1))), text)

    out: list = []
    lines = text.split("\n")
    fence = None
    in_table = False
    for index, raw in enumerate(lines):
        match = _FENCE.match(raw)
        if fence is not None:
            if match and match.group(1)[0] == fence[0] and len(match.group(1)) >= len(fence) and not raw.strip(" `~"):
                fence = None
            elif code_blocks == "keep":
                out.append(raw.strip())
            continue
        if match:
            fence = match.group(1)
            out.append("")
            continue
        stripped = raw.strip()
        if _is_separator_row(stripped):
            in_table = True
        elif "|" not in stripped:
            in_table = False
        out.extend(_block_line(raw, lines[index + 1] if index + 1 < len(lines) else "", in_table))
    result = "\n".join(out)
    result = _BLANK_RUN.sub("\n\n", result).strip("\n")
    result = "\n".join(line.rstrip() for line in result.split("\n"))
    result = _UNSAFE.sub("", _ENTITY.sub(_entity, result))
    return "".join(_unescape(c) for c in result)


def _unescape(char: str) -> str:
    code = ord(char) - _ESCAPE_BASE
    return _ESCAPABLE[code] if 0 <= code < len(_ESCAPABLE) else char


def _strip_comments(text: str) -> str:
    """Removes `<!-- ... -->`, which may span lines. An unclosed one stays."""
    if "<!--" not in text:
        return text
    parts = []
    cursor = 0
    while True:
        start = text.find("<!--", cursor)
        if start < 0:
            break
        end = text.find("-->", start + 4)
        if end < 0:
            break
        parts.append(text[cursor:start])
        cursor = end + 3
    parts.append(text[cursor:])
    return "".join(parts)


def _is_rule(stripped: str) -> bool:
    """A thematic break or setext underline: three or more of one of `-*_=`."""
    squeezed = stripped.replace(" ", "").replace("\t", "")
    return len(squeezed) >= 3 and len(set(squeezed)) == 1 and squeezed[0] in "-*_="


def _is_separator_row(stripped: str) -> bool:
    if "|" not in stripped or "-" not in stripped:
        return False
    cells = [c.strip() for c in stripped.strip("|").split("|")]
    return all(_SEPARATOR_CELL.match(c) for c in cells)


def _end_punctuated(line: str) -> str:
    if line and line[-1] not in ".!?:;,)\"'…":
        return line + "."
    return line


def _block_line(raw: str, following: str, in_table: bool) -> list:
    """The output lines for one input line (usually one; a heading adds a
    blank line on each side)."""
    stripped = raw.strip()
    if not stripped or _is_rule(stripped) or _REF_DEFINITION.match(raw):
        return [""]
    if _is_separator_row(stripped):
        return []

    heading = False
    punctuate = False
    # Block quote prefixes, then a heading or a list marker inside them.
    while stripped.startswith(">"):
        stripped = stripped[1:].lstrip(" \t")
    if stripped.startswith("#"):
        level = len(stripped) - len(stripped.lstrip("#"))
        if 1 <= level <= 6 and (len(stripped) == level or stripped[level] in " \t"):
            stripped = stripped[level:].strip()
            closed = stripped.rstrip("#")
            if closed != stripped and (not closed or closed[-1] in " \t"):
                stripped = closed.rstrip()
            heading = True
            punctuate = True
    if not heading:
        marker = _LIST_MARKER.match(stripped)
        if marker:
            stripped = stripped[marker.end():].lstrip(" \t")
            task = _TASK_MARKER.match(stripped)
            if task:
                stripped = stripped[task.end():]
            punctuate = True

    cells = None
    if "|" in stripped and (stripped.startswith("|") or stripped.endswith("|") or in_table
                          or _is_separator_row(following.strip())):
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        stripped = ", ".join(c for c in (_inline(c) for c in cells) if c)
        punctuate = True
    else:
        stripped = _inline(stripped)
    if not stripped:
        return [""]
    if punctuate:
        stripped = _end_punctuated(stripped)
    return ["", stripped, ""] if heading else [stripped]


def _inline(line: str) -> str:
    line = _AUTOLINK.sub(lambda m: m.group(1), line)
    line = _IMAGE.sub(lambda m: m.group(1), line)
    line = _LINK.sub(lambda m: m.group(1), line)
    line = _REF_LINK.sub(lambda m: m.group(1), line)
    line = _FOOTNOTE.sub("", line)
    line = _inline_code(line)
    line = _HTML_TAG.sub("", line)
    line = _URL.sub(_url_host, line)
    line = _STRIKE.sub("", line)
    line = _STAR_RUN.sub(lambda m: _emphasis_run(m, "*"), line)
    line = _UNDERSCORE_RUN.sub(lambda m: _emphasis_run(m, "_"), line)
    return _SPACES.sub(" ", line).strip()


def _emphasis_run(match: re.Match, char: str) -> str:
    """Drops a run of `*` or `_` that marks emphasis and keeps one that
    doesn't: a lone `*` between spaces, or `snake_case`."""
    line = match.string
    before = line[match.start() - 1] if match.start() else " "
    after = line[match.end()] if match.end() < len(line) else " "
    if char == "_" and before.isalnum() and after.isalnum():
        return match.group(0)
    if before.isspace() and after.isspace():
        return match.group(0)
    return ""


def _url_host(match: re.Match) -> str:
    """A bare URL as its host, with the punctuation that ended the sentence
    kept: `See https://a.com/x).` gives `See a.com).`"""
    url = match.group(0)
    core = url.rstrip(_TRAILING_URL_PUNCTUATION)
    tail = url[len(core):]
    host = re.sub(r"^(?:https?://)?(?:www\.)?", "", core, count=1)
    host = re.split(r"[/?#]", host, maxsplit=1)[0]
    return host + tail


def _inline_code(line: str) -> str:
    """Removes backtick marks: a run pairs with the next run of the same
    length, and the text between stays. An unpaired run is dropped."""
    if "`" not in line:
        return line
    runs = [(m.start(), m.end()) for m in _BACKTICK_RUN.finditer(line)]
    by_length: dict = {}
    for position, (start, end) in enumerate(runs):
        by_length.setdefault(end - start, deque()).append(position)
    keep = []  # (start, end) of the stretches kept
    cursor = 0
    position = 0
    while position < len(runs):
        start, end = runs[position]
        queue = by_length[end - start]
        while queue and queue[0] <= position:
            queue.popleft()
        if not queue:
            keep.append((cursor, start))
            cursor = end
            position += 1
            continue
        closing = queue.popleft()
        keep.append((cursor, start))
        keep.append((end, runs[closing][0]))
        cursor = runs[closing][1]
        position = closing + 1
    keep.append((cursor, len(line)))
    return "".join(line[a:b] for a, b in keep)
