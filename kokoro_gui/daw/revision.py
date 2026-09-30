"""Change counters for the document model (Claude/PLAN_performance.md).

The GUI caches what it derives from a `Document` (the run index, the dirty
set, the arrangement) and needs to know when that went out of date. Three
module-level counters say so:

- `TEXT` moves when anything the run index reads changes: a `Run`'s fields,
  `Document.runs`/`clips`/`characters` being reassigned, or an object's `id`.
- `MODEL` moves on any other attribute set on a `Document`, `Clip`,
  `Segment`, `Character` or `Track`, on every undo stack push, undo and
  redo, and on `Document.touch()`.
- `FILES` moves when files the dirty check reads may have changed on disk:
  after Generate, Open, Save and import, when the window is re-activated, on
  a voice save, and on Options > Force refresh.

Setting an attribute moves a counter by itself (`Tracked.__setattr__`), so a
code path that forgets to invalidate can't leave the screen stale. An
in-place change to a list or dict field doesn't: those go through an undo
command, or the caller calls `Document.touch()`. The caches also key on the
identity and length of the lists they walk, and on the content of each
clip's fields, so an untracked in-place edit is picked up by the next move of
any counter.

The counters are global rather than per document: an edit in a subproject's
document costs every cache one cheap revalidation and nothing else, and no
`Run` or `Segment` needs a back-reference to its owner (which `copy.deepcopy`
in the undo commands would have to preserve).
"""
from __future__ import annotations

import os

TEXT = 0
MODEL = 0
FILES = 0

# Positive `os.path.isfile` answers and mtimes, valid until `FILES` moves.
_file_cache: dict = {}
_file_cache_epoch = -1


def bump_text() -> None:
    global TEXT
    TEXT += 1


def bump_model() -> None:
    global MODEL
    MODEL += 1


def bump_files() -> None:
    global FILES
    FILES += 1


def _cache() -> dict:
    global _file_cache, _file_cache_epoch
    if _file_cache_epoch != FILES:
        _file_cache = {}
        _file_cache_epoch = FILES
    return _file_cache


def file_exists(path: str) -> bool:
    """`os.path.isfile(path)`, remembered until `FILES` moves when True. A
    missing file is asked again every time, so a file that appears (a
    Generate writing it) is seen at once; a file deleted by hand is seen at
    the next `bump_files()`."""
    cache = _cache()
    if cache.get(("isfile", path)):
        return True
    exists = os.path.isfile(path)
    if exists:
        cache[("isfile", path)] = True
    return exists


def file_mtime(path: str):
    """`os.path.getmtime(path)`, remembered until `FILES` moves; None when
    it can't be read (not remembered)."""
    cache = _cache()
    key = ("mtime", path)
    if key in cache:
        return cache[key]
    try:
        stamp = os.path.getmtime(path)
    except OSError:
        return None
    cache[key] = stamp
    return stamp


class Tracked:
    """Base for the model dataclasses: every attribute set moves `MODEL`,
    or `TEXT` for a name in `_TEXT_FIELDS` (and always for `id`)."""

    _TEXT_FIELDS: frozenset = frozenset()

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if name == "id" or name in self._TEXT_FIELDS:
            bump_text()
        else:
            bump_model()


VERIFY = bool(os.environ.get("KOKOROGUI_VERIFY_DERIVED"))


def set_verify(enabled: bool) -> None:
    """Verify mode: every cached answer in `kokoro_gui/daw/derived.py` is
    also computed from scratch and a mismatch raises. The test suite turns
    it on (both conftests); `KOKOROGUI_VERIFY_DERIVED=1` does for a run of
    the app."""
    global VERIFY
    VERIFY = bool(enabled)
