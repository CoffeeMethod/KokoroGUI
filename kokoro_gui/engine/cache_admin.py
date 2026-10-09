"""Size, clear and trim for the two derived caches. No Qt.

`runtime.CACHE_DIR` holds more than the segment cache: `projects/` is every
open and recent project's working copy (unsaved work lives there) and
`logs/` is the log. So nothing here recurses to delete, ever. A delete
touches only a top-level regular file whose name matches the cache's own
pattern, and only after a realpath check that it sits directly in the
cache dir (a symlink is skipped, never followed). `projects_usage` measures
`projects/` for display and has no clear counterpart.

- The segment cache is `<64 hex>_<n>.<ext>` files in `CACHE_DIR`, written by
  the whole-document and JIT paths (`caching.py`, legacy naming). A hit
  touches the files it reuses, so mtime means "last used" and `trim`
  evicts the least recently used key first, all of a key's files together.
- Audio8's reference codes are `<16 hex>.npy` files under its references
  folder (`audio8_tts._ref_codes_cache_dir()`, not under `CACHE_DIR`).
  Deleting one costs a re-encode.

Every function takes the folder to work in and reads the live one when it
is left out, so tests point it at a tmp dir. The GUI resolves the folder on
its own thread before it starts a worker, so a worker never reads a
changed working directory.
"""
from __future__ import annotations

import os
import re
from typing import Callable, NamedTuple

from kokoro_gui.engine import runtime

SEGMENT_FILE = re.compile(r"^([0-9a-f]{64})_\d+\.(wav|flac|mp3|ogg)$")
REF_CODES_FILE = re.compile(r"^[0-9a-f]{16}\.npy$")

MIB = 1024 * 1024
DEFAULT_MAX_MB = 2048  # `settings["segment_cache_max_mb"]`; 0 means no limit


class Usage(NamedTuple):
    count: int
    bytes: int


NOTHING = Usage(0, 0)


def format_size(n: int) -> str:
    """"0 B", "812 KB", "3.4 MB", "1.20 GB"."""
    size = max(0, int(n))
    if size < 1024:
        return f"{size} B"
    value = float(size)
    for unit, digits in (("KB", 0), ("MB", 1), ("GB", 2)):
        value /= 1024
        if value < 1024 or unit == "GB":
            return f"{value:.{digits}f} {unit}"


def describe(usage: Usage) -> str:
    """"12 files, 3.4 MB"."""
    noun = "file" if usage.count == 1 else "files"
    return f"{usage.count} {noun}, {format_size(usage.bytes)}"


# -- Where things live ----------------------------------------------------------


def segment_cache_dir() -> str:
    return os.path.abspath(runtime.CACHE_DIR)


def projects_dir() -> str:
    """`cache/projects/`, the same folder `project.projects_root()` names."""
    return os.path.join(segment_cache_dir(), "projects")


def ref_codes_dir() -> str:
    from kokoro_gui.engines import audio8_tts

    return os.path.abspath(audio8_tts._ref_codes_cache_dir())


# -- The one place that decides what may be deleted -----------------------------


class _Entry(NamedTuple):
    name: str
    path: str
    size: int
    mtime: float


def _matching(folder: str, pattern: re.Pattern) -> list[_Entry]:
    """The regular files directly in `folder` whose name matches `pattern`.
    Never recurses, and a symlink (or anything else that isn't a plain
    file) is left out. A missing or unreadable folder gives an empty list."""
    found = []
    try:
        with os.scandir(folder) as it:
            for entry in it:
                if not pattern.match(entry.name):
                    continue
                try:
                    if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                        continue
                    st = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                found.append(_Entry(entry.name, entry.path, st.st_size, st.st_mtime))
    except OSError:
        return []
    return found


def _remove(folder: str, entries: list[_Entry], pattern: re.Pattern,
            should_stop: Callable[[], bool] | None = None) -> Usage:
    """Deletes `entries`, each only if it still matches `pattern`, is not a
    link, and resolves to a path directly inside `folder`. A file that is in
    use (Windows) or already gone is skipped. Returns what was removed."""
    real_folder = os.path.normcase(os.path.realpath(folder))
    count = freed = 0
    for entry in entries:
        if should_stop is not None and should_stop():
            break
        if not pattern.match(entry.name) or os.path.islink(entry.path):
            continue
        real_parent = os.path.normcase(os.path.dirname(os.path.realpath(entry.path)))
        if real_parent != real_folder:
            continue
        try:
            os.remove(entry.path)
        except OSError:
            continue
        count += 1
        freed += entry.size
    return Usage(count, freed)


def _sum(entries: list[_Entry]) -> Usage:
    return Usage(len(entries), sum(e.size for e in entries))


# -- Segment cache --------------------------------------------------------------


def segment_cache_usage(folder: str | None = None) -> Usage:
    return _sum(_matching(folder or segment_cache_dir(), SEGMENT_FILE))


def clear_segment_cache(folder: str | None = None,
                        should_stop: Callable[[], bool] | None = None) -> Usage:
    """Deletes every segment-cache file in `folder` and nothing else.
    Returns what was removed."""
    folder = folder or segment_cache_dir()
    return _remove(folder, _matching(folder, SEGMENT_FILE), SEGMENT_FILE, should_stop)


def trim_segment_cache(max_bytes: int, folder: str | None = None,
                       should_stop: Callable[[], bool] | None = None) -> Usage:
    """Brings the segment cache under `max_bytes` by deleting the least
    recently used keys, a key's files together so a half entry can't
    linger. A key's age is the newest mtime among its files (a cache hit
    touches them). Returns what was removed."""
    folder = folder or segment_cache_dir()
    entries = _matching(folder, SEGMENT_FILE)
    total = sum(e.size for e in entries)
    if total <= max_bytes:
        return NOTHING
    groups: dict[str, list[_Entry]] = {}
    for entry in entries:
        groups.setdefault(SEGMENT_FILE.match(entry.name).group(1), []).append(entry)
    ordered = sorted(groups.items(), key=lambda kv: (max(e.mtime for e in kv[1]), kv[0]))
    removed = NOTHING
    for _key, files in ordered:
        if total <= max_bytes or (should_stop is not None and should_stop()):
            break
        gone = _remove(folder, files, SEGMENT_FILE, should_stop)
        total -= gone.bytes
        removed = Usage(removed.count + gone.count, removed.bytes + gone.bytes)
    return removed


def trim_to_setting(max_mb, folder: str | None = None,
                    should_stop: Callable[[], bool] | None = None) -> Usage:
    """`trim_segment_cache` for `settings["segment_cache_max_mb"]`: a value
    of 0, below 0 or not a number means no limit and removes nothing."""
    if isinstance(max_mb, bool) or not isinstance(max_mb, (int, float)) or not max_mb > 0:
        return NOTHING
    return trim_segment_cache(int(max_mb * MIB), folder, should_stop)


# -- Audio8 reference codes -----------------------------------------------------


def ref_codes_usage(folder: str | None = None) -> Usage:
    return _sum(_matching(folder or ref_codes_dir(), REF_CODES_FILE))


def clear_ref_codes(folder: str | None = None,
                    should_stop: Callable[[], bool] | None = None) -> Usage:
    folder = folder or ref_codes_dir()
    return _remove(folder, _matching(folder, REF_CODES_FILE), REF_CODES_FILE, should_stop)


# -- Project working copies (display only) --------------------------------------


def projects_usage(folder: str | None = None,
                   should_stop: Callable[[], bool] | None = None) -> Usage:
    """Files and bytes under `cache/projects/`. A read, nothing more: there
    is no clear function for this folder, and the app manages it
    (`project.evict_project_dirs`, `project.gc_project_dir`). Links are not
    followed."""
    folder = folder or projects_dir()
    count = size = 0
    for root, _dirs, files in os.walk(folder, followlinks=False):
        if should_stop is not None and should_stop():
            break
        for name in files:
            path = os.path.join(root, name)
            try:
                if os.path.islink(path):
                    continue
                size += os.stat(path).st_size
            except OSError:
                continue
            count += 1
    return Usage(count, size)
