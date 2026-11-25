"""Where headless output goes, and the guard on a caller-chosen `out`.

Renders land in `audio_output/headless/<YYYYmmdd-HHMMSS>_<slug>/` under the
current directory (grill PG22). A custom `out` has to resolve under that root
or under the current directory. The slug comes from caller text, so it is
reduced to a safe file name.
"""
from __future__ import annotations

import os
import re
import unicodedata
from datetime import datetime

from .errors import HeadlessError

ROOT_PARTS = ("audio_output", "headless")
MAX_SLUG = 40
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def renders_root(base: str | None = None) -> str:
    """Absolute `audio_output/headless` under `base` (default: the current directory)."""
    return os.path.abspath(os.path.join(base or os.getcwd(), *ROOT_PARTS))


def slugify(text: str, fallback: str = "render", max_len: int = MAX_SLUG) -> str:
    """A name safe on Windows, macOS and Linux: ASCII letters, digits, `_` and
    `-`; no separators, dots or spaces; never empty; never a device name such
    as `CON` or `COM1`; at most `max_len` characters."""
    text = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-_")[:max_len].strip("-_")
    if not slug:
        slug = fallback
    if re.split(r"[-_]", slug)[0].upper() in _RESERVED:
        slug = "_" + slug
    return slug[:max_len]


def new_render_dir(slug: str = "", root: str | None = None, now: datetime | None = None) -> str:
    """Create and return a fresh `<YYYYmmdd-HHMMSS>_<slug>` folder under the
    renders root. It never reuses an existing folder: on a clash (two calls in
    the same second) it appends `-2`, `-3`, and so on. `mkdir` is the check, so
    two processes can't both get the same one."""
    root = os.path.abspath(root) if root else renders_root()
    os.makedirs(root, exist_ok=True)
    stem = f"{(now or datetime.now()):%Y%m%d-%H%M%S}_{slugify(slug)}"
    n = 1
    while True:
        path = os.path.join(root, stem if n == 1 else f"{stem}-{n}")
        try:
            os.mkdir(path)
            return path
        except FileExistsError:
            n += 1


def _key(path: str) -> str:
    return os.path.normcase(os.path.realpath(path))


def _is_under(path: str, parent: str) -> bool:
    """True when `path` is `parent` or inside it. Compares whole path
    components, so `headless_evil` is not under `headless`; a path on another
    drive raises in `commonpath` and counts as outside."""
    try:
        return os.path.commonpath([path, parent]) == parent
    except ValueError:
        return False


def is_render(path: str) -> bool:
    """True for a path inside the renders root, links followed."""
    return _is_under(_key(path), _key(renders_root()))


_SEPS = "\\/"
_DEVICE = re.compile(r"^(?:CON|PRN|AUX|NUL|CONIN\$|CONOUT\$|(?:COM|LPT)[1-9¹²³])$", re.IGNORECASE)


def _is_network(path: str) -> bool:
    """Two separators first, in any mix: a UNC or device path on Windows."""
    return len(path) >= 2 and path[0] in _SEPS and path[1] in _SEPS


def _check_components(out: str) -> None:
    """Refuse names that mean something other than what they say, on every OS:
    reserved device names (`CON`, `nul.txt`, `COM¹`, with trailing dots or
    spaces), a colon (an NTFS alternate data stream, or a drive letter that is
    not the one leading drive), and a trailing dot or space, which Windows
    drops, so `x.wav.` is `x.wav`. `.` and `..` pass; `..` is the realpath
    check's job."""
    drive, rest = os.path.splitdrive(out)
    if drive:
        if not rest or rest[0] not in _SEPS:
            raise HeadlessError("out_not_allowed", "a drive-relative path is not allowed")
        out = rest
    for part in re.split(r"[\\/]", out):
        if part in ("", ".", ".."):
            continue
        if part != part.rstrip(" ."):
            raise HeadlessError("out_not_allowed", f"a name ending in a dot or a space is not allowed: {part!r}")
        if ":" in part:
            raise HeadlessError("out_not_allowed", f"a colon in a name is not allowed: {part!r}")
        if _DEVICE.match(part.split(".")[0].rstrip(" ")):
            raise HeadlessError("out_not_allowed", f"a reserved device name is not allowed: {part!r}")


def check_out_path(out, root: str | None = None, cwd: str | None = None) -> str:
    """Return the real path of `out` or raise `HeadlessError("out_not_allowed")`.

    `out` must resolve, symlinks followed, under the renders root or the
    current directory. Everything that can be refused from the text alone is
    refused before any filesystem call, so a network path never reaches
    `realpath`: an empty path, a NUL byte, a path whose first two characters
    are both separators (any mix of backslash and slash), a drive-relative path
    (`C:x.wav`), a colon in a name, a name with a trailing dot or space, and a
    reserved device name in any component. The joined path is checked for a
    network form too. A relative `out` resolves against `cwd`.

    The guard checks where `out` points, not whether it is a file or a folder.
    `.` and an existing directory pass; the caller decides what it needs. `-`
    (stdout) is the caller's to handle; here it is an ordinary relative name.
    """
    cwd = cwd or os.getcwd()
    try:
        out = os.fspath(out)
    except TypeError as exc:
        raise HeadlessError("bad_arguments", "out must be a path") from exc
    if not isinstance(out, str):
        raise HeadlessError("bad_arguments", "out must be a path")
    if not out.strip() or "\0" in out:
        raise HeadlessError("out_not_allowed", "out is empty or contains a NUL byte")
    if _is_network(out):
        raise HeadlessError("out_not_allowed", "UNC and device paths are not allowed")
    _check_components(out)
    full = out if os.path.isabs(out) else os.path.join(cwd, out)
    base = root or renders_root(cwd)
    if _is_network(full) or _is_network(cwd) or _is_network(base):
        raise HeadlessError("out_not_allowed", "UNC and device paths are not allowed")
    allowed = [_key(base), _key(cwd)]
    try:
        real = os.path.realpath(full)
    except (OSError, ValueError) as exc:
        raise HeadlessError("out_not_allowed", f"out can't be resolved: {exc}") from exc
    if not any(_is_under(os.path.normcase(real), b) for b in allowed):
        raise HeadlessError(
            "out_not_allowed",
            "out must be inside audio_output/headless or the current directory",
            valid=[renders_root(cwd), cwd],
        )
    return real
