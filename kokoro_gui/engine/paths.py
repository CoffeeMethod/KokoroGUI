"""Private (0o700) directories for the cache and user stores.

Segment files are named by their segment key, which is the same on every
machine by design. On a shared host, a cache directory other users can
write to would let one of them plant a file under a key another user's
generation then plays. A directory only its owner can open can't be
planted in, so the file names don't need a salt.

POSIX only in effect: on Windows the default locations sit under the
user's own profile and NTFS ACLs, not mode bits, decide access, so
`ensure_private_dir` only creates the directory there.
"""
import os
import stat
import sys

PRIVATE_MODE = 0o700
_APP_CACHE_NAME = "kokorogui"


def _warn(message):
    print(f"Warning: {message}")


def user_cache_root():
    """`$XDG_CACHE_HOME/kokorogui`, else `~/.cache/kokorogui`."""
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, _APP_CACHE_NAME)


def _fallback_for(path):
    return os.path.join(user_cache_root(), os.path.basename(os.path.normpath(path)))


def _create_private(path):
    os.makedirs(path, mode=PRIVATE_MODE, exist_ok=True)


def ensure_private_dir(path, fallback=True):
    """Creates `path` as a 0o700 directory and returns the directory to use.

    On POSIX, for a directory that already exists: owned by this user with
    group or other bits set, it's chmod'ed to 0o700 once, with a line
    saying so (a chmod that fails, e.g. on a read-only mount, is a warning).
    Owned by another user: with `fallback`, a per-user directory under
    `user_cache_root()` is created the same way and returned, with a
    warning naming both paths; without it (a store of the user's own files,
    where moving silently would lose them) `path` is returned with a
    warning. On Windows it only creates the directory."""
    _create_private(path)
    if sys.platform == "win32" or not hasattr(os, "getuid"):
        return path

    try:
        st = os.stat(path)
    except OSError as e:
        _warn(f"can't stat {path}: {e}")
        return path

    if st.st_uid != os.getuid():
        if not fallback:
            _warn(f"{path} belongs to another user; other users may be able to read or change its files.")
            return path
        alt = _fallback_for(path)
        _warn(f"{path} belongs to another user; using {alt} instead.")
        _create_private(alt)
        _tighten(alt)
        return alt

    _tighten(path)
    return path


def _tighten(path):
    try:
        mode = stat.S_IMODE(os.stat(path).st_mode)
    except OSError:
        return
    if mode & 0o077:
        try:
            os.chmod(path, PRIVATE_MODE)
            print(f"Made {path} private to this user (was {oct(mode)}, now 0o700).")
        except OSError as e:
            _warn(f"couldn't make {path} private: {e}")
