"""The headless folder in the cache: what `speak` generates, kept, and the
files that only get played, deleted once they have."""
from __future__ import annotations

import os

from kokoro_gui.engine import runtime

CACHE_FOLDER = "headless_speak"


def speak_dir() -> str:
    """`<CACHE_DIR>/headless_speak`, read at call time."""
    return os.path.abspath(os.path.join(runtime.CACHE_DIR, CACHE_FOLDER))


def scratch_path() -> str:
    """A new name in `speak_dir` for a file that only gets played."""
    os.makedirs(speak_dir(), exist_ok=True)
    return os.path.join(speak_dir(), f".play-{os.urandom(8).hex()}.wav")
