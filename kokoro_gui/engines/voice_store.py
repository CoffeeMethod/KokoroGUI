"""Where an engine's own voices live, for the two kinds that are files.

An adapter declares `voice_kind` (kokoro_gui/engines/base.py):

- "named": voices built into the model, nothing on disk (Dummy).
- "embedding": one file per voice, `<name><extension>`, in the global
  `runtime.CUSTOM_VOICES_DIR` and the project's `engines/<id>/voices/`
  (`EmbeddingStore`; Kokoro's `.pt` mixes).
- "reference": a `<name>.wav` + `<name>.txt` transcript pair per voice, in
  `custom_voices/<id>_refs/` and the project's `engines/<id>/refs/`
  (`ReferenceStore`; Audio8's clones).

Reads look in the open project's copy first (grill TB3); writes go to the
global store only (nothing writes into a project dir but Open and clip
generation). Names are reduced to their basename before they touch a path.
`BackendHooksMixin` builds `get_voices` and `collect_project_assets` from
the store, so an engine of either kind gets listing, bundling and (for
"reference") the Voice Reference editor without code of its own.
"""
from __future__ import annotations

import os
import shutil
from typing import Callable, Optional

from kokoro_gui.engine import runtime
from kokoro_gui.engine.paths import ensure_private_dir


def _safe(name) -> str:
    return os.path.basename(name or "")


class EmbeddingStore:
    """One file per voice. `global_dir` is a zero-argument callable, read at
    call time, so a test's patch of `runtime.CUSTOM_VOICES_DIR` reaches it."""

    def __init__(self, engine_id: str, extension: str, global_dir: Optional[Callable[[], str]] = None):
        self.engine_id = engine_id
        self.extension = extension
        self._global_dir = global_dir or (lambda: runtime.CUSTOM_VOICES_DIR)

    @property
    def global_dir(self) -> str:
        return self._global_dir()

    @property
    def project_subdir(self) -> str:
        return f"engines/{self.engine_id}/voices"

    def search_dirs(self, project_dir: Optional[str] = None) -> list:
        dirs = [os.path.join(project_dir, *self.project_subdir.split("/"))] if project_dir else []
        return dirs + [self.global_dir]

    def find(self, name: str, project_dir: Optional[str] = None) -> Optional[str]:
        """Absolute path of the voice's file, project copy first, or None."""
        safe = _safe(name)
        if not safe:
            return None
        for directory in self.search_dirs(project_dir):
            path = os.path.join(directory, f"{safe}{self.extension}")
            if os.path.exists(path):
                return os.path.abspath(path)
        return None

    def list_voices(self, project_dir: Optional[str] = None) -> list:
        """Voice names, the project's first, each once."""
        names = []
        for directory in self.search_dirs(project_dir):
            if not os.path.isdir(directory):
                continue
            for f in sorted(os.listdir(directory)):
                if f.endswith(self.extension):
                    name = f[: -len(self.extension)]
                    if name not in names:
                        names.append(name)
        return names

    def bundle_assets(self, name: str, project_dir: Optional[str] = None) -> list:
        from kokoro_gui.engines.base import bundle_asset_for

        asset = bundle_asset_for(name, self.global_dir, self.extension, self.project_subdir, project_dir)
        return [asset] if asset is not None else []

    def delete(self, name: str) -> None:
        path = os.path.join(self.global_dir, f"{_safe(name)}{self.extension}")
        if os.path.exists(path):
            os.remove(path)


class ReferenceStore:
    """A wav + transcript pair per voice. `global_dir` is a zero-argument
    callable read at call time (Audio8 passes its module's `AUDIO8_REFS_DIR`,
    which its tests patch); by default `custom_voices/<id>_refs/`.
    Transcripts are cached by `(path, mtime, size)`, so the dirty check,
    which reads them through `cache_key_extra`, costs a stat per clip."""

    def __init__(self, engine_id: str, global_dir: Optional[Callable[[], str]] = None):
        self.engine_id = engine_id
        self._global_dir = global_dir or (lambda: os.path.join(runtime.CUSTOM_VOICES_DIR, f"{engine_id}_refs"))
        self._transcript_cache: dict = {}

    @property
    def global_dir(self) -> str:
        return self._global_dir()

    @property
    def project_subdir(self) -> str:
        return f"engines/{self.engine_id}/refs"

    @staticmethod
    def _safe_name(name: str) -> str:
        # The path-traversal guard every voice store applies.
        return _safe(name)

    def search_dirs(self, project_dir: Optional[str] = None) -> list:
        dirs = [os.path.join(project_dir, *self.project_subdir.split("/"))] if project_dir else []
        return dirs + [self.global_dir]

    def find_wav(self, name: str, project_dir: Optional[str] = None) -> Optional[str]:
        """Absolute path of `<name>.wav`, project-local first, or `None`."""
        safe_name = self._safe_name(name)
        if not safe_name:
            return None
        for directory in self.search_dirs(project_dir):
            path = os.path.join(directory, f"{safe_name}.wav")
            if os.path.isfile(path):
                return os.path.abspath(path)
        return None

    find = find_wav

    def read_transcript_file(self, txt_path: str) -> str:
        """The stripped text of `txt_path`, memoized on the file's
        `(mtime, size)` (size too, so a rewrite inside one mtime tick still
        misses); `""` when the file is missing."""
        try:
            stat = os.stat(txt_path)
        except OSError:
            return ""
        stamp = (stat.st_mtime, stat.st_size)
        cached = self._transcript_cache.get(txt_path)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
        except OSError:
            return ""
        self._transcript_cache[txt_path] = (stamp, text)
        return text

    def save_reference(self, name: str, wav_path: str, transcript: str) -> str:
        """Copies `wav_path` and writes `transcript` under a sanitized
        `name` in the global store, creating it (private) if needed. Returns
        the saved wav's absolute path."""
        safe_name = self._safe_name(name)
        if not safe_name:
            raise ValueError("Reference name must not be empty.")
        directory = self.global_dir
        ensure_private_dir(directory, fallback=False)
        out_wav = os.path.join(directory, f"{safe_name}.wav")
        out_txt = os.path.join(directory, f"{safe_name}.txt")
        shutil.copyfile(wav_path, out_wav)
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(transcript.strip())
        return os.path.abspath(out_wav)

    def list_references(self, project_dir: Optional[str] = None) -> list:
        """Sorted names of every wav+txt pair in the project dir or the
        global store (a lone `.wav` or `.txt` is an interrupted save, not a
        usable reference)."""
        names = set()
        for directory in self.search_dirs(project_dir):
            if not os.path.isdir(directory):
                continue
            for f in os.listdir(directory):
                if not f.endswith(".wav"):
                    continue
                name = f[:-4]
                if os.path.isfile(os.path.join(directory, f"{name}.txt")):
                    names.add(name)
        return sorted(names)

    list_voices = list_references

    def get_transcript(self, name: str, project_dir: Optional[str] = None) -> str:
        wav = self.find_wav(name, project_dir)
        if wav is None:
            return ""
        return self.read_transcript_file(os.path.splitext(wav)[0] + ".txt")

    def global_wav_path(self, name: str) -> str:
        """Where `name`'s wav goes in the global store (it may not exist)."""
        return os.path.abspath(os.path.join(self.global_dir, f"{self._safe_name(name)}.wav"))

    def delete_reference(self, name: str) -> None:
        safe_name = self._safe_name(name)
        for ext in (".wav", ".txt"):
            path = os.path.join(self.global_dir, f"{safe_name}{ext}")
            if os.path.exists(path):
                os.remove(path)

    delete = delete_reference

    def bundle_assets(self, name: str, project_dir: Optional[str] = None) -> list:
        """The wav and txt of `name` as `engines/<id>/refs/<name>.{wav,txt}`."""
        from kokoro_gui.engines.base import bundle_asset_for

        assets = []
        for ext in (".wav", ".txt"):
            asset = bundle_asset_for(name, self.global_dir, ext, self.project_subdir, project_dir)
            if asset is not None:
                assets.append(asset)
        return assets
