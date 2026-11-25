"""The files a render writes, so a failure can take them back.

A file that was there first is never replaced, except the ones an update names
(its bundle and, inside the renders root, its mixdown). Those get a hard link
beside them before they are written, and `discard` puts them back.
"""
from __future__ import annotations

import logging
import os
import shutil
from typing import Optional

from kokoro_gui.daw.mixdown import unused_path
from kokoro_gui.qt import project as project_io

from . import paths
from .errors import HeadlessError

log = logging.getLogger(__name__)


def with_parents(path: str) -> list:
    """`path` and every folder above it, outermost first."""
    chain = [os.path.abspath(path)]
    while os.path.dirname(chain[-1]) != chain[-1]:
        chain.append(os.path.dirname(chain[-1]))
    return chain[::-1]


class Output:
    """One call's output folder. With `directory` None it is a fresh
    `<stamp>_<stem>` folder under the renders root."""

    def __init__(self, directory: Optional[str], stem: str):
        target = directory or paths.renders_root()
        self._made = [d for d in with_parents(target) if not os.path.isdir(d)]
        self._files: list = []
        self._backups: list = []  # (backup, path)
        self._replaceable: set = set()
        if directory is None:
            self.dir = paths.new_render_dir(stem)
            self._made.append(self.dir)
        else:
            os.makedirs(directory, exist_ok=True)
            self.dir = directory
        self.project_path = os.path.join(self.dir, stem + project_io.DEFAULT_EXTENSION)
        self.mixdown_path = os.path.join(self.dir, stem + ".wav")

    @classmethod
    def for_update(cls, project_path: str) -> "Output":
        """Saves over `project_path`. The mixdown beside it is replaced inside
        the renders root, where every file is a render's. Anywhere else a
        `.wav` of that name may be the user's own, so the new one gets a free
        name."""
        directory, name = os.path.split(project_path)
        output = cls(directory, os.path.splitext(name)[0])
        output.project_path = project_path
        output._replaceable.add(project_path)
        if paths.is_render(project_path):
            output._replaceable.add(output.mixdown_path)
        else:
            output.mixdown_path = unused_path(output.mixdown_path)
        return output

    def check(self) -> None:
        """Before the long work: the names are free."""
        for path in (self.project_path, self.project_path + ".tmp", self.mixdown_path):
            self._refuse_existing(path)

    def _refuse_existing(self, path: str) -> None:
        if os.path.lexists(path) and path not in self._replaceable:
            raise HeadlessError("bad_arguments", f"{path} already exists")

    def claim(self, token: str, *names: str) -> None:
        """Right before writing `names`: another render may have taken one
        since the first check. `token` makes the backups' names unique."""
        for path in names:
            self._refuse_existing(path)
        for path in names:
            if not os.path.lexists(path):
                self._files.append(path)
                continue
            backup = os.path.join(self.dir, f".{token}.{len(self._backups)}.bak")
            try:
                os.link(path, backup)
            except OSError:  # no hard links here
                shutil.copy2(path, backup)
            self._backups.append((backup, path))

    def commit(self) -> None:
        for backup, _path in self._backups:
            try:
                os.remove(backup)
            except OSError:
                log.warning("couldn't remove %s", backup)

    def discard(self) -> None:
        for path in self._files:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            except OSError:
                log.warning("couldn't remove %s", path)
        for backup, path in self._backups:
            try:
                os.replace(backup, path)
            except OSError:
                log.warning("couldn't put %s back; it is at %s", path, backup)
        for directory in reversed(self._made):
            try:
                os.rmdir(directory)
            except OSError:  # gone, or someone else uses it
                pass
