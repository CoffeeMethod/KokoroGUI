"""Show a file or folder in the system file manager. Used by File > Show in
Folder. No app import, so tests drive it with `subprocess.Popen` and
`QDesktopServices.openUrl` patched."""
from __future__ import annotations

import os
import subprocess
import sys

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices


def reveal(path: str | None) -> bool:
    """Open `path`'s folder with the file selected (a file) or the folder
    itself (a directory). False when the path is empty or doesn't exist."""
    if not path:
        return False
    target = os.path.realpath(path)
    if not os.path.exists(target):
        return False
    if os.path.isdir(target):
        return bool(QDesktopServices.openUrl(QUrl.fromLocalFile(target)))
    if sys.platform == "win32":
        subprocess.Popen(["explorer", "/select,", os.path.normpath(target)])
        return True
    if sys.platform == "darwin":
        subprocess.Popen(["open", "-R", target])
        return True
    return bool(QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(target))))
