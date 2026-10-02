"""Help menu dialogs: About (versions, device, engines, paths) and the
keyboard shortcut sheet. Both are read-only plain text built from live
objects, so they can't drift from the app. No network calls, and no app
import (the app passes itself in)."""
from __future__ import annotations

import os
import sys

from PySide6.QtGui import QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QPlainTextEdit, QPushButton, QVBoxLayout

import kokoro_gui
from kokoro_gui.engine import asr, runtime
from kokoro_gui.engines import registry

SHORTCUT_DESCRIPTION_PROPERTY = "description"  # set on a QShortcut to label its row in the sheet


def device_summary() -> str:
    """What the engines can use: "CUDA (<device name>)", "MPS (Apple)" or
    "CPU". Every torch call is guarded, so a missing or broken torch reads
    as CPU."""
    try:
        import torch
    except Exception:
        return "CPU (torch not installed)"
    try:
        if torch.cuda.is_available():
            return f"CUDA ({torch.cuda.get_device_name(0)})"
    except Exception:
        pass
    try:
        if torch.backends.mps.is_available():
            return "MPS (Apple)"
    except Exception:
        pass
    return "CPU"


def _torch_version() -> str:
    try:
        import torch

        return str(torch.__version__)
    except Exception:
        return "not installed"


def _pyside_version() -> str:
    try:
        import PySide6

        return str(PySide6.__version__)
    except Exception:
        return "unknown"


def _engine_lines() -> list[str]:
    lines = []
    for engine_id in registry.list_all_engines():
        name = registry.get_display_name(engine_id)
        reason = registry.unavailable_reason(engine_id)
        if reason:
            lines.append(f"  {name} ({engine_id}): unavailable. {reason}")
            continue
        try:
            version = registry.package_version(engine_id)
        except Exception:
            version = None
        lines.append(f"  {name} ({engine_id}): {version or 'version unknown'}")
    return lines


def about_text(config_file: str) -> str:
    """The About dialog's text. `config_file` is the program settings file
    the app reads (the app owns that path)."""
    model = asr.get_whisper_model_name()
    cached = "cached" if asr.whisper_model_cached(model) else "not downloaded yet"
    lines = [
        f"KokoroGUI {kokoro_gui.APP_VERSION}",
        f"Python {sys.version.split()[0]}, PySide6 {_pyside_version()}, torch {_torch_version()}",
        f"Device: {device_summary()}",
        "",
        "Engines:",
        *_engine_lines(),
        "",
        f"Cache folder: {os.path.abspath(runtime.CACHE_DIR)}",
        f"Custom voices folder: {os.path.abspath(runtime.CUSTOM_VOICES_DIR)}",
        f"Settings file: {os.path.abspath(config_file)}",
        f"Whisper model: {model} ({cached})",
    ]
    return "\n".join(lines)


def _plain(text: str) -> str:
    return text.replace("&", "").removesuffix("...").strip()


def _menu_shortcuts(menu, trail: str) -> list[tuple[str, str, str]]:
    rows = []
    for action in menu.actions():
        submenu = action.menu()
        if submenu is not None:
            rows.extend(_menu_shortcuts(submenu, f"{trail} > {_plain(action.text())}"))
        elif not action.isSeparator() and not action.shortcut().isEmpty():
            keys = ", ".join(s.toString(QKeySequence.SequenceFormat.PortableText) for s in action.shortcuts())
            rows.append((trail, _plain(action.text()), keys))
    return rows


def _humanize(attribute: str) -> str:
    return attribute.removesuffix("_shortcut").replace("_", " ").strip().capitalize()


def collect_shortcuts(app) -> list[tuple[str, str, str]]:
    """(group, label, keys) for every menu-bar action with a shortcut, then
    every `QShortcut` kept as an attribute of `app`. Menus come in menu-bar
    order; the loose shortcuts are grouped under "Other"."""
    rows: list[tuple[str, str, str]] = []
    for top in app.menuBar().actions():
        menu = top.menu()
        if menu is not None:
            rows.extend(_menu_shortcuts(menu, _plain(top.text())))
    for attribute, value in vars(app).items():
        if isinstance(value, QShortcut) and not value.key().isEmpty():
            label = value.property(SHORTCUT_DESCRIPTION_PROPERTY) or _humanize(attribute)
            rows.append(("Other", str(label), value.key().toString(QKeySequence.SequenceFormat.PortableText)))
    return rows


def shortcuts_text(app) -> str:
    lines: list[str] = []
    group = None
    for row_group, label, keys in collect_shortcuts(app):
        if row_group != group:
            if lines:
                lines.append("")
            lines.append(row_group)
            group = row_group
        lines.append(f"  {keys:<18}{label}")
    return "\n".join(lines)


class _TextDialog(QDialog):
    def __init__(self, parent, title: str, text: str, copy_button: bool, size=(560, 420)):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(*size)
        layout = QVBoxLayout(self)
        self.text_edit = QPlainTextEdit(text)
        self.text_edit.setReadOnly(True)
        layout.addWidget(self.text_edit)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        self.copy_button = None
        if copy_button:
            self.copy_button = QPushButton("Copy")
            self.copy_button.clicked.connect(self.copy_text)
            buttons.addButton(self.copy_button, QDialogButtonBox.ButtonRole.ActionRole)
        layout.addWidget(buttons)

    def text(self) -> str:
        return self.text_edit.toPlainText()

    def copy_text(self) -> None:
        QGuiApplication.clipboard().setText(self.text())


class AboutDialog(_TextDialog):
    def __init__(self, parent, config_file: str):
        super().__init__(parent, "About KokoroGUI", about_text(config_file), copy_button=True)


class ShortcutsDialog(_TextDialog):
    def __init__(self, app):
        super().__init__(app, "Keyboard shortcuts", shortcuts_text(app), copy_button=False, size=(480, 520))
