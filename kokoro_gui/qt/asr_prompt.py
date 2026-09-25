"""The ask-before-download step for the Whisper ASR model (grill PR5).

Whisper's weights (about 1.6 GB for the default `large-v3-turbo`) download
the first time a model loads. Every GUI path that can trigger that first
load calls `confirm_whisper_download` before scheduling the work, so the
user is told before the download, not after.
"""
from __future__ import annotations

from PySide6.QtWidgets import QMessageBox

from kokoro_gui.engine import asr

# `confirm_whisper_download` results.
PROCEED = "proceed"    # cached already, or the user said Yes
OTHER_ENGINE = "other"  # the user picked "Use another engine"
CANCEL = "cancel"


def ask_whisper_download(parent, model_name: str, size: str | None) -> str:
    """The Yes / Use another engine / Cancel box. Its own function so the
    GUI test fixture can answer it without a modal."""
    about = f" about {size}" if size else ""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle("Download Whisper model")
    box.setText(f"Whisper {model_name} downloads{about} the first time. Continue?")
    yes = box.addButton("Yes", QMessageBox.ButtonRole.AcceptRole)
    other = box.addButton("Use another engine", QMessageBox.ButtonRole.ActionRole)
    box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(yes)
    box.exec()
    clicked = box.clickedButton()
    if clicked is yes:
        return PROCEED
    if clicked is other:
        return OTHER_ENGINE
    return CANCEL


def confirm_whisper_download(parent) -> tuple[str, bool]:
    """`(choice, downloading)`. `choice` is `PROCEED` straight away when the
    configured model is already on disk, else what the user answered.
    `downloading` is True when the load will fetch the weights, so the
    caller can say "Downloading Whisper model..." until it returns."""
    name = asr.get_whisper_model_name()
    if asr.whisper_model_cached(name):
        return PROCEED, False
    choice = ask_whisper_download(parent, name, asr.whisper_model_size(name))
    return choice, choice == PROCEED
