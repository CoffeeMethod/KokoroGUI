"""What the Voice & Clip and Audio FX tabs are editing, in words.

The tabs resolve their own mode ("none", "clip", "character"), target and
targets from `app.selection`; `describe` only names that, so there is one
scope model. `ScopeHeader` shows it above a tab's scroll area.

Nothing selected is "Defaults", not the project: those values are in
`config_qt.json` and every project starts from them. The project's own
defaults (gaps, ripple, track layout) are in Options > Settings.
"""
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from kokoro_gui.i18n import N_, tr



NAME_LIMIT = 40
EXCERPT_LIMIT = 60
DEFAULTS_TITLE = N_("Defaults")


def shorten(text: str, limit: int = NAME_LIMIT) -> str:
    """`text` on one line, cut to `limit` characters."""
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit - 3].rstrip() + "..."


def join_names(names) -> str:
    """"A", "A and B", "A, B and C"."""
    names = list(names)
    if len(names) < 2:
        return "".join(names)
    return tr("{names} and {last}", names=", ".join(names[:-1]), last=names[-1])


@dataclass(frozen=True)
class ScopeTarget:
    context: str  # "Subproject: <title>", or ""
    title: str    # "Defaults", "Character: Marta", "Clip: Marta", "3 clips: Marta, Tomas"
    detail: str   # the clip's text, "" otherwise


def describe(app, mode: str, target, targets) -> ScopeTarget:
    document = app.document
    context = (app.scope_text() if hasattr(app, "scope_text") else None) or ""
    focus = getattr(app, "focus", None)
    if not context and getattr(focus, "parent_id", None) is not None:
        context = tr("Subproject: {title}", title=focus.title())
    if mode == "character":
        return ScopeTarget(context, tr("Character: {name}", name=shorten(target.name)), "")
    if mode == "clip":
        targets = targets or [target]
        names: list = []
        for clip in targets:
            character = document.get_character(clip.character_id)
            name = shorten(character.name) if character is not None else tr("no character")
            if name not in names:
                names.append(name)
        if len(targets) > 1:
            return ScopeTarget(context, tr("{count} clips: {names}", count=len(targets),
                                              names=shorten(", ".join(names), EXCERPT_LIMIT)), "")
        return ScopeTarget(context, tr("Clip: {name}", name=names[0]), shorten(document.clip_text(target),
                                                                               EXCERPT_LIMIT))
    return ScopeTarget(context, tr(DEFAULTS_TITLE), "")


def library_note(app, character) -> str:
    """What else an edit to `character` changes (`app.commit_character_edit`
    writes a linked character through), or ""."""
    if character is None:
        return ""
    scope = app.character_scope(character)
    if scope == "global":
        return tr("It is in the character library, so every project that uses it changes.")
    if scope == "project":
        return tr("It is shared with this project's subprojects, so they change too.")
    if scope == "missing":
        return tr("Its library entry isn't on this machine, so only this project changes.")
    return ""


def muted_label(text: str = "", wrap: bool = True) -> QLabel:
    label = QLabel(text)
    label.setProperty("muted", True)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(wrap)
    return label


def set_hint(label: QLabel, text: str) -> None:
    label.setText(text)
    label.setVisible(bool(text))


class ScopeHeader(QWidget):
    """One line each, cut instead of wrapped, so the header stays short and
    doesn't raise its dock's minimum size; the tooltip has the whole text."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.context_label = muted_label(wrap=False)
        self.title_label = QLabel()
        self.title_label.setProperty("scopeTitle", True)
        self.title_label.setTextFormat(Qt.TextFormat.PlainText)
        self.detail_label = muted_label(wrap=False)
        for label in (self.context_label, self.title_label, self.detail_label):
            label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            layout.addWidget(label)
        self.context_label.hide()
        self.detail_label.hide()

    def show_target(self, target: ScopeTarget) -> None:
        set_hint(self.context_label, target.context)
        self.title_label.setText(target.title)
        set_hint(self.detail_label, target.detail)
        whole = "\n".join(part for part in (target.context, target.title, target.detail) if part)
        self.setToolTip(whole)
        self.setAccessibleName(tr("Editing {whole}", whole=whole))
