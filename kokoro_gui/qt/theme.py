"""Light/dark palettes for the Qt shell (UI10 of
Claude/PLAN_ui_shell_redesign.md).

One `Palette` dataclass of named color tokens, two instances (`LIGHT`,
`DARK`). Custom-painted widgets (the transcript gutter, the timeline lanes
and ruler, the playhead) read `current()` at paint time instead of holding
hard-coded hex constants, so a theme switch only needs a repaint.

`apply(qapp, name)` sets the `QApplication` style (Fusion, for both themes,
since the native Windows style ignores most palette roles) and a `QPalette`
built from the tokens. `set_active(name)` alone is enough for tests and
for headless code that only needs the token values.

Only `PySide6.QtGui`/`QtWidgets` are imported inside `apply`, so importing
this module for its token values doesn't need a QApplication.
"""
from __future__ import annotations

from dataclasses import dataclass

THEME_NAMES = ("light", "dark")
DEFAULT_THEME = "light"


@dataclass(frozen=True)
class Palette:
    name: str
    window: str
    panel: str
    panel_alt: str
    text: str
    text_muted: str
    gutter_bg: str
    gutter_text: str
    lane_bg: str
    lane_alt_bg: str
    lane_border: str
    ruler_bg: str
    ruler_text: str
    playhead: str
    selection_border: str
    dirty_underline: str
    split_rule: str
    fx_badge_bg: str
    fx_badge_text: str
    playing_highlight: str
    estimated_outline: str
    accent: str


LIGHT = Palette(
    name="light",
    window="#f3f3f3",
    panel="#ffffff",
    panel_alt="#ececec",
    text="#1e1e1e",
    text_muted="#6a6a6a",
    gutter_bg="#f7f7f7",
    gutter_text="#555555",
    lane_bg="#fafafa",
    lane_alt_bg="#f0f0f0",
    lane_border="#c9c9c9",
    ruler_bg="#e6e6e6",
    ruler_text="#333333",
    playhead="#d0342c",
    selection_border="#1a73e8",
    dirty_underline="#d0342c",
    split_rule="#9e9e9e",
    fx_badge_bg="#333333",
    fx_badge_text="#ffffff",
    playing_highlight="#ffe58a",
    estimated_outline="#888888",
    accent="#1a73e8",
)

DARK = Palette(
    name="dark",
    window="#202124",
    panel="#2b2b2b",
    panel_alt="#333333",
    text="#e8e8e8",
    text_muted="#a0a0a0",
    gutter_bg="#1e1e1e",
    gutter_text="#dddddd",
    lane_bg="#2b2b2b",
    lane_alt_bg="#262626",
    lane_border="#444444",
    ruler_bg="#1a1a1a",
    ruler_text="#cccccc",
    playhead="#ff6b60",
    selection_border="#ffd700",
    dirty_underline="#ff6b60",
    split_rule="#666666",
    fx_badge_bg="#000000",
    fx_badge_text="#ffffff",
    playing_highlight="#5a4d1a",
    estimated_outline="#999999",
    accent="#4a90d9",
)

_PALETTES = {"light": LIGHT, "dark": DARK}
_active: Palette = LIGHT


def palette_for(name: str) -> Palette:
    return _PALETTES.get(name, LIGHT)


def set_active(name: str) -> Palette:
    """Makes `name` the palette `current()` returns. Unknown names fall back
    to light rather than raising, same tolerance the rest of the settings
    reads have for a hand-edited config_qt.json."""
    global _active
    _active = palette_for(name)
    return _active


def current() -> Palette:
    return _active


def apply(qapp, name: str) -> Palette:
    """Applies `name` to the running QApplication: style + QPalette. Returns
    the palette so the caller can emit its own theme-changed signal."""
    from PySide6.QtGui import QColor, QPalette
    from PySide6.QtWidgets import QStyleFactory

    pal = set_active(name)
    if qapp is None:
        return pal

    # Fusion for both themes: it honors every QPalette role on every
    # platform (the native Windows style ignores most of them, and paints
    # black boxes under the offscreen platform the screenshot script and
    # the test suite use), so light and dark render through one code path.
    qapp.setStyle(QStyleFactory.create("Fusion"))
    qpal = QPalette()
    roles = {
        QPalette.ColorRole.Window: pal.window,
        QPalette.ColorRole.WindowText: pal.text,
        QPalette.ColorRole.Base: pal.panel,
        QPalette.ColorRole.AlternateBase: pal.panel_alt,
        QPalette.ColorRole.ToolTipBase: pal.panel_alt,
        QPalette.ColorRole.ToolTipText: pal.text,
        QPalette.ColorRole.Text: pal.text,
        QPalette.ColorRole.Button: pal.panel_alt,
        QPalette.ColorRole.ButtonText: pal.text,
        QPalette.ColorRole.BrightText: "#ff5555",
        QPalette.ColorRole.Highlight: pal.accent,
        QPalette.ColorRole.HighlightedText: "#ffffff",
        QPalette.ColorRole.Link: pal.accent,
        QPalette.ColorRole.PlaceholderText: pal.text_muted,
        QPalette.ColorRole.Light: pal.panel if pal.name == "light" else pal.panel_alt,
        QPalette.ColorRole.Midlight: pal.panel_alt,
        QPalette.ColorRole.Mid: pal.lane_border,
        QPalette.ColorRole.Dark: pal.lane_border,
        QPalette.ColorRole.Shadow: pal.text_muted if pal.name == "light" else "#000000",
    }
    for role, color in roles.items():
        qpal.setColor(role, QColor(color))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText, QPalette.ColorRole.WindowText):
        qpal.setColor(QPalette.ColorGroup.Disabled, role, QColor(pal.text_muted))
    qapp.setPalette(qpal)
    return pal


