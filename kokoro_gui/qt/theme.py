"""Light/dark palettes for the Qt shell (UI10 of
Claude/PLAN_ui_shell_redesign.md).

One `Palette` dataclass of named color tokens, two instances (`LIGHT`,
`DARK`). Custom-painted widgets (the transcript gutter, the timeline lanes
and ruler, the playhead) read `current()` at paint time instead of holding
hard-coded hex constants, so a theme switch only needs a repaint.

`apply(qapp, name)` sets the `QApplication` style (Fusion, for both themes,
since the native Windows style ignores most palette roles), a `QPalette`
built from the tokens, the application font (`FONT_FAMILIES`,
`FONT_POINT_SIZE`) and the stylesheet `stylesheet(pal)` renders from the
same tokens (flat dock titles, borderless group boxes, rounded inputs and
buttons, underlined tabs, thin scrollbars, a flat progress bar). Widgets
opt into the two button variants with dynamic properties:
`setProperty("primary", True)` for the one filled accent button in a row,
`setProperty("transport", True)` for the round play/pause/stop buttons.
`set_active(name)` alone is enough for tests and for headless code that
only needs the token values.

Only `PySide6.QtGui`/`QtWidgets` are imported inside `apply`, so importing
this module for its token values doesn't need a QApplication.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# Chevrons for combo/spin boxes and the checkbox tick, one file per theme
# where the stroke color differs. QSS takes `image: url(<file>)` only, so
# these are files on disk rather than inline data.
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

THEME_NAMES = ("light", "dark")
DEFAULT_THEME = "dark"

# First installed family wins (Qt resolves the list itself); the last two
# are on every Linux box, Segoe UI on every Windows one.
FONT_FAMILIES = ("Segoe UI", "Inter", "SF Pro Text", "Helvetica Neue", "Ubuntu", "Noto Sans", "DejaVu Sans")
FONT_POINT_SIZE = 10
# The transcript is read for minutes at a time; one point over the UI font.
EDITOR_FONT_POINT_SIZE = FONT_POINT_SIZE + 1


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
    accent_hover: str
    border: str
    hover: str


# Three surface levels per theme: `window` (dock area, menus, ruler,
# gutter), `panel` (editor, lists, timeline lanes) and `panel_alt` (dock
# titles, buttons, track headers). Everything else is a line or a text
# color on one of those.
LIGHT = Palette(
    name="light",
    window="#f4f4f5",
    panel="#ffffff",
    panel_alt="#e9e9eb",
    text="#1c1c1e",
    text_muted="#6b6f76",
    gutter_bg="#f4f4f5",
    gutter_text="#55595f",
    lane_bg="#ffffff",
    lane_alt_bg="#f7f7f8",
    lane_border="#d4d4d8",
    ruler_bg="#ececee",
    ruler_text="#55595f",
    playhead="#e5484d",
    selection_border="#1c1c1e",
    dirty_underline="#e5484d",
    split_rule="#a1a1aa",
    fx_badge_bg="#3f3f46",
    fx_badge_text="#ffffff",
    playing_highlight="#f5d87a",
    estimated_outline="#8a8f98",
    accent="#2563eb",
    accent_hover="#1d4fd8",
    border="#d4d4d8",
    hover="#dedee1",
)

DARK = Palette(
    name="dark",
    window="#1e1f22",
    panel="#26282b",
    panel_alt="#2f3136",
    text="#e4e5e7",
    text_muted="#8f939a",
    gutter_bg="#1e1f22",
    gutter_text="#b0b3b8",
    lane_bg="#26282b",
    lane_alt_bg="#222427",
    lane_border="#3a3d42",
    ruler_bg="#1e1f22",
    ruler_text="#b0b3b8",
    playhead="#f0716a",
    selection_border="#f5f5f5",
    dirty_underline="#f0716a",
    split_rule="#5a5e66",
    fx_badge_bg="#111214",
    fx_badge_text="#f0f0f0",
    playing_highlight="#8a7a2a",
    estimated_outline="#8f939a",
    accent="#4c8df6",
    accent_hover="#6ba1f8",
    border="#3a3d42",
    hover="#383b41",
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


def _rgba(hex_color: str, alpha: float) -> str:
    """`#rrggbb` -> `rgba(r, g, b, a)` for the few QSS rules that need a
    translucent token (the progress chunk under its own status text)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha:.2f})"


def stylesheet(pal: Palette) -> str:
    """The application stylesheet for `pal`. Only tokens from the palette
    go in, so a theme switch is `qapp.setStyleSheet(stylesheet(pal))` and
    nothing else."""
    on_accent = "#ffffff"
    chunk = _rgba(pal.accent, 0.45)

    def asset(name: str) -> str:
        return os.path.join(ASSETS_DIR, name).replace("\\", "/")

    down = asset(f"chevron_down_{pal.name}.svg")
    up = asset(f"chevron_up_{pal.name}.svg")
    check = asset("check.svg")
    return f"""
QToolTip {{
    background: {pal.panel_alt}; color: {pal.text};
    border: 1px solid {pal.border}; padding: 4px 6px;
}}
QMainWindow::separator {{ background: {pal.window}; width: 4px; height: 4px; }}
QMainWindow::separator:hover {{ background: {pal.accent}; }}

QDockWidget::title {{
    background: {pal.panel_alt}; color: {pal.text_muted};
    padding: 4px 8px; text-align: left;
}}
QDockWidget::close-button, QDockWidget::float-button {{
    border: none; background: transparent; padding: 0; icon-size: 10px;
}}
QDockWidget::close-button:hover, QDockWidget::float-button:hover {{ background: {pal.hover}; border-radius: 3px; }}

QGroupBox {{
    border: none; margin-top: 16px; padding: 0;
    font-weight: 600; color: {pal.text_muted};
}}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; left: 0; padding: 0; }}
QGroupBox QWidget {{ font-weight: normal; color: {pal.text}; }}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit,
QListWidget, QListView, QTreeView, QTableView {{
    background: {pal.panel}; color: {pal.text};
    border: 1px solid {pal.border}; border-radius: 4px;
    selection-background-color: {pal.accent}; selection-color: {on_accent};
}}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{ padding: 2px 6px; min-height: 20px; }}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QTextEdit:focus, QPlainTextEdit:focus {{ border-color: {pal.accent}; }}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    color: {pal.text_muted}; background: {pal.window};
}}
QComboBox::drop-down {{ border: none; width: 20px; subcontrol-origin: padding; subcontrol-position: right center; }}
QComboBox::down-arrow {{ image: url({down}); width: 10px; height: 10px; }}
QComboBox QAbstractItemView {{
    background: {pal.panel}; border: 1px solid {pal.border};
    selection-background-color: {pal.accent}; selection-color: {on_accent}; outline: none;
}}
QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::down-button {{
    border: none; width: 16px; background: transparent;
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url({up}); width: 8px; height: 8px; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url({down}); width: 8px; height: 8px; }}

QCheckBox {{ spacing: 6px; }}
QCheckBox::indicator, QGroupBox::indicator {{
    width: 14px; height: 14px; border-radius: 3px;
    border: 1px solid {pal.border}; background: {pal.panel};
}}
QCheckBox::indicator:hover {{ border-color: {pal.text_muted}; }}
QCheckBox::indicator:checked, QGroupBox::indicator:checked {{
    background: {pal.accent}; border-color: {pal.accent}; image: url({check});
}}
QCheckBox::indicator:disabled {{ background: {pal.window}; }}
QCheckBox::indicator:checked:disabled {{ background: {pal.text_muted}; border-color: {pal.text_muted}; }}
QMenu::indicator {{ width: 14px; height: 14px; }}

QPushButton, QToolButton {{
    background: {pal.panel_alt}; color: {pal.text};
    border: 1px solid {pal.border}; border-radius: 4px;
    padding: 3px 12px; min-height: 20px;
}}
QToolButton {{ padding: 3px 8px; }}
QPushButton:hover, QToolButton:hover {{ background: {pal.hover}; }}
QPushButton:pressed, QToolButton:pressed {{ background: {pal.border}; }}
QPushButton:checked, QToolButton:checked {{ background: {pal.accent}; color: {on_accent}; border-color: {pal.accent}; }}
QPushButton:disabled, QToolButton:disabled {{ color: {pal.text_muted}; background: {pal.window}; border-color: {pal.border}; }}
QPushButton:flat, QToolButton[autoRaise="true"] {{ background: transparent; border-color: transparent; }}
QPushButton:flat:hover, QToolButton[autoRaise="true"]:hover {{ background: {pal.hover}; }}
QToolButton::menu-button {{ border: none; border-left: 1px solid {pal.border}; width: 16px; }}
QToolButton::menu-arrow {{ image: url({down}); width: 10px; height: 10px; }}
QToolButton[primary="true"]::menu-arrow {{ image: url({asset("chevron_down_on_accent.svg")}); }}

QPushButton[primary="true"], QToolButton[primary="true"] {{
    background: {pal.accent}; color: {on_accent}; border-color: {pal.accent}; font-weight: 600;
}}
QPushButton[primary="true"]:hover, QToolButton[primary="true"]:hover {{ background: {pal.accent_hover}; border-color: {pal.accent_hover}; }}
QPushButton[primary="true"]:disabled, QToolButton[primary="true"]:disabled {{
    background: {pal.panel_alt}; color: {pal.text_muted}; border-color: {pal.border};
}}
QToolButton[primary="true"]::menu-button {{ border-left-color: {_rgba("#ffffff", 0.35)}; }}
QToolButton[primary="true"]:disabled::menu-button {{ border-left-color: {pal.border}; }}

QToolButton[transport="true"] {{
    min-width: 30px; max-width: 30px; min-height: 30px; max-height: 30px;
    padding: 0; border-radius: 15px;
}}

QTabBar::tab {{
    background: transparent; color: {pal.text_muted};
    border: none; padding: 5px 12px; margin: 0;
}}
QTabBar::tab:top {{ border-bottom: 2px solid transparent; }}
QTabBar::tab:bottom {{ border-top: 2px solid transparent; }}
QTabBar::tab:hover {{ color: {pal.text}; }}
QTabBar::tab:selected {{ color: {pal.text}; }}
QTabBar::tab:top:selected {{ border-bottom-color: {pal.accent}; }}
QTabBar::tab:bottom:selected {{ border-top-color: {pal.accent}; }}
QTabWidget::pane {{ border: none; border-top: 1px solid {pal.border}; }}

QProgressBar {{
    background: {pal.panel_alt}; color: {pal.text};
    border: none; border-radius: 4px; text-align: center;
    min-height: 20px; max-height: 20px;
}}
QProgressBar::chunk {{ background: {chunk}; border-radius: 4px; }}

QSlider::groove:horizontal {{ height: 4px; background: {pal.border}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {pal.accent}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    width: 14px; height: 14px; margin: -5px 0; border-radius: 7px;
    background: {pal.text}; border: none;
}}
QSlider::handle:horizontal:hover {{ background: {pal.accent}; }}
QSlider::groove:vertical {{ width: 4px; background: {pal.border}; border-radius: 2px; }}
QSlider::add-page:vertical {{ background: {pal.accent}; border-radius: 2px; }}
QSlider::handle:vertical {{
    width: 14px; height: 14px; margin: 0 -5px; border-radius: 7px;
    background: {pal.text}; border: none;
}}

QMenuBar {{ background: {pal.window}; padding: 2px 4px; }}
QMenuBar::item {{ padding: 4px 8px; border-radius: 4px; background: transparent; }}
QMenuBar::item:selected {{ background: {pal.hover}; }}
QMenu {{ background: {pal.panel}; border: 1px solid {pal.border}; padding: 4px; }}
QMenu::item {{ padding: 4px 24px 4px 10px; border-radius: 3px; }}
QMenu::item:selected {{ background: {pal.accent}; color: {on_accent}; }}
QMenu::item:disabled {{ color: {pal.text_muted}; }}
QMenu::separator {{ height: 1px; background: {pal.border}; margin: 4px 6px; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {pal.border}; border-radius: 3px; min-height: 24px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {pal.border}; border-radius: 3px; min-width: 24px; margin: 2px; }}
QScrollBar::handle:hover {{ background: {pal.text_muted}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QHeaderView::section {{
    background: {pal.panel_alt}; color: {pal.text_muted};
    border: none; border-bottom: 1px solid {pal.border}; padding: 3px 6px;
}}
QSplitter::handle {{ background: {pal.window}; }}
QStatusBar {{ background: {pal.window}; }}
"""


def apply(qapp, name: str) -> Palette:
    """Applies `name` to the running QApplication: style, QPalette, font
    and stylesheet. Returns the palette so the caller can emit its own
    theme-changed signal."""
    from PySide6.QtGui import QColor, QFont, QPalette
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

    font = QFont()
    font.setFamilies(list(FONT_FAMILIES))
    font.setPointSize(FONT_POINT_SIZE)
    qapp.setFont(font)
    qapp.setStyleSheet(stylesheet(pal))
    return pal


