"""Vector icons painted at runtime, so the shell needs no icon files and no
icon-font dependency, and every glyph can be tinted to the active theme.

`icon(name, color, size)` returns a `QIcon` for one of `ICON_NAMES`
("play", "pause", "stop") drawn as a filled shape in `color`. Pixmaps are
rendered at 2x for high-DPI screens. Callers re-request icons on
`themeChanged` (see `TransportDock._apply_icons`) since a `QIcon` holds
pixels, not a color token.
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPixmap, QPolygonF

ICON_NAMES = ("play", "pause", "stop")
_SCALE = 2


def _path(name: str, s: float) -> QPainterPath:
    path = QPainterPath()
    if name == "play":
        path.addPolygon(QPolygonF([QPointF(s * 0.32, s * 0.18), QPointF(s * 0.86, s * 0.5), QPointF(s * 0.32, s * 0.82)]))
        path.closeSubpath()
    elif name == "pause":
        path.addRoundedRect(QRectF(s * 0.24, s * 0.2, s * 0.2, s * 0.6), s * 0.05, s * 0.05)
        path.addRoundedRect(QRectF(s * 0.56, s * 0.2, s * 0.2, s * 0.6), s * 0.05, s * 0.05)
    elif name == "stop":
        path.addRoundedRect(QRectF(s * 0.24, s * 0.24, s * 0.52, s * 0.52), s * 0.08, s * 0.08)
    else:
        raise ValueError(f"unknown icon {name!r}")
    return path


def pixmap(name: str, color: str, size: int = 16) -> QPixmap:
    px = QPixmap(size * _SCALE, size * _SCALE)
    px.setDevicePixelRatio(_SCALE)
    px.fill(Qt.GlobalColor.transparent)
    painter = QPainter(px)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(color))
    painter.drawPath(_path(name, float(size)))
    painter.end()
    return px


def icon(name: str, color: str, size: int = 16) -> QIcon:
    return QIcon(pixmap(name, color, size))
