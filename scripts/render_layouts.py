"""Render the two candidate default layouts of plan 47 pass 3 (grill PG33).

    python scripts/render_layouts.py --all OUT_DIR
    python scripts/render_layouts.py --layout strip --size 1440x900 --theme dark out.png

`grid` is today's two-by-two arrangement with the proportions rebalanced so
Transport takes less height. `strip` puts Timeline across the full width and
Transport in a compact full-width strip beneath it. Both use the real docks and
the existing right-hand tab group. The arrangement is applied here, after the
window is built, so no app default changes and no saved layout is touched.

The window comes from `render_screenshot.build_app`: StubEngine, offscreen, a
temp working directory, the three-character sample project. Nothing reads or
writes the user's real config. `--all` renders 2 layouts x 2 sizes x 2 themes,
one subprocess each (the app patches module globals), and writes
`measurements.json` next to the PNGs.

For the strip the script also re-flows the Transport dock's own widgets into
two lines. That re-flow is part of the mock-up only: today's Transport is a
four-row column and cannot be this short.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32":
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LAYOUTS = ("grid", "strip")
SIZES = ("1280x800", "1440x900")
THEMES = ("dark", "light")


def _pump(qapp, rounds: int = 6) -> None:
    from PySide6.QtCore import QCoreApplication, QEvent

    for _ in range(rounds):
        qapp.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.LayoutRequest)


def apply_grid(app, qapp, width: int, height: int) -> None:
    """The default grid, rebalanced: the bottom row shrinks to what Transport's
    controls need (its own minimum), and the rows above take the rest."""
    from PySide6.QtCore import Qt

    app.arrange_docks_default()
    _pump(qapp)
    need = app.transport_dock.minimumSizeHint().height() + 4
    bottom_h = max(need, int(height * 0.25))
    top_h = height - bottom_h
    left_w = int(width * 0.66)
    app.resizeDocks([app.transcript_dock, app.settings_dock], [left_w, width - left_w], Qt.Orientation.Horizontal)
    app.resizeDocks([app.timeline_dock, app.transport_dock], [left_w, width - left_w], Qt.Orientation.Horizontal)
    app.resizeDocks([app.transcript_dock, app.timeline_dock], [top_h, bottom_h], Qt.Orientation.Vertical)


def _reflow_transport_as_strip(dock) -> None:
    """Mock-up only. Line 1: transport buttons and Preview/Generate side by
    side. Line 2: level meter, progress bar and the preview target line."""
    from PySide6.QtWidgets import QHBoxLayout

    content = dock.widget()
    outer = content.layout()
    parts = []
    while outer.count():
        item = outer.takeAt(0)
        parts.append(item.layout() if item.layout() is not None else item.widget())
    row1, meter, row2, preview, progress = parts[:5]
    outer.setContentsMargins(6, 3, 6, 3)
    outer.setSpacing(3)
    line1 = QHBoxLayout()
    line1.setSpacing(16)
    line1.addLayout(row1, 3)
    line1.addLayout(row2, 2)
    outer.addLayout(line1)
    line2 = QHBoxLayout()
    line2.setSpacing(8)
    for widget, stretch in ((meter, 1), (progress, 3), (preview, 2)):
        line2.addWidget(widget, stretch)
    outer.addLayout(line2)


def apply_strip(app, qapp, width: int, height: int) -> None:
    """Timeline across the full width, Transport as a strip beneath it. The
    Bottom dock area takes both bottom corners, so the strip spans the window."""
    from PySide6.QtCore import Qt

    app.arrange_docks_default()
    _pump(qapp)
    app.setCorner(Qt.Corner.BottomLeftCorner, Qt.DockWidgetArea.BottomDockWidgetArea)
    app.setCorner(Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.BottomDockWidgetArea)
    app.removeDockWidget(app.transport_dock)
    app.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, app.transport_dock)
    app.transport_dock.show()
    _reflow_transport_as_strip(app.transport_dock)
    _pump(qapp)
    strip_h = app.transport_dock.minimumSizeHint().height() + 2
    timeline_h = int(height * 0.33)
    top_h = height - strip_h - timeline_h
    left_w = int(width * 0.64)
    app.resizeDocks([app.transcript_dock, app.settings_dock], [left_w, width - left_w], Qt.Orientation.Horizontal)
    app.resizeDocks([app.transcript_dock, app.timeline_dock, app.transport_dock], [top_h, timeline_h, strip_h],
                    Qt.Orientation.Vertical)


def _hidden_tabs(bar) -> list[str]:
    """Tab names not fully inside the bar's visible width. They take the
    scroll buttons to reach."""
    from PySide6.QtWidgets import QToolButton

    buttons = [b for b in bar.findChildren(QToolButton) if b.isVisible()]
    limit = bar.width() - sum(b.width() for b in buttons)
    return [bar.tabText(i).replace("&&", "&") for i in range(bar.count()) if bar.tabRect(i).right() > limit]


def measure(app, qapp) -> dict:
    """Pixel heights of the four regions and what needs scrolling."""
    from PySide6.QtWidgets import QAbstractButton, QAbstractScrollArea, QComboBox, QTabBar

    out = {"transcript": app.transcript_dock.height(), "timeline": app.timeline_dock.height(),
           "transport": app.transport_dock.height(), "right_panel": app.settings_dock.height(),
           "timeline_width": app.timeline_dock.width(), "transport_width": app.transport_dock.width()}
    content = app.transport_dock.widget()
    out["transport_content_needs"] = content.minimumSizeHint().height()
    bars = [b for b in app.findChildren(QTabBar) if b.count() >= 8 and b.isVisible()]
    out["tabs_behind_scroll_buttons"] = _hidden_tabs(bars[0]) if bars else []

    group = [app.settings_dock] + [d for d in app.tabifiedDockWidgets(app.settings_dock) if not d.isHidden()]
    overflow = {}
    for dock in group:
        dock.raise_()
        _pump(qapp, 3)
        extra = 0
        for area in dock.findChildren(QAbstractScrollArea):
            bar = area.verticalScrollBar()
            if area.isVisible() and bar.maximum() > 0:
                extra = max(extra, bar.maximum())
        overflow[dock.windowTitle().replace("&&", "&")] = extra
    app.settings_dock.raise_()
    _pump(qapp, 3)
    out["right_tab_scroll_px"] = overflow

    clipped = []
    for widget in [*content.findChildren(QAbstractButton), *content.findChildren(QComboBox)]:
        if not widget.isVisibleTo(content):
            continue
        bottom = widget.mapTo(content, widget.rect().bottomLeft()).y()
        right = widget.mapTo(content, widget.rect().bottomRight()).x()
        if bottom > content.height() or right > content.width():
            clipped.append(getattr(widget, "text", lambda: "")() or widget.toolTip() or type(widget).__name__)
    out["transport_controls_clipped"] = clipped
    return out


def render_one(layout: str, size: str, theme: str, out: str) -> dict:
    import tempfile

    from render_screenshot import build_app

    width, height = (int(v) for v in size.lower().split("x"))
    workdir = tempfile.mkdtemp(prefix="kokorogui_layout_")
    qapp, app = build_app(workdir, theme, "Advanced")
    app.resize(width, height)
    app.show()
    _pump(qapp)
    (apply_grid if layout == "grid" else apply_strip)(app, qapp, width, height)
    _pump(qapp, 10)
    app.timeline_dock.timeline_view.set_playhead(4.2)
    _pump(qapp)
    app.grab().save(out)
    result = measure(app, qapp)
    result.update(layout=layout, size=size, theme=theme, file=os.path.basename(out))
    app._ask_close_choice = lambda: "discard"
    app.close()
    return result


def render_all(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for layout in LAYOUTS:
        for size in SIZES:
            for theme in THEMES:
                out = os.path.join(out_dir, f"{layout}_{size}_{theme}.png")
                proc = subprocess.run([sys.executable, os.path.abspath(__file__), "--layout", layout,
                                       "--size", size, "--theme", theme, out],
                                      capture_output=True, text=True, check=False)
                line = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
                if proc.returncode != 0 or not line:
                    sys.stderr.write(proc.stdout + proc.stderr)
                    raise SystemExit(f"render failed for {layout} {size} {theme}")
                results.append(json.loads(line[-1]))
                print(os.path.basename(out))
    with open(os.path.join(out_dir, "measurements.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", nargs="?")
    parser.add_argument("--all", metavar="OUT_DIR", help="render all 8 into this folder")
    parser.add_argument("--layout", choices=LAYOUTS, default="grid")
    parser.add_argument("--size", default="1280x800")
    parser.add_argument("--theme", choices=THEMES, default="dark")
    args = parser.parse_args()
    if args.all:
        render_all(os.path.abspath(args.all))
        return
    if not args.out:
        parser.error("give an output PNG or --all OUT_DIR")
    print(json.dumps(render_one(args.layout, args.size, args.theme, os.path.abspath(args.out))))


if __name__ == "__main__":
    main()
