"""PySide6 (Qt) frontend — Workstream 3a of PLAN_qt_and_engine_abstraction.md.

This package is an alternative presentation layer that talks to the exact same
`KokoroEngine` / `kokoro_gui.engines` backend registry the CustomTkinter `gui.py`
app uses. Nothing here imports from `gui.py` or `kokoro_gui/ui/*.py` (the Tk
tab-builder mixins), and nothing in those Tk files imports from here — the two
frontends are independent and ship side by side (`python main.py` vs
`python main_qt.py`) until this one reaches parity, per the plan's own risk
mitigation ("no forced cutover").
"""
