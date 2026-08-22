"""PySide6 (Qt) frontend — Workstream 3a of PLAN_qt_and_engine_abstraction.md.

This is the sole GUI frontend (`python main.py`). It talks to `KokoroEngine` /
the `kokoro_gui.engines` backend registry through the same interface the
retired Tk frontend (`gui.py`, `kokoro_gui/ui/*.py`) used to — nothing here
depends on anything Tk-specific.
"""
