"""What the app does with engines, clips and configs, without widgets
(Claude/PLAN_headless_cli_mcp.md).

`QtTTSApp` and the docks call these functions with plain values they read
off their own widgets; a headless caller builds the same values from
defaults. Nothing here imports PySide6, `qt/app.py`, `qt/subprojects.py`,
`qt/docks/*` or `audio/transport.py` (`tests/core/test_imports.py`).
`qt/project.py` and `qt/spec.py` are Qt-free and may be imported.
"""
