"""`kokoro_gui/core/` stays Qt-free (Claude/PLAN_headless_cli_mcp.md, section 1).

Checked in a fresh interpreter: the test process itself may already have
PySide6 or the app loaded by another test.
"""
import json
import os
import pkgutil
import subprocess
import sys

import kokoro_gui.core

FORBIDDEN = (
    "PySide6",
    "kokoro_gui.qt.app",
    "kokoro_gui.qt.subprojects",
    "kokoro_gui.qt.docks",
    "kokoro_gui.audio.transport",
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _core_modules() -> list:
    return ["kokoro_gui.core"] + [f"kokoro_gui.core.{m.name}" for m in pkgutil.iter_modules(kokoro_gui.core.__path__)]


def test_core_imports_no_qt():
    modules = _core_modules()
    assert {"kokoro_gui.core.engines", "kokoro_gui.core.fx", "kokoro_gui.core.config"} <= set(modules)
    script = (
        "import importlib, json, sys\n"
        f"for name in {modules!r}:\n"
        "    importlib.import_module(name)\n"
        "print(json.dumps(sorted(sys.modules)))\n"
    )
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    loaded = json.loads(result.stdout.strip().splitlines()[-1])
    bad = [name for name in loaded if any(name == f or name.startswith(f + ".") for f in FORBIDDEN)]
    assert bad == []
