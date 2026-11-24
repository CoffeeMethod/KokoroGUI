"""`kokoro_gui/headless/` stays Qt-free, the rule `kokoro_gui/core/` follows
(Claude/PLAN_headless_cli_mcp.md, section 1). Checked in a fresh interpreter."""
import json
import os
import pkgutil
import subprocess
import sys

import kokoro_gui.headless

FORBIDDEN = (
    "PySide6",
    "kokoro_gui.qt.app",
    "kokoro_gui.qt.subprojects",
    "kokoro_gui.qt.docks",
    "kokoro_gui.audio.transport",
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Needs the optional `mcp` package; tested with it in H10.
OPTIONAL = {"mcp_server", "__main__"}


def _modules() -> list:
    names = [m.name for m in pkgutil.iter_modules(kokoro_gui.headless.__path__) if m.name not in OPTIONAL]
    return ["kokoro_gui.headless"] + [f"kokoro_gui.headless.{n}" for n in names]


def test_headless_imports_no_qt():
    modules = _modules()
    assert {"kokoro_gui.headless.errors", "kokoro_gui.headless.results", "kokoro_gui.headless.paths",
            "kokoro_gui.headless.stdio"} <= set(modules)
    script = (
        "import importlib, json, sys\n"
        f"for name in {modules!r}:\n"
        "    importlib.import_module(name)\n"
        "print(json.dumps(sorted(sys.modules)))\n"
    )
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    loaded = json.loads(result.stdout.strip().splitlines()[-1])
    bad = [n for n in loaded if any(n == f or n.startswith(f + ".") for f in FORBIDDEN)]
    assert bad == []
