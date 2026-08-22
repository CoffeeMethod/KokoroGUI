"""Fixtures for the Qt (PySide6) frontend test suite - workstream 3a of
PLAN_qt_and_engine_abstraction.md.

`pytest.importorskip` at the top means this whole tree self-skips when the
optional `requirements-qt.txt`/`requirements-qt-test.txt` extras aren't
installed, same pattern `tests/conftest.py`'s `espeak_available()` uses for
the integration suite - `pytest` (no args) stays runnable for Tk-only
contributors who never `pip install`ed PySide6.
"""
import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

# Qt needs a platform plugin even to construct widgets; "offscreen" needs no
# real display, so the suite runs the same way in this sandbox, in CI, and on
# a dev machine with no monitor attached. Set before any PySide6 import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# tests/ has an __init__.py (package import mode), so this is tests/conftest.py's
# StubEngine, not a name collision with this file (also called conftest.py).
from tests.conftest import StubEngine  # noqa: E402


@pytest.fixture
def qt_app(tmp_path, monkeypatch, qtbot):
    import kokoro_gui.qt.app as qt_app_module
    from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(qt_app_module, "CONFIG_FILE", str(tmp_path / "config_qt.json"))
    monkeypatch.setattr(qt_app_module, "PRESETS_DIR", str(tmp_path / "presets"))
    monkeypatch.setattr(qt_app_module, "FX_PRESETS_DIR", str(tmp_path / "presets" / "fx"))
    monkeypatch.setattr(qt_app_module, "KokoroEngine", StubEngine)
    (tmp_path / "custom_voices").mkdir(exist_ok=True)

    # Modal dialogs (QMessageBox.exec/QInputDialog.exec/...) block on the
    # "offscreen" platform exactly like they would on a real display - patch
    # the statics globally (same class object every dock module imports) so
    # no test hangs waiting for a click that can never happen. Individual
    # tests can re-monkeypatch a specific return value (e.g. a preset name)
    # on top of this, since `monkeypatch` is shared across one test's fixtures.
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("", False)))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: ("", "")))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: ""))

    app = qt_app_module.QtTTSApp()
    qtbot.addWidget(app)
    yield app
    app.close()
