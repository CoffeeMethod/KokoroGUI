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
    monkeypatch.setattr(qt_app_module, "DOCUMENT_FILE", str(tmp_path / "document.json"))
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

    # A dirty project asks Save / Discard / Cancel on close and before New
    # or Open (grill TB12); nearly every test leaves edits behind, so the
    # fixture answers Discard. A test about the prompt re-patches this.
    monkeypatch.setattr(qt_app_module.QtTTSApp, "_ask_close_choice", lambda self: "discard")

    app = qt_app_module.QtTTSApp()
    qtbot.addWidget(app)
    yield app
    app.wait_for_project_io()
    app.close()


@pytest.fixture
def make_tagged_document():
    """Factory for a `kokoro_gui.daw.models.Document` whose clips are placed
    at specific text offsets - a test-only convenience, since `Document`
    itself has no offsets to set directly any more
    (Claude/PLAN_text_editor_redesign.md's run-list rework). Pass
    `tagged_ranges` as `[(start, end, clip), ...]`; everything else forwards
    straight to `Document(...)`."""
    from kokoro_gui.daw.models import Document, Run

    def _make(text: str, tagged_ranges=(), **kwargs):
        runs = []
        cursor = 0
        for start, end, clip in sorted(tagged_ranges, key=lambda t: t[0]):
            if start > cursor:
                runs.append(Run(text=text[cursor:start]))
            runs.append(Run(text=text[start:end], clip_id=clip.id, kind=clip.source))
            cursor = end
        if cursor < len(text):
            runs.append(Run(text=text[cursor:]))
        clips = kwargs.pop("clips", None)
        if clips is None:
            clips = [clip for _start, _end, clip in tagged_ranges]
        return Document(runs=runs, clips=clips, **kwargs)

    return _make
