"""
Shared fixtures for the KokoroGUI test suite.

Policy: every test that generates a config dict should build it through
`make_config`, which defaults `caching=False`. Only tests/test_caching.py
is allowed to override that to True (enforced by
tests/test_meta_caching_policy.py). This keeps caching off by default
without every test having to remember to pass it explicitly.
"""
import os
import re
import sys
import time
import threading
import concurrent.futures
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

import kokoro_engine
from kokoro_engine import KokoroEngine

# On some Windows Store ("WindowsApps") Python installs, Tcl/Tk's own
# init.tcl discovery intermittently fails against the package-virtualized
# path when many Tk() roots are created/destroyed across a test session
# (each GUI test builds a real TTSApp). Pointing TCL_LIBRARY/TK_LIBRARY at
# the known-good path once avoids repeated, occasionally-flaky rediscovery.
_tcl_dir = os.path.join(sys.base_prefix, "tcl", "tcl8.6")
_tk_dir = os.path.join(sys.base_prefix, "tcl", "tk8.6")
if os.path.isdir(_tcl_dir):
    os.environ.setdefault("TCL_LIBRARY", _tcl_dir)
if os.path.isdir(_tk_dir):
    os.environ.setdefault("TK_LIBRARY", _tk_dir)

# One shared timestamp per pytest invocation, mirroring gui.py's
# self.timecode_format = "%Y%m%d%H%M%S" convention (gui.py:96).
_RUN_TS = time.strftime("%Y%m%d%H%M%S")


# ---------------------------------------------------------------------------
# Engine-level fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_dirs(tmp_path, monkeypatch):
    """Redirect kokoro_engine's module-level storage dirs into tmp_path."""
    custom_voices = tmp_path / "custom_voices"
    cache_dir = tmp_path / "cache"
    out_dir = tmp_path / "out"
    for d in (custom_voices, cache_dir, out_dir):
        d.mkdir()
    monkeypatch.setattr(kokoro_engine, "CUSTOM_VOICES_DIR", str(custom_voices))
    monkeypatch.setattr(kokoro_engine, "CACHE_DIR", str(cache_dir))
    return SimpleNamespace(custom_voices=custom_voices, cache_dir=cache_dir, out_dir=out_dir)


@pytest.fixture
def engine(isolated_dirs, monkeypatch):
    # Never touch the real audio device from a test.
    monkeypatch.setattr(kokoro_engine, "playback", MagicMock())
    e = KokoroEngine()
    yield e
    e.worker.stop()


@pytest.fixture
def real_engine(isolated_dirs, monkeypatch):
    """Real, unmocked KokoroEngine for tests/integration's opt-in real-pipeline
    tests. Identical to `engine` (isolated custom_voices/cache dirs, mocked
    playback so audio never touches the real audio device) but never
    combined with `fake_pipeline` - get_thread_pipeline/KPipeline resolve to
    the real kokoro.KPipeline, so synthesis actually runs torch + espeak-ng."""
    monkeypatch.setattr(kokoro_engine, "playback", MagicMock())
    e = KokoroEngine()
    yield e
    e.worker.stop()


class FakePipeline:
    """Mimics kokoro.KPipeline's calling convention without any model/espeak-ng."""

    def __init__(self, lang_code="a", segment_duration_s=0.05, sr=24000):
        self.lang_code = lang_code
        self.voices = {}
        self._sr = sr
        self._dur = segment_duration_s

    def __call__(self, text, voice=None, speed=1.0, split_pattern=r"\n+"):
        try:
            parts = [t.strip() for t in re.split(split_pattern, text) if t.strip()]
        except re.error:
            parts = []
        if not parts:
            parts = [text]
        n = max(1, int(self._sr * self._dur))
        for p in parts:
            audio = (0.1 * np.sin(2 * np.pi * 220 * np.arange(n) / self._sr)).astype(np.float32)
            yield p, "", audio

    def load_voice(self, name):
        return torch.zeros(510, 1, 256)


@pytest.fixture
def fake_pipeline(monkeypatch):
    fp = FakePipeline()
    monkeypatch.setattr(kokoro_engine, "get_thread_pipeline", lambda lang_code="a": fp)
    monkeypatch.setattr(kokoro_engine, "KPipeline", lambda lang_code="a": fp)
    return fp


@pytest.fixture
def callback_recorder(engine):
    rec = SimpleNamespace(statuses=[], progresses=[], finished=threading.Event())
    engine.on_status = lambda msg, is_err: rec.statuses.append((msg, is_err))
    engine.on_progress = lambda *a: rec.progresses.append(a)
    engine.on_finish = lambda: rec.finished.set()
    return rec


def wait_for_finish(rec, timeout=30):
    assert rec.finished.wait(timeout), "engine.on_finish was never called within timeout"


@pytest.fixture
def make_config(isolated_dirs):
    def _make(**overrides):
        cfg = {
            "voice": "af_heart",
            "speed": 1.0,
            "lang_code": "a",
            "split_pattern": r"\n+",
            "out_dir": str(isolated_dirs.out_dir),
            "filename": "output",
            "time_id": "0",
            "format": "wav",
            "num_threads": 1,
            "combine": True,
            "separate": True,
            "export_subtitles": False,
            "caching": False,  # hard default OFF - see module docstring
            "lexicon": {},
        }
        cfg.update(overrides)
        return cfg
    return _make


@pytest.fixture
def timestamped_output_dir(request):
    """
    tests/output/<run-timestamp>/<slugified-nodeid>/ - for tests whose generated
    audio should persist for manual inspection (real-pipeline integration tests,
    and a couple of "leaves_inspectable_output" smoke tests). Not used by
    throwaway unit tests, which use tmp_path/isolated_dirs instead.
    """
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", request.node.nodeid)
    d = Path(__file__).parent / "output" / _RUN_TS / slug
    d.mkdir(parents=True, exist_ok=True)
    return d


def espeak_available():
    # kokoro_engine never shells out to an `espeak-ng` CLI - phonemization
    # goes through misaki -> phonemizer's EspeakWrapper, pointed at the DLL
    # and data dir that the `espeakng_loader` package bundles/resolves
    # (see misaki/espeak.py). That's the actual runtime dependency, so
    # check for it directly instead of probing PATH for a binary the app
    # doesn't use.
    try:
        import espeakng_loader
        return (
            os.path.isfile(espeakng_loader.get_library_path())
            and os.path.isdir(espeakng_loader.get_data_path())
        )
    except Exception:
        return False


# ---------------------------------------------------------------------------
# GUI-level fixtures
# ---------------------------------------------------------------------------

class StubEngine:
    """Drop-in replacement for KokoroEngine used by GUI tests - never touches
    the real Kokoro pipeline/model."""

    def __init__(self):
        self.pipeline = object()  # truthy - passes the "engine still initializing" gate
        self.worker = SimpleNamespace(run_coro=MagicMock(return_value=concurrent.futures.Future()))
        self.cancel_event = threading.Event()
        self.on_progress = None
        self.on_status = None
        self.on_finish = None
        self.init_pipeline_async = MagicMock(return_value=None)
        self.start_conversion = MagicMock()
        self.start_jit_conversion = MagicMock()
        self.generate_preview = MagicMock()
        self.mix_voices = MagicMock()
        self.extract_text_from_file = MagicMock(return_value="")
        self.cancel = MagicMock()


@pytest.fixture
def tts_app(tmp_path, monkeypatch):
    import gui
    import tkinter

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gui, "CONFIG_FILE", str(tmp_path / "config.json"))
    monkeypatch.setattr(gui, "PRESETS_DIR", str(tmp_path / "presets"))
    monkeypatch.setattr(gui, "FX_PRESETS_DIR", str(tmp_path / "presets" / "fx"))
    monkeypatch.setattr(gui, "KokoroEngine", StubEngine)
    monkeypatch.setattr(gui, "messagebox", MagicMock())
    monkeypatch.setattr(gui, "filedialog", MagicMock())
    (tmp_path / "custom_voices").mkdir()

    # Creating many real Tk() interpreters across a test session intermittently
    # hits the same WindowsApps init.tcl read glitch as above - retry a few
    # times rather than failing the whole test on a transient hiccup.
    app = None
    last_err = None
    for _ in range(5):
        try:
            app = gui.TTSApp()
            break
        except tkinter.TclError as e:
            last_err = e
            time.sleep(0.2)
    if app is None:
        raise last_err

    yield app
    app.destroy()
