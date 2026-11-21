"""Time Open, layout, the transport load, memory, Save and a project-scope FX
change on a book-sized project, headless.

    python scripts/bench_book.py [--clips N] [--json] [--keep]

Phase 6 section 0 of the plan decides from these numbers whether the streaming
transport, lazy extraction and an append-only Save get built. Measurement only.

Two child processes, so the Open runs cold:

1. build: a `QtTTSApp` over `tests.conftest.StubEngine` (no model, no audio
   device) gets N clips of one sentence each. Every clip has a real 24 kHz wav
   segment, a `DummyEngine` tone, copied from one written file. The project is
   saved as a `.tbaw`.
2. measure: a fresh app opens that file and is timed. The transport's stream
   factory is a fake, so no audio device is ever opened.

Everything lives in one temp dir (config, presets, characters, templates, the
cache and the bundle), removed at the end, and the storage roots are
redirected the way `tests/conftest.py`'s `isolated_dirs` does. 10,000 clips
need about 8 GB of free disk at the peak.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32":
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SENTENCE = ("She pushed the door open, and somewhere inside a radio was playing an old song "
            "about the sea. The kettle was still warm when she reached the kitchen.")
RATE = 24000
WAIT_S = 3600.0
ROWS = (
    ("open_total", "Open, total"),
    ("open_extract_small", "  extract_small"),
    ("open_extract_audio", "  extract_audio"),
    ("open_finish_open", "  project_io.finish_open"),
    ("open_switch_document", "  _switch_document"),
    ("open_settle", "  first schedule rebuild after Open"),
    ("build_arrangement", "build_arrangement (memos cleared)"),
    ("transport_load", "Transport.load (first, after Open)"),
    ("rss_after_load_mb", "RSS after load (MB)"),
    ("save_total", "Save, nothing new"),
    ("save_plan", "  plan_save"),
    ("save_write", "  write_bundle"),
    ("fx_change", "Project-scope FX change to first audio"),
    ("fx_transport_load", "  Transport.load in it"),
    ("rss_after_fx_mb", "RSS after the FX change (MB)"),
)


def rss_bytes() -> tuple[int | None, str]:
    """Resident set size and where it came from. psutil when installed, else
    the platform's own counter."""
    try:
        import psutil
        return psutil.Process().memory_info().rss, "psutil"
    except ImportError:
        pass
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]

            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return int(counters.WorkingSetSize), "GetProcessMemoryInfo"
        elif os.path.isfile("/proc/self/statm"):
            with open("/proc/self/statm") as f:
                return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE"), "/proc/self/statm"
        else:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return int(peak), "ru_maxrss (peak, bytes on macOS)"
    except Exception:  # noqa: BLE001 - memory is optional
        pass
    return None, "unavailable (pip install psutil)"


def redirect_storage(base: str, cache: str) -> None:
    """Everything the app writes goes under `base`, as `isolated_dirs` and
    `qt_app` do in the test suite."""
    import kokoro_engine
    import kokoro_gui.qt.app as qt_app_module
    from kokoro_gui.daw import library as library_module
    from kokoro_gui.daw import templates as templates_module
    from kokoro_gui.engine import runtime
    from kokoro_gui.qt.docks import video_dock
    from tests.conftest import StubEngine

    os.makedirs(base, exist_ok=True)
    os.chdir(base)
    for name in ("custom_voices", "out"):
        os.makedirs(os.path.join(base, name), exist_ok=True)
    os.makedirs(cache, exist_ok=True)
    runtime.CUSTOM_VOICES_DIR = os.path.join(base, "custom_voices")
    runtime.CACHE_DIR = cache
    runtime.STATS_FILE = os.path.join(base, "generation_stats.json")
    library_module.LIBRARY_DIR = os.path.join(base, "characters")
    templates_module.TEMPLATES_DIR = os.path.join(base, "templates")
    qt_app_module.CONFIG_FILE = os.path.join(base, "config_qt.json")
    qt_app_module.PRESETS_DIR = os.path.join(base, "presets")
    qt_app_module.FX_PRESETS_DIR = os.path.join(base, "presets", "fx")
    qt_app_module.DOCUMENT_FILE = os.path.join(base, "document.json")
    os.makedirs(qt_app_module.FX_PRESETS_DIR, exist_ok=True)
    kokoro_engine.KokoroEngine = StubEngine
    video_dock.PLAYER_ENABLED = False


def make_app():
    from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

    import kokoro_gui.qt.app as qt_app_module

    # A dialog would block forever on the offscreen platform.
    for name in ("information", "critical", "warning"):
        setattr(QMessageBox, name, staticmethod(lambda *a, **k: None))
    QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: ("", ""))
    qt_app_module.QtTTSApp._ask_close_choice = lambda self: "discard"
    qapp = QApplication.instance() or QApplication(sys.argv)
    app = qt_app_module.QtTTSApp()
    app.resize(1600, 1000)
    app.workspaces.apply_default("Advanced")
    app.show()
    return qapp, app


def spin_until(qapp, predicate, timeout_s: float = WAIT_S) -> bool:
    end = time.perf_counter() + timeout_s
    while time.perf_counter() < end:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.002)
    return False


def settle(qapp, seconds: float = 0.3) -> None:
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        qapp.processEvents()
        time.sleep(0.005)


def build(workdir: str, bundle: str, count: int) -> None:
    import shutil as sh

    import soundfile as sf

    from kokoro_gui.daw.dirty import (build_segments_from_results, compute_expected_cache_hash, predict_segment_texts,
                                      spoken_text)
    from kokoro_gui.daw.models import DEFAULT_HIGHLIGHT_PALETTE, Character, Track
    from kokoro_gui.engines.dummy import _tone_for

    redirect_storage(workdir, os.path.join(workdir, "cache"))
    qapp, app = make_app()
    doc = app.document
    doc.characters = [Character.from_preset_dict(name, {"voice": voice}, highlight_color=DEFAULT_HIGHLIGHT_PALETTE[i])
                      for i, (name, voice) in enumerate([("Narrator", "af_heart"), ("Tomas", "am_michael"),
                                                         ("Marta", "bf_emma")])]
    doc.tracks = [Track(name=c.name, character_id=c.id, order_index=i) for i, c in enumerate(doc.characters)]
    text = "\n".join(SENTENCE for _ in range(count))
    app.editor.load_text(text)
    doc.set_plain_text(text)
    clips = []
    pos = 0
    for i in range(count):
        clips.append(doc.assign_character_to_range(pos, pos + len(SENTENCE), doc.characters[i % 3].id))
        pos += len(SENTENCE) + 1
    audio_dir = os.path.join(app.root.project_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    template = os.path.join(workdir, "tone.wav")
    sf.write(template, _tone_for(SENTENCE, 1.0, "af_heart"), RATE)
    duration = os.path.getsize(template) / (RATE * 2.0)
    for clip in clips:
        clip_text = doc.clip_text(clip)
        config = app._assemble_clip_config(clip)
        expected = compute_expected_cache_hash(clip_text, config)
        results = []
        for j, piece in enumerate(predict_segment_texts(spoken_text(clip_text, config), config)):
            path = os.path.join(audio_dir, f"{clip.id}_{j}.wav")
            sh.copyfile(template, path)
            results.append({"text": piece, "path": path, "duration": duration, "onset_s": 0.0, "tail_s": 0.0})
        clip.segments = build_segments_from_results(expected, results)
        clip.status = "generated"
    app.refresh_timeline()
    settle(qapp)
    app.save_project_as(bundle)
    app.wait_for_project_io(WAIT_S)
    if not os.path.isfile(bundle):
        raise SystemExit("build: the bundle was not written")
    print("BUILT " + json.dumps({"bytes": os.path.getsize(bundle), "clips": count}), flush=True)
    os._exit(0)


class FakeStream:
    """Stands in for the transport's audio stream: nothing opens a device."""

    def __init__(self, *_a, **_k):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass


def timed(owner, name, sink, key):
    """Wraps `owner.name` so each call's seconds are added to `sink[key]`."""
    original = getattr(owner, name)

    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            sink[key] = sink.get(key, 0.0) + time.perf_counter() - start

    setattr(owner, name, wrapper)


def measure(workdir: str, bundle: str) -> None:
    redirect_storage(workdir, os.path.join(workdir, "cache"))
    import kokoro_gui.qt.project as project_io
    from kokoro_gui.audio import post
    from kokoro_gui.audio.transport import Transport

    qapp, app = make_app()
    app.transport._stream_factory = lambda *_a, **_k: FakeStream()
    out: dict = {}
    loads: list = []
    original_load = Transport.load

    def load(self, *args, **kwargs):
        start = time.perf_counter()
        try:
            return original_load(self, *args, **kwargs)
        finally:
            loads.append((start, time.perf_counter()))

    app.transport.load = load.__get__(app.transport)

    for name, key in (("extract_small", "open_extract_small"), ("extract_audio", "open_extract_audio"),
                      ("finish_open", "open_finish_open"), ("plan_save", "save_plan"),
                      ("write_bundle", "save_write")):
        timed(project_io, name, out, key)
    timed(app, "_switch_document", out, "open_switch_document")

    # Open: from the call to the editor writable again, then until the
    # debounced schedule rebuild has loaded the transport.
    loads.clear()
    start = time.perf_counter()
    app.open_project(bundle)
    app.wait_for_project_io(WAIT_S)
    out["open_total"] = time.perf_counter() - start
    if app.project_path != os.path.abspath(bundle):
        raise SystemExit("measure: the project did not open")
    if not spin_until(qapp, lambda: bool(loads) and not post.busy() and not app._schedule_timer.isActive()):
        raise SystemExit("measure: the schedule never loaded")
    out["open_settle"] = time.perf_counter() - start - out["open_total"]
    out["transport_load"] = loads[0][1] - loads[0][0]
    out["clips"] = len(app.document.clips)
    out["scheduled_clips"] = len(app.transport.loaded_clips())
    rss, source = rss_bytes()
    out["rss_after_load_mb"] = None if rss is None else rss / 2 ** 20
    out["rss_source"] = source

    app._arrangement_memo.clear()
    app._duration_memo.clear()
    start = time.perf_counter()
    app.build_arrangement()
    out["build_arrangement"] = time.perf_counter() - start

    # Save with nothing new: the whole bundle is written again.
    settle(qapp)
    start = time.perf_counter()
    app.save_project()
    app.wait_for_project_io(WAIT_S)
    out["save_total"] = time.perf_counter() - start
    settle(qapp)

    # A project-scope FX change, with no clip selected, as the FX tab edits it.
    app.selection.clear()
    qapp.processEvents()
    loads.clear()
    start = time.perf_counter()
    app.fx_dock._enabled_checks["reverb_enabled"].setChecked(True)
    app.fx_dock._project_timer.stop()
    app._save_timer.stop()  # the autosave of the whole document is its own cost
    app.refresh_timeline()
    app.timeline_dock.timeline_view.viewport().repaint()
    if not spin_until(qapp, lambda: bool(loads) and not post.busy() and not app._schedule_timer.isActive()):
        raise SystemExit("measure: the FX change never reached the transport")
    out["fx_change"] = loads[-1][1] - start
    out["fx_transport_load"] = loads[-1][1] - loads[-1][0]
    rss, _source = rss_bytes()
    out["rss_after_fx_mb"] = None if rss is None else rss / 2 ** 20
    print("RESULT " + json.dumps(out), flush=True)
    os._exit(0)


def run_child(stage: str, args: list, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ, KOKOROGUI_BENCH_STAGE=stage, **(env_extra or {}))
    return subprocess.run([sys.executable, "-u", os.path.abspath(__file__), *args], env=env,
                          capture_output=True, text=True)


def machine() -> dict:
    info = {"cpu_cores": os.cpu_count(), "os": sys.platform, "python": sys.version.split()[0]}
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 2 ** 30, 1)
    except ImportError:
        if sys.platform == "win32":
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("avail", ctypes.c_ulonglong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                            ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ae", ctypes.c_ulonglong)]

            status = Status()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                info["ram_gb"] = round(status.total / 2 ** 30, 1)
    return info


def print_table(result: dict) -> None:
    machine_info = result["machine"]
    print(f"clips: {result['clips']}   cores: {machine_info['cpu_cores']}   RAM: {machine_info.get('ram_gb', '?')} GB   "
          f"os: {machine_info['os']}   rss source: {result.get('rss_source')}")
    print(f"scheduled clips: {result.get('scheduled_clips')}   bundle: {result['bundle_mb']:.0f} MB")
    for key, label in ROWS:
        value = result.get(key)
        if value is None:
            cell = "n/a"
        elif key.startswith("rss"):
            cell = f"{value:,.0f}"
        else:
            cell = f"{value:,.2f} s"
        print(f"{label:<42}{cell:>14}")


def main() -> None:
    stage = os.environ.get("KOKOROGUI_BENCH_STAGE")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--clips", type=int, default=3000)
    parser.add_argument("--json", action="store_true", help="print the numbers as JSON instead of a table")
    parser.add_argument("--keep", action="store_true", help="keep the temp dir and print its path")
    parser.add_argument("--dir", help=argparse.SUPPRESS)
    parser.add_argument("--bundle", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if stage == "build":
        build(args.dir, args.bundle, args.clips)
    if stage == "measure":
        measure(args.dir, args.bundle)

    root = tempfile.mkdtemp(prefix="kokorogui_bench_")
    bundle = os.path.join(root, "book.tbaw")
    try:
        started = time.perf_counter()
        for name in ("build", "measure"):
            workdir = os.path.join(root, name)
            proc = run_child(name, ["--clips", str(args.clips), "--dir", workdir, "--bundle", bundle])
            line = next((ln for ln in proc.stdout.splitlines() if ln.startswith(("BUILT ", "RESULT "))), None)
            if line is None:
                sys.exit(f"{name} failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-3000:]}")
            if name == "build":
                print(f"built {args.clips} clips in {time.perf_counter() - started:.0f} s", file=sys.stderr)
                shutil.rmtree(workdir, ignore_errors=True)  # the build's extracted copy: Open must start cold
            else:
                result = json.loads(line.split(" ", 1)[1])
        result["machine"] = machine()
        result["bundle_mb"] = os.path.getsize(bundle) / 2 ** 20
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print_table(result)
    finally:
        if args.keep:
            print(f"kept {root}", file=sys.stderr)
        else:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
