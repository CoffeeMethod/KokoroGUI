"""Time the GUI's hot paths on a synthetic project, headless.

    python scripts/perf_probe.py [CLIPS ...] [--details] [--profile]

Builds a `QtTTSApp` against `tests.conftest.StubEngine` (no model, no audio
device), fills the transcript with CLIPS lines (default 100 400 1500), one
clip of about 29 words per line cycling three characters, generates every
other clip with a 4 s tone per segment (with the onset and tail generation stores), and times: a keystroke in the middle
of the document (the synchronous part, then the deferred flush), a
transcript scroll tick, a timeline repaint while scrolling, a full
`refresh_timeline`, a full `rehighlight`, a project-scope reverb change
(the synchronous part, then the longest freeze in the next few seconds,
timeline scrolled to the middle) and one
`AssignCharacterCommand` and one `TextEditCommand` push. Each size runs in its own
process so one size's caches can't help the next. `--profile` prints a
cProfile of the keystroke and the scroll, and of the FX change. The numbers are what
`Claude/old/PLAN_performance.md` records per step.
"""
from __future__ import annotations

import os
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
SCROLL_STEPS = 60
FX_WINDOW_S = 6.0
COLUMNS = ("keystroke", "keystroke flush", "scroll tick", "timeline repaint", "refresh_timeline", "rehighlight",
           "fx change", "fx stall", "assign push", "text edit push")


def build(count: int, details: bool):
    import numpy as np
    import soundfile as sf

    import kokoro_engine
    import kokoro_gui.qt.app as qt_app_module
    from PySide6.QtWidgets import QApplication
    from tests.conftest import StubEngine

    from kokoro_gui.daw.dirty import build_segments_from_results, compute_expected_cache_hash, predict_segment_texts, spoken_text
    from kokoro_gui.daw.models import DEFAULT_HIGHLIGHT_PALETTE, Character, Track
    from kokoro_gui.qt import settings as qt_settings
    from kokoro_gui.qt.docks import video_dock

    workdir = tempfile.mkdtemp(prefix="kokorogui_perf_")
    os.chdir(workdir)
    qt_app_module.CONFIG_FILE = os.path.join(workdir, "config_qt.json")
    qt_app_module.PRESETS_DIR = os.path.join(workdir, "presets")
    qt_app_module.FX_PRESETS_DIR = os.path.join(workdir, "presets", "fx")
    qt_app_module.DOCUMENT_FILE = os.path.join(workdir, "document.json")
    kokoro_engine.KokoroEngine = StubEngine
    video_dock.PLAYER_ENABLED = False
    os.makedirs(qt_app_module.FX_PRESETS_DIR, exist_ok=True)
    qapp = QApplication.instance() or QApplication(sys.argv)
    settings = qt_settings.load_settings(qt_app_module.CONFIG_FILE)
    settings["transcript_details"] = details
    qt_settings.save_settings(qt_app_module.CONFIG_FILE, settings)

    app = qt_app_module.QtTTSApp()
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
    audio_dir = os.path.join(workdir, "audio")
    os.makedirs(audio_dir)
    tone = (0.3 * np.sin(np.linspace(0, 2000, 24000 * 4))).astype(np.float32)
    for i, clip in enumerate(clips):
        if i % 2:
            continue
        clip_text = doc.clip_text(clip)
        config = app._assemble_clip_config(clip)
        expected = compute_expected_cache_hash(clip_text, config)
        results = []
        for j, piece in enumerate(predict_segment_texts(spoken_text(clip_text, config), config)):
            path = os.path.join(audio_dir, f"{clip.id}_{j}.wav")
            sf.write(path, tone, 24000)
            results.append({"text": piece, "path": path, "duration": 4.0, "onset_s": 0.0, "tail_s": 0.0})
        clip.segments = build_segments_from_results(expected, results)
        clip.status = "generated"
    app.resize(1600, 1000)
    app.workspaces.apply_default("Advanced")
    app.show()
    app.editor.rehighlight()
    app.refresh_timeline()
    settle(qapp)
    return qapp, app


def settle(qapp, seconds: float = 0.25) -> None:
    """Runs the event loop until the debounce timers have fired."""
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        qapp.processEvents()
        time.sleep(0.005)
    qapp.processEvents()


def longest_stall(qapp, seconds: float) -> float:
    """The longest single `processEvents` call, in ms, over `seconds` of
    running the event loop: how long the GUI stops answering at worst. After
    an FX change that is the 100 ms schedule timer's rebuild, the waveform
    repaints and, with a render pool, the results landing."""
    worst = 0.0
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        start = time.perf_counter()
        qapp.processEvents()
        worst = max(worst, (time.perf_counter() - start) * 1000)
        time.sleep(0.002)
    return worst


def ms(fn, reps: int = 1) -> float:
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - start) * 1000 / reps


def measure(count: int, details: bool, profile: bool) -> dict:
    import cProfile
    import io
    import pstats

    from kokoro_gui.daw.undo import AssignCharacterCommand, TextEditCommand
    from kokoro_gui.qt.transcript_editor import TranscriptGutter

    qapp, app = build(count, details)
    editor = app.editor
    gutter = editor.findChild(TranscriptGutter)
    results = {}
    profiler = cProfile.Profile() if profile else None

    def keystroke():
        cursor = editor.textCursor()
        cursor.setPosition(len(SENTENCE) * (count // 2) + 10)
        editor.setTextCursor(cursor)
        editor.insertPlainText("x")
        qapp.processEvents()

    keystroke()
    settle(qapp)
    if profiler:
        profiler.enable()
    results["keystroke"] = ms(keystroke)
    start = time.perf_counter()
    settle(qapp)
    results["keystroke flush"] = (time.perf_counter() - start - 0.25) * 1000

    scrollbar = editor.verticalScrollBar()

    def scroll():
        top = scrollbar.maximum()
        for k in range(SCROLL_STEPS):
            scrollbar.setValue(int(top * k / SCROLL_STEPS))
            gutter.repaint()
            editor.viewport().repaint()
            qapp.processEvents()

    results["scroll tick"] = ms(scroll) / SCROLL_STEPS
    if profiler:
        profiler.disable()
        out = io.StringIO()
        pstats.Stats(profiler, stream=out).sort_stats("cumulative").print_stats(30)
        print(out.getvalue())

    view = app.timeline_dock.timeline_view
    view = getattr(view, "timeline_view", view)
    hbar = view.horizontalScrollBar()

    def timeline_scroll():
        top = hbar.maximum()
        for k in range(SCROLL_STEPS):
            hbar.setValue(int(top * k / SCROLL_STEPS))
            view.viewport().repaint()
            qapp.processEvents()

    results["timeline repaint"] = ms(timeline_scroll) / SCROLL_STEPS

    def refresh():
        app.refresh_timeline()
        flush = getattr(app, "flush_updates", None)
        if flush is not None:
            flush()
        qapp.processEvents()

    results["refresh_timeline"] = ms(refresh)
    results["rehighlight"] = ms(editor.rehighlight)

    # A project-scope reverb change with the timeline scrolled to the middle:
    # from the FX edit to the first moment the GUI answers events again.
    hbar.setValue(hbar.maximum() // 2)
    app.selection.clear()  # the keystroke selected a clip; the FX tab edits project scope with none
    qapp.processEvents()

    def fx_change():
        app.fx_dock._enabled_checks["reverb_enabled"].setChecked(True)
        app.fx_dock._project_timer.stop()
        app._save_timer.stop()  # the autosave of the whole document is its own cost
        app.refresh_timeline()
        view.viewport().repaint()
        qapp.processEvents()

    fx_profiler = cProfile.Profile() if profile else None
    if fx_profiler:
        fx_profiler.enable()
    results["fx change"] = ms(fx_change)
    results["fx stall"] = longest_stall(qapp, FX_WINDOW_S)
    if fx_profiler:
        fx_profiler.disable()
        out = io.StringIO()
        pstats.Stats(fx_profiler, stream=out).sort_stats("cumulative").print_stats(35)
        print(out.getvalue())

    def undo_push(command):
        start = time.perf_counter()
        doc.undo_stack.push(command)
        return (time.perf_counter() - start) * 1000

    doc = app.document
    mid = len(SENTENCE) * (count // 2)
    results["assign push"] = undo_push(AssignCharacterCommand(mid + 5, mid + 20, doc.characters[1].id))
    results["text edit push"] = undo_push(TextEditCommand(mid + 3, 0, 1, doc.text[:mid + 3] + "y" + doc.text[mid + 3:]))
    app._ask_close_choice = lambda: "discard"
    return results


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    details = "--details" in sys.argv
    profile = "--profile" in sys.argv
    if os.environ.get("KOKOROGUI_PERF_CHILD"):
        results = measure(int(args[0]), details, profile)
        print("RESULT " + " ".join(f"{results[c]:.1f}" for c in COLUMNS), flush=True)
        os._exit(0)
    sizes = [int(a) for a in args] or [100, 400, 1500]
    print(f"{'clips':>6} " + " ".join(f"{c:>17}" for c in COLUMNS) + "   (ms)")
    for size in sizes:
        env = dict(os.environ, KOKOROGUI_PERF_CHILD="1")
        flags = [f for f in ("--details", "--profile") if f in sys.argv]
        proc = subprocess.run([sys.executable, "-u", os.path.abspath(__file__), str(size), *flags],
                              env=env, capture_output=True, text=True)
        line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
        if profile:
            print(proc.stdout.replace(line or "", ""))
        if line is None:
            print(f"{size:>6} failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
            continue
        values = line.split()[1:]
        print(f"{size:>6} " + " ".join(f"{v:>17}" for v in values))


if __name__ == "__main__":
    main()
