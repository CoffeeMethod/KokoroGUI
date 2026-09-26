"""Render the Qt shell to a PNG without a display or a model.

    python scripts/render_screenshot.py out.png [--theme dark] [--workspace Simple]
                                                 [--size 1600x1000] [--subproject]

Builds a `QtTTSApp` against `tests.conftest.StubEngine` (no Kokoro, no
eSpeak, no audio device) in a temp working directory, loads a small sample
project with three characters, marks two clips as generated with synthetic
audio so the timeline shows waveforms next to estimated clips, and grabs
the window. `--subproject` turns the last line into a subproject (phase 4),
so the transcript shows its placeholder line and the timeline its block and
breadcrumb. Used to compare each step of Claude/PLAN_ui_shell_redesign.md
against the wireframe, and to refresh the README/docs screenshots.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32":
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

# The same short radio drama the docs site's home page shows.
SAMPLE_TEXT = (
    "The house at the end of the lane had been empty for years.\n"
    "She pushed the door. It didn't creak, and somewhere inside a radio was playing.\n"
    "Marta, come in. Marta, are you there?\n"
    "I'm in the kitchen. The light is on, and the kettle is still warm."
)


def _write_tone(path: str, seconds: float, freq: float, rate: int = 24000) -> None:
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t)
    data = (0.4 * np.sin(2 * np.pi * freq * t) * env).astype(np.float32)
    sf.write(path, data, rate)


def build_app(workdir: str, theme_name: str, workspace: str, subproject: bool = False):
    from tests.conftest import StubEngine  # noqa: E402

    import kokoro_engine
    import kokoro_gui.qt.app as qt_app_module  # noqa: E402
    from PySide6.QtWidgets import QApplication  # noqa: E402

    from kokoro_gui.daw.dirty import build_segments_from_results, compute_expected_cache_hash  # noqa: E402
    from kokoro_gui.daw.models import DEFAULT_HIGHLIGHT_PALETTE, Character, Track  # noqa: E402

    os.chdir(workdir)
    qt_app_module.CONFIG_FILE = os.path.join(workdir, "config_qt.json")
    qt_app_module.PRESETS_DIR = os.path.join(workdir, "presets")
    qt_app_module.FX_PRESETS_DIR = os.path.join(workdir, "presets", "fx")
    qt_app_module.DOCUMENT_FILE = os.path.join(workdir, "document.json")
    kokoro_engine.KokoroEngine = StubEngine
    # The video dock stays a stub: no QtMultimedia in a headless run.
    from kokoro_gui.qt.docks import video_dock  # noqa: E402

    video_dock.PLAYER_ENABLED = False
    os.makedirs(qt_app_module.FX_PRESETS_DIR, exist_ok=True)
    with open(os.path.join(qt_app_module.FX_PRESETS_DIR, "Echo.json"), "w", encoding="utf-8") as f:
        f.write('{"delay_enabled": true, "delay_time": 0.3, "delay_feedback": 0.3, "delay_mix": 0.4}')

    qapp = QApplication.instance() or QApplication(sys.argv)
    from kokoro_gui.qt import settings as qt_settings  # noqa: E402

    settings = qt_settings.load_settings(qt_app_module.CONFIG_FILE)
    settings["theme"] = theme_name
    settings["active_workspace"] = workspace
    qt_settings.save_settings(qt_app_module.CONFIG_FILE, settings)

    app = qt_app_module.QtTTSApp()
    doc = app.document
    doc.characters = [
        Character.from_preset_dict("Narrator", {"voice": "af_heart"}, highlight_color=DEFAULT_HIGHLIGHT_PALETTE[0]),
        Character.from_preset_dict("Tomas", {"voice": "am_michael", "fx_preset": "Echo"},
                                   highlight_color=DEFAULT_HIGHLIGHT_PALETTE[1]),
        Character.from_preset_dict("Marta", {"voice": "bf_emma"}, highlight_color=DEFAULT_HIGHLIGHT_PALETTE[2]),
    ]
    doc.tracks = [Track(name=c.name, character_id=c.id, order_index=i) for i, c in enumerate(doc.characters)]
    narrator, tomas, marta = doc.characters

    app.editor.load_text(SAMPLE_TEXT)
    doc.set_plain_text(SAMPLE_TEXT)
    lines = SAMPLE_TEXT.split("\n")
    offsets = []
    pos = 0
    for line in lines:
        offsets.append((pos, pos + len(line)))
        pos += len(line) + 1
    c1 = doc.assign_character_to_range(offsets[0][0], offsets[1][1], narrator.id)
    c2 = doc.assign_character_to_range(offsets[2][0], offsets[2][1], tomas.id)
    c3 = doc.assign_character_to_range(offsets[3][0], offsets[3][1], marta.id)

    # Two generated clips (waveforms), one still estimated (dashed).
    audio_dir = os.path.join(workdir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    for clip, seconds, freq in ((c1, 6.5, 220.0), (c2, 3.2, 330.0)):
        path = os.path.join(audio_dir, f"{clip.id}.wav")
        _write_tone(path, seconds, freq)
        text = doc.clip_text(clip)
        config = app._assemble_clip_config(clip)
        expected = compute_expected_cache_hash(text, config)
        clip.segments = build_segments_from_results(expected, [{"text": text, "path": path, "duration": seconds}])
    if subproject:
        start, end = doc.clip_extent(c3.id)
        app.new_subproject(start, end, title="Chapter 2")
        app.editor.load_text(doc.text)
    del c3

    app.editor.rehighlight()
    app.transcript_dock.refresh_character_choices()
    app.refresh_timeline()
    app._rebuild_transport_schedule()
    app.transport.seek(4.2)
    app._on_transport_position(4.2)
    return qapp, app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--theme", default="light", choices=("light", "dark"))
    parser.add_argument("--workspace", default="Advanced", choices=("Advanced", "Simple"))
    parser.add_argument("--size", default="1600x1000")
    parser.add_argument("--subproject", action="store_true", help="show a subproject line, block and breadcrumb")
    args = parser.parse_args()
    width, height = (int(v) for v in args.size.lower().split("x"))

    out = os.path.abspath(args.out)
    workdir = tempfile.mkdtemp(prefix="kokorogui_shot_")
    qapp, app = build_app(workdir, args.theme, args.workspace, args.subproject)
    app.resize(width, height)
    app.workspaces.apply_default(args.workspace)
    app.show()
    for _ in range(3):  # let the deferred showEvent proportion pass run
        qapp.processEvents()
    app.timeline_dock.timeline_view.set_playhead(4.2)
    qapp.processEvents()
    app.grab().save(out)
    print(out)
    # The sample project is Untitled and edited: closing would ask Save /
    # Discard / Cancel (grill TB12), which a headless run can't answer.
    app._ask_close_choice = lambda: "discard"
    app.close()


if __name__ == "__main__":
    main()
