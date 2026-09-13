# Kokoro TTS GUI

[![Tests](https://github.com/CoffeeMethod/KokoroGUI/actions/workflows/tests.yml/badge.svg)](https://github.com/CoffeeMethod/KokoroGUI/actions/workflows/tests.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

A desktop text-to-speech app built with Python: a dockable PySide6 (Qt) interface over a pluggable
synthesis backend, edited less like a form and more like a DAW project, with a document of text,
timed clips on character tracks, and undo/redo. Powered by [Kokoro](https://github.com/hexgrad/kokoro)
by default, with a zero-shot voice-cloning backend also built in.

<img width="1600" height="1000" alt="KokoroGUI 4.0: transcript and settings tabs over a seconds-axis timeline and transport, dark theme" src="docs/assets/shell_dark.png" />

*(demo sounds better in `.wav` but GitHub doesn't support that so it's kinda bad)*

https://github.com/user-attachments/assets/c75e7141-5d73-40f4-b182-d4f5bc49ad1e

## New in Beta 4.0.0

The rebuild. 3.2.0 was a CustomTkinter form over one `kokoro_engine.py`; 4.0.0 is a PySide6
shell over a document with clips, tracks and characters, a pluggable engine layer with a second
real backend, and projects that live in one file.

-   **Qt frontend, the only frontend.** `python main.py`/`run.bat` launches a PySide6 shell of
    dockable panels (`kokoro_gui/qt/`) in a 2x2 grid: Transcript | Settings / Audio FX / Lexicon /
    Voices tabs on top, Timeline | Transport underneath, under File / Edit / Options / Workspace
    menus. Every panel is a dock you can drag; **Workspace > Advanced / Simple / Reset layout** are
    saved layouts (Simple hides the timeline and gives the transcript the full height). The
    CustomTkinter app (`gui.py`) is gone; PySide6 is a regular dependency in `requirements.txt`.
-   **A document, not a text box.** The old "generate this text" input is a project: a `Document`
    of canonical text with `Clip`/`Track`/`Character` metadata layered on top (`kokoro_gui/daw/`).
    The transcript is the source of truth and generated audio is a render of it, tracked per clip
    with hash-based dirty detection. Generate regenerates every out-of-date clip in one pass with
    bounded concurrency instead of the whole document every time; a document with no clips yet
    still uses the whole-document pipeline. Auto-split turns a `[Speaker:FX]`-tagged document
    (optionally per paragraph) into clips and generates them in one action.
-   **Transcript panel with character highlighting and a live gutter.** Each run is tinted by its
    character, so speaker boundaries are visible without reading the inline `[Speaker:FX]:`
    syntax, which converts into a real assignment the moment you finish a tagged line. Above the
    editor sit two combos, Character and FX, that reflect the caret's clip and reassign the
    selection (or the whole clip). The gutter labels once per character/FX change (`Narrator` /
    `FX: Echo`) and shows a play button beside each out-of-date clip; click it to regenerate just
    that clip. Out-of-date text is dash-underlined, thin rules show where clips end and where
    Auto-split would cut. Copy/paste carries the character assignment along (with a setting for
    whether a paste splits off its own run or inherits the destination's).
-   **A multi-track timeline on a real seconds axis.** One lane per character; clips sit end to end
    in text order at their real duration once generated and an estimated one (dashed outline, no
    waveform) before, learned from `generation_stats.json`. A ruler with a playhead, a fixed
    track-header column, Ctrl+wheel zoom. Dragging a clip pins it to a time or moves it to another
    character's track; dragging it before an earlier clip also moves its text there. Shift+drag
    inside a clip carves out a sub-range and replaces it with fresh TTS under any character. Each
    clip has its own FX button for overrides to its character's preset.
-   **Playback.** Play / pause / stop, click the ruler to seek, a playhead across all lanes, and
    the transcript highlights and scrolls to the clip being played. Space toggles playback
    anywhere but the text editor; Ctrl+Space toggles everywhere. Built on one
    `sounddevice.OutputStream` that mixes the arrangement in the callback
    (`kokoro_gui/audio/transport.py`), so the position is sample accurate. Loop toggle included.
-   **Export.** File > Export mixes every clip down to one file (wav/mp3/flac/ogg) at its timeline
    position, optionally with a `.srt` and per-clip files (`<base>_001_Narrator.wav`). It warns
    when clips are out of date and offers to generate first. The dialog also holds the project's
    two bundle options: whether to bundle generated audio, and the audio format for new segments
    (wav or flac).
-   **`.tbaw` project bundles.** A project is one zip file that carries everything: the text and
    clips, every generated segment, and every named voice mix, voice reference and FX preset it
    uses, so it opens on another machine with the same engines installed. Save writes the whole
    file in the background (the progress line shows it) and never leaves a half-written project
    behind; Save As keeps the same working copy. Autosave writes only into the app's own working
    copy (`cache/projects/`), so the file on disk is as new as your last Save. Closing with
    unsaved changes asks Save / Discard / Cancel, and a crash offers to recover the unsaved
    session next time the project opens, even after the file was renamed or moved. Launch reopens
    the last project; New inherits the previous project's characters; Import Text asks whether to
    add to the current project or start a new one. The window title names the project and shows
    `*` while it has unsaved changes. A bundle only ever names audio inside itself: a
    `document.json` pointing at some other file on the machine reads as a missing segment, and
    Save never copies a file from outside the project's working copy into the bundle. (`.json`
    projects from the 4.0 previews still open and are converted on the spot, a `.tbaw` written
    next to the untouched `.json`, with matching audio carried over.)
-   **Generation writes once.** A clip's audio lands straight in the project's working copy under
    a name derived from what produced it, instead of one copy in `cache/` and another in the
    output folder. Regenerating a clip that's already up to date (the gutter button) makes a fresh
    take under a new name and leaves the old file for any other identical clip that plays it.
    Opening a project made with another version of an engine keeps its clips clean and says so in
    the status line; only clips you regenerate use the installed version.
-   **Welcome screen.** Launch opens the last project, then puts a dialog over it: recent projects
    (the open one first, Resume as the default button), New project, New from text file, Open
    other, right-click to drop a row, Clear list, and a details pane with the file's path,
    modified time, character and clip counts, audio length and engines. Untick "Show at startup"
    for a silent resume; File > Welcome... brings it up any time.
-   **Undo/redo.** Edit > Undo/Redo over a plain-Python undo stack. Typing undoes like a normal
    text editor; character/FX assignments and timeline moves have their own history, and Ctrl+Z
    reverts whichever happened most recently.
-   **Settings and Audio FX follow the selection.** The old always-global Generation fields
    (voice, speed, split pattern, plus volume/pitch/normalize/trim) live in a Settings dock that
    reads and writes whatever's selected: the whole document's defaults, one clip's overrides, or
    a character's preset. Editing a character affects every clip using it unless that clip has
    its own override. Audio FX works the same way (project defaults, a character's preset with a
    prompt before changing one that several clips share, or a clip override where slider drags
    become one undoable step); the timeline's FX button selects the clip and raises the tab.
-   **Audio FX are non-destructive.** Clips are generated as raw model output and the FX chain,
    volume, pitch, normalize and trim are applied when the transport, the export or the timeline
    waveform reads them. Move a slider, pick a preset, toggle "Apply": you hear it on the next
    play, the clip stays generated, nothing is marked out of date. The Audio FX tab and playback
    resolve a clip's stack through one function (`kokoro_gui/qt/fx_resolve.py`), and a clip with
    its own FX override counts as FX-on even if its character's preset says off.
-   **Characters replace bare presets.** Existing `presets/*.json` files migrate into `Character`
    objects on first load (one per file, or a single "Default" character seeded from your last
    settings if you had none), each with its own highlight color; Edit > Characters... edits
    name, color, voice and FX preset. The preset files themselves are untouched, so this is a safe
    downgrade path. Output folder, filename and format moved from the Settings tab to Export.
-   **Dark and light themes** (Options > Theme, dark by default). One palette module feeds the
    custom-painted widgets, the Qt palette and a stylesheet that styles every control: borderless
    setting groups, rounded inputs and buttons, an underlined tab strip, thin scrollbars, a flat
    progress line, painted play / pause / stop glyphs and a filled Generate button. The UI font
    is Segoe UI / Inter / Noto Sans at 10pt, the transcript one point larger. Timeline clips have
    rounded corners, a waveform in the clip's own darker shade and a label in black or white by
    contrast; the default character palette is eight hues at one lightness and doubles as the
    color picker's presets.
-   **Pluggable engines.** `kokoro_gui/engines/` defines a backend interface (config schema,
    voices, capabilities, project hooks) with three registered backends: Kokoro, Audio8 and a
    sine-tone Dummy that exists to prove the abstraction isn't Kokoro-shaped. Options > Engine
    swaps the Settings tab's fields and the Voices tab live. `kokoro_engine.py` is a slim core
    backed by a `kokoro_gui/engine/` package split by feature area (text extraction, caching,
    lexicon, presets, voice mixing, conversion, JIT, SRT).
-   **Second TTS engine, Audio8 (voice cloning).**
    [Audio8-TTS-Preview-0.6b](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6b), a zero-shot
    voice-cloning model, is selectable from Options > Engine alongside Kokoro. Unlike Kokoro's
    named voices, it clones a voice from a **reference WAV plus a transcript of what's said in it**;
    a Voice Reference dock (shown only for engines that support this) lets you browse a WAV,
    auto-transcribe it, edit the transcript, and save it under a name that then shows up in the
    normal Voice dropdown. The TTS model pulls in `transformers`/`torchaudio` (new
    `requirements.txt` entries) and loads with `trust_remote_code=True`. First use downloads it
    from Hugging Face.
-   **Two auto-transcription engines for Audio8's voice reference.** The Voice Reference dock's
    "Auto-Transcribe" button has an engine picker (`kokoro_gui/engine/asr.py`, also runnable
    standalone as `python -m kokoro_gui.engine.asr <wav>`). Default is
    [Audio8-ASR-0.1B](https://huggingface.co/Audio8/Audio8-ASR-0.1B), online, higher quality, but
    CC-BY-NC-4.0 (non-commercial), worth knowing if you build on this fork commercially. The
    alternative is [Vosk](https://alphacephei.com/vosk), fully offline and Apache-2.0. Vosk needs a
    model folder downloaded from https://alphacephei.com/vosk/models; the dock has a field for it
    with Browse/Save/Reload buttons, but the value itself lives in `VOSK_MODEL_PATH` in a `.env`
    file at the project root (copy `.env.example`) rather than in `config_qt.json` like every other
    setting, since it's a one-time deployment detail rather than a per-session preference. Whatever
    WAV format the reference audio is in, it's converted to the 16-bit mono PCM Vosk requires
    before recognition runs, so you don't have to pre-convert it.
-   **Segment cache rekeyed.** Cache entries are keyed on the voice's name and content rather than
    its path, so the first generate after upgrading from 3.2.0 misses the old `cache/` entries.
-   Not yet shipped: importing an existing audio recording and anchoring it to a transcript
    (ASR-anchored import) is a planned follow-up, not part of this release.

## New in 3.2.0

-   **Cross-platform audio playback.** Preview and JIT playback now go through `sounddevice`/
    `soundfile` instead of the Windows-only `winsound` module, removing a hard Windows dependency
    from `kokoro_engine.py`.

## New in 3.1.0

-   **JIT (Just-In-Time) generation.** Real-time audio streaming. Start listening to your text
    immediately as it's being generated.
-   **Audio FX pipeline.** Integrated [Pedalboard](https://github.com/spotify/pedalboard) support
    for Reverb, Compression, and EQ.
-   **Pronunciation lexicon.** Create a custom dictionary to override how specific words or
    acronyms are pronounced.
-   **Advanced voice mixing.** Create unique custom voices by mixing existing ones with precise
    control.
-   **Scripted multi-speaker and FX.** Use a simple syntax `[Speaker:FX]: Text` to switch voices
    and audio effects on the fly.
-   **Intelligent caching.** Automatically caches generated segments to speed up repeated tasks.
-   **Windows quick start.** New `run.bat` for easy one-click startup on Windows.

## Features

-   **Document-based editing:**
    -   The transcript is the source of truth. Generated audio is a render of the document's
        current state, tracked per clip with cache-hash-based dirty detection.
    -   Multi-track timeline: one lane per character, drag clips to reassign or move them, carve
        out and replace a sub-range with fresh TTS.
    -   Undo/redo for text edits and character reassignments.
    -   Auto-split a `[Speaker:FX]`-tagged script into clips and generate them in one action.
    -   Batch-generate only what's stale, or fall back to whole-document generation for projects
        that don't use clips.
-   **Multi-source input:**
    -   **Direct text:** type or paste directly into the transcript panel.
    -   **File support:** load `.txt`, `.pdf`, and `.epub` files. Good for turning e-books into
        audiobooks.
-   **Two synthesis engines, one interface:**
    -   **Kokoro** (default): named base voices plus custom mixing, 8 languages, 24,000 Hz, one
        pipeline per worker thread for true parallel generation.
    -   **Audio8** (voice cloning): zero-shot cloning from a reference WAV + transcript, 44,100 Hz,
        one shared lock-serialized model.
    -   Both register behind the same backend abstraction, so switching engines (Options > Engine) swaps
        voices, sample rate, and the docks that make sense for that engine, live.
-   **Generation modes:**
    -   **Standard:** parallel batch processing across a thread pool.
    -   **JIT (real-time):** streamed generation with immediate playback, for engines fast enough
        to outrun playback.
-   **Audio FX and post-processing** (non-destructive: applied on playback and export, never
    written into a generated clip, so changing them never regenerates anything):
    -   **Live FX (Pedalboard):** Compressor, Limiter, Gain, shelf EQ, high/low-pass filters, Reverb,
        Delay, Chorus, Distortion, Phaser, Clipping, Pitch Shift, Bitcrush, GSM Compressor.
    -   **Per-clip FX override**, layered on top of a character's own FX preset, on top of the
        project's FX.
    -   **Traditional controls:** Speed (0.5x-2.0x), Volume, Pitch.
    -   **Cleanup:** Normalize and trim silence.
-   **Smart splitting:** split text by newlines, paragraphs, or sentences for better prosody at the
    seams.
-   **Flexible output:**
    -   Combine all segments into one final `.wav` (or `.flac`/`.mp3`/`.ogg`), or keep the individual
        segment files.
    -   **Subtitle export:** generate `.srt` files synced to the actual generated-segment durations.
    -   Custom output filenames and directories.
-   **Presets and characters:**
    -   Characters wrap the existing `presets/*.json` shape: name, voice, settings, and a highlight
        color, reusable across clips.
    -   Save and load FX presets separately from generation presets.
    -   Pronunciation lexicon: case-insensitive literal find-and-replace overrides, applied before
        synthesis.
-   **UI:** dark or light theme, persistent dock layouts (Workspace menu).

## Prerequisites

-   **Python 3.11+**
-   **[eSpeak NG](https://github.com/espeak-ng/espeak-ng)**

## Installation

Two lines are available. Pick one before cloning.

| | 4.0.0 beta (recommended) | 3.2.0 (old stable) |
|---|---|---|
| What it is | The DAW-style rebuild described above: PySide6, characters and clips on a timeline, `.tbaw` projects, Kokoro + Audio8 | The previous CustomTkinter app: one text box, one voice, generate to a folder |
| Status | Beta. Under active development; bugs are expected and reports are welcome | Frozen. No further fixes |
| Project files | `.tbaw`. The plan is for every 4.x release to open a `.tbaw` from any earlier 4.x, with the beta included (that's a goal, not a guarantee, until 4.0.0 final) | None. Output is loose `.wav` files, nothing to carry forward |
| Presets, mixes, lexicon | `presets/*.json` load as characters; `custom_voices/` mixes and the lexicon carry over | As-is |

The two are separate codebases that share a name and the Kokoro model. There is no upgrade path
for a 3.2.0 install other than cloning 4.0.0 alongside it; there is nothing to migrate except the
`presets/` and `custom_voices/` folders, which you can copy across.

1.  **Clone the version you want:**

    4.0.0 beta:
    ```bash
    git clone --branch 4.0.0-beta.1 --depth 1 https://github.com/CoffeeMethod/KokoroGUI.git
    cd KokoroGUI
    ```
    `4.0.0-beta.1` is the tag of the current beta; the
    [Releases](https://github.com/CoffeeMethod/KokoroGUI/releases) page lists every version and
    has a source zip for each. Drop `--branch` to run the development branch instead.

    3.2.0:
    ```bash
    git clone --branch 3.2.0 --depth 1 https://github.com/CoffeeMethod/KokoroGUI.git
    cd KokoroGUI
    ```
    Then follow the README in that checkout, not this one: its dependencies, prerequisites and
    launch command differ.

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    # On Windows:
    .venv\Scripts\activate
    # On macOS/Linux:
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

    *Note: If you have issues with `torch`, visit [pytorch.org](https://pytorch.org/get-started/locally/)
    for specific installation instructions tailored to your OS and hardware.*

4.  **(Optional) Configure offline ASR:** copy `.env.example` to `.env` and set `VOSK_MODEL_PATH` if
    you plan to use the Vosk auto-transcription engine for the Voice Reference dock instead of the
    default (online, non-commercial) Audio8 ASR model.

## Usage

1.  **Run the application:**
    -   **Windows:** double-click `run.bat` or run `python main.py`
    -   **Other:** run `python main.py`

    This launches the PySide6 (Qt) frontend. A welcome dialog lists recent projects with Resume,
    New, New from text file and Open (untick "Show at startup" to skip it; File > Welcome...
    reopens it); its details pane reads a project's clip count, audio length and engines straight
    from the `.tbaw` manifest. Behind it, a menu bar (File / Edit / Options / Workspace) over a
    2x2 grid of docks:

    -   **Transcript** (top-left): the editor, with Character and FX combos above it and a gutter
        that names the speaker and offers a per-clip regenerate button.
    -   **Settings / Audio FX / Lexicon / Voices** (top-right, tabbed): voice, speed, language and
        audio controls scoped to whatever's selected (document, clip or character); the Pedalboard
        chain, also scoped; pronunciation overrides; and the engine's voice tools (Mixing for
        Kokoro, Voice Reference for Audio8).
    -   **Timeline** (bottom-left): one lane per character on a seconds axis, ruler, playhead.
    -   **Transport** (bottom-right): play / pause / stop, Preview, Generate (with Auto-split in its
        menu), Cancel, and the progress line.

    The engine, compute device and theme live under **Options**; **Workspace** switches between
    the full grid and a Simple layout without the timeline.

2.  **Write and assign:**
    -   Type or paste text into the transcript, or File > Import Text for `.txt`/`.pdf`/`.epub`.
    -   Select a range and pick a character from the header combo (or the right-click Characters
        menu), or write inline `[Speaker:FX]: Text` tags and let Generate > Auto-split turn them
        into clips for you.
    -   Each character carries its own highlight color, visible right in the transcript.

3.  **Preview, generate, play, export:**
    -   Preview speaks a short sample of the current settings.
    -   Generate renders every out-of-date clip (or the whole document, for clip-free projects);
        the gutter's play buttons regenerate one clip at a time.
    -   Play (or Space) plays the arrangement; the timeline playhead and the transcript follow.
    -   Drag a clip on the timeline to move it in time, onto another lane to reassign it, or
        Shift+drag inside it to replace a sub-range with new TTS.
    -   File > Export mixes the timeline down to one file (plus optional `.srt` and per-clip files).
        The same dialog holds the project's bundle options (bundle generated audio, wav or flac).
    -   File > Save writes the `.tbaw`; Ctrl+S is safe to hit any time, it runs in the background.
    -   Undo/redo any of the above from the Edit menu.

## Running Tests

The project has a `pytest` suite under `tests/` covering the DAW document model (`tests/daw/`), the
Qt frontend (`tests/gui_qt/`), and `kokoro_engine.py`. Playback isn't Windows-only (see
[`playback.py`](playback.py)), and CI (`.github/workflows/tests.yml`) runs the engine, DAW and
audio suites on both `windows-latest` and `ubuntu-latest` (the Linux leg installs `libportaudio2`
for `sounddevice`). The Qt suite runs headless via `QT_QPA_PLATFORM=offscreen`, no virtual display
needed, but only locally; CI skips `tests/gui_qt/`. `macos-latest` isn't set up yet.

1.  **Install test dependencies** (on top of `requirements.txt`):
    ```bash
    pip install -r requirements-test.txt
    ```

2.  **Run the fast suite** (default):
    ```bash
    pytest
    ```
    This mocks the Kokoro pipeline, so it runs in seconds with no model download and no eSpeak NG
    required. Caching is disabled by default in every test except `tests/test_caching.py`.

3.  **Run the integration suite** (opt-in, real synthesis):
    ```bash
    pytest -m integration tests/integration -s
    ```
    Uses the real Kokoro pipeline, so it needs eSpeak NG on `PATH` (see Prerequisites) and downloads
    model weights on first use. It skips automatically if `espeak-ng` isn't found. Since real
    synthesis can't be verified automatically, each test speaks a short, self-describing sample
    naming the voice/mode and writes it to `tests/output/<timestamp>/.../*_transcript.txt` next to
    the generated `.wav`, listen to the audio and compare against the transcript to confirm it
    sounds right. The `-s` flag also prints the same text to the terminal as each test runs.

### CI

[.github/workflows/tests.yml](.github/workflows/tests.yml) runs step 2 above on push/PR against
`windows-latest` and `ubuntu-latest` (the Linux leg additionally installs `libportaudio2`) after
installing `requirements.txt` + `requirements-test.txt`, as
`pytest --ignore=tests/gui_qt -p no:pytest-qt`: the engine, caching, DAW model, mixer and
transport tests, without the Qt widget suite. The fast suite needs no eSpeak NG or model download,
so it's safe to run on every push/PR. The integration suite is slow and pulls model weights, so
it's intentionally left out as a manual/opt-in run rather than part of the default pipeline.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, the test conventions and what a PR needs.
Security reports go through the repository's Security tab, not Issues ([SECURITY.md](SECURITY.md)).

## Technologies Used

-   **[Kokoro](https://github.com/hexgrad/kokoro):** The default TTS engine.
-   **[Audio8-TTS-Preview-0.6b](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6b):** The
    voice-cloning TTS engine.
-   **[Pedalboard](https://github.com/spotify/pedalboard):** Audio effects processing.
-   **[PySide6](https://doc.qt.io/qtforpython/):** The graphical user interface.
-   **PyTorch:** Deep learning backend.
-   **SoundFile / sounddevice:** Audio file I/O and cross-platform playback.
-   **PyPDF & EbookLib:** Document parsing.
