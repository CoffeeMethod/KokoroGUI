# Kokoro TTS GUI

A modern, high-quality Text-to-Speech (TTS) application built with Python, featuring a user-friendly graphical interface and powered by the [Kokoro](https://github.com/hexgrad/kokoro) library.

<img width="933" height="1787" alt="Screenshot 2026-02-13 172822" src="https://github.com/user-attachments/assets/a6ec3d30-e837-4815-a0e1-1ff69d1e31be" />


(demo sounds better in `.wav` but GitHub dosent suport that so its kinda bad)

https://github.com/user-attachments/assets/c75e7141-5d73-40f4-b182-d4f5bc49ad1e

## New in Beta 3.2.0

-   **Modular codebase:** `gui.py` and `kokoro_engine.py` are now split into a `kokoro_gui/engine/` and `kokoro_gui/ui/` package by feature area (text extraction, caching, lexicon, presets, voice mixing, per-tab UI builders), making the codebase easier to navigate and extend. No user-facing behavior change.
-   **Cross-Platform Audio Playback:** Preview and JIT playback now go through `sounddevice`/`soundfile` instead of the Windows-only `winsound` module, removing a hard Windows dependency from `kokoro_engine.py`/`gui.py`.

## New in 3.1.0

-   **JIT (Just-In-Time) Generation:** Real-time audio streaming. Start listening to your text immediately as it's being generated.
-   **Audio FX Pipeline:** Integrated [Pedalboard](https://github.com/spotify/pedalboard) support for Reverb, Compression, and EQ.
-   **Pronunciation Lexicon:** Create a custom dictionary to override how specific words or acronyms are pronounced.
-   **Advanced Voice Mixing:** Create unique custom voices by mixing existing ones with precise control.
-   **Scripted Multi-Speaker & FX:** Use a simple syntax `[Speaker:FX]: Text` to switch voices and audio effects on the fly.
-   **Intelligent Caching:** Automatically caches generated segments to speed up repeated tasks.
-   **Windows Quick Start:** New `run.bat` for easy one-click startup on Windows.

## Features

-   **Multi-Source Input:**
    -   **Direct Text:** Paste text directly into the application.
    -   **File Support:** Load and process `.txt`, `.pdf`, and `.epub` files. Ideal for converting e-books to audiobooks.
-   **High-Quality Voices & Languages:** 
    -   Supports American English, British English, Spanish, French, Italian, Portuguese, Japanese, and Chinese.
    -   Wide variety of base voices plus custom voice mixing.
-   **Generation Modes:**
    -   **Standard:** High-speed parallel processing for batch conversion.
    -   **JIT (Real-time):** Sequential generation with immediate playback and buffer management.
-   **Audio FX & Post-Processing:**
    -   **Live FX:** Reverb, Compressor, Low/High Shelf filters.
    -   **Traditional:** Adjust Speed (0.5x to 2.0x), Volume, and Pitch.
    -   **Cleanup:** Normalize audio and Trim silence.
-   **Smart Splitting:** Split text by newlines, paragraphs, or sentences for optimal prosody.
-   **Flexible Output:**
    -   **Automatic Merging:** Combine all segments into a single high-quality `.wav`.
    -   **Subtitle Export:** Generate `.srt` files synchronized with the audio.
    -   **Custom Naming:** Define base filenames and output directories.
-   **User Experience:**
    -   **Presets:** Save and load your favorite configurations (including FX).
    -   **Lexicon:** User-defined pronunciation overrides.
    -   **UI Customization:** Adjustable interface scaling and theme (Dark/Light/System).

## Prerequisites

-   **Python 3.11+**
-   **[eSpeak NG](https://github.com/espeak-ng/espeak-ng)**

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/CoffeeMethod/KokoroGUI.git
    cd KokoroGUI
    ```

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

    *Note: If you have issues with `torch`, visit [pytorch.org](https://pytorch.org/get-started/locally/) for specific installation instructions tailored to your OS and hardware.*

## Usage

1.  **Run the application:**
    -   **Windows:** Double-click `run.bat` or run `python main.py`
    -   **Other:** Run `python main.py`

2.  **Configure your conversion:**
    -   Choose your input method (Direct Text or Load File).
    -   Select a voice and language from the dropdown menus.
    -   (Optional) Enable **JIT Generation** in Settings for real-time playback.
    -   (Optional) Use the **Lexicon** tab to add pronunciation overrides.
    -   (Optional) Use the **Custom Voice** tab to mix new voices.
    -   (Optional) Use the **FX** settings to add Reverb or Compression.

3.  **Preview & Convert:**
    -   Click "Preview Audio" to hear a short sample.
    -   Click "Start Generation" (or "Start Real-time JIT") to begin.

## Running Tests

The project has a `pytest` suite under `tests/` covering both `gui.py` and `kokoro_engine.py`. Playback no longer forces Windows-only (see [`playback.py`](playback.py)), and CI (`.github/workflows/tests.yml`) now runs the suite on both `windows-latest` and `ubuntu-latest` (the Linux leg installs `libportaudio2` for `sounddevice` and runs under `xvfb-run` since the GUI tests build real Tk windows). `macos-latest` isn't set up yet.

1.  **Install test dependencies** (on top of `requirements.txt`):
    ```bash
    pip install -r requirements-test.txt
    ```

2.  **Run the fast suite** (default):
    ```bash
    pytest
    ```
    This mocks the Kokoro pipeline, so it runs in seconds with no model download and no eSpeak NG required. Caching is disabled by default in every test except `tests/test_caching.py`.

3.  **Run the integration suite** (opt-in, real synthesis):
    ```bash
    pytest -m integration tests/integration -s
    ```
    Uses the real Kokoro pipeline, so it needs eSpeak NG on `PATH` (see Prerequisites) and downloads model weights on first use. It skips automatically if `espeak-ng` isn't found. Since real synthesis can't be verified automatically, each test speaks a short, self-describing sample naming the voice/mode and writes it to `tests/output/<timestamp>/.../*_transcript.txt` next to the generated `.wav` — listen to the audio and compare against the transcript to confirm it sounds right. The `-s` flag also prints the same text to the terminal as each test runs.

### CI

[.github/workflows/tests.yml](.github/workflows/tests.yml) runs step 2 above (`pytest`) on push/PR against `windows-latest` and `ubuntu-latest` (the Linux leg additionally installs `libportaudio2` and runs under `xvfb-run`, as noted above) after installing `requirements.txt` + `requirements-test.txt`. The fast suite needs no eSpeak NG or model download, so it's safe to run on every push/PR. The integration suite is slow and pulls model weights, so it's intentionally left out as a manual/opt-in run rather than part of the default pipeline.

## Technologies Used

-   **[Kokoro](https://github.com/hexgrad/kokoro):** The core TTS engine.
-   **[Pedalboard](https://github.com/spotify/pedalboard):** Audio effects processing.
-   **Customtkinter:** For the graphical user interface.
-   **PyTorch:** Deep learning backend.
-   **SoundFile:** For writing high-quality audio files.
-   **PyPDF & EbookLib:** For parsing documents.
