# Kokoro TTS GUI

A modern, high-quality Text-to-Speech (TTS) application built with Python, featuring a user-friendly graphical interface and powered by the [Kokoro](https://github.com/hexgrad/kokoro) library.

<img width="1225" height="1598" alt="image" src="https://github.com/user-attachments/assets/56a6a3aa-6fa1-4638-ab91-212cdad2eeae" />


(demo sounds better in `.wav` but GitHub dosent suport that so its kinda bad)

https://github.com/user-attachments/assets/c75e7141-5d73-40f4-b182-d4f5bc49ad1e

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

## Technologies Used

-   **[Kokoro](https://github.com/hexgrad/kokoro):** The core TTS engine.
-   **[Pedalboard](https://github.com/spotify/pedalboard):** Audio effects processing.
-   **Customtkinter:** For the graphical user interface.
-   **PyTorch:** Deep learning backend.
-   **SoundFile:** For writing high-quality audio files.
-   **PyPDF & EbookLib:** For parsing documents.
