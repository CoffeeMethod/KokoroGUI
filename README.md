# Kokoro TTS GUI

A modern, high-quality Text-to-Speech (TTS) application built with Python, featuring a user-friendly graphical interface and powered by the [Kokoro](https://github.com/hexgrad/kokoro) library.

![Kokoro GUI Screenshot](<img width="1225" height="1598" alt="image" src="https://github.com/user-attachments/assets/56a6a3aa-6fa1-4638-ab91-212cdad2eeae" />
)

[output_20260129210938_combined.wav](https://github.com/user-attachments/files/24953984/output_20260129210938_combined.wav)


## Overview

Kokoro TTS GUI provides a convenient way to convert large amounts of text or entire books into natural-sounding speech. It leverages the power of the Kokoro TTS engine and provides advanced features like parallel processing, document parsing (PDF, EPUB, TXT), and customizable audio output.

## Features

-   **Multi-Source Input:**
    -   **Direct Text:** Paste text directly into the application.
    -   **File Support:** Load and process `.txt`, `.pdf`, and `.epub` files. Ideal for converting e-books to audiobooks.
-   **High-Quality Voices:** Choose from a wide variety of American English voices.
-   **Advanced Configuration:**
    -   **Parallel Processing:** Utilize multiple threads to speed up generation (configurable number of processes).
    -   **Audio Speed:** Adjust playback speed from 0.5x to 2.0x.
    -   **Audio Control:** Fine-tune Volume and Pitch.
    -   **Post-Processing:** Options to Normalize audio and Trim silence.
    -   **Smart Splitting:** Split text by newlines, paragraphs, or sentences for optimal prosody.
-   **Flexible Output:**
    -   **Automatic Merging:** Automatically combine all segments into a single high-quality `.wav` audio file.
    -   **Subtitle Export:** Generate `.srt` subtitle files synchronized with the audio.
    -   **Chunking:** Option to keep individual speech segments as separate files.
    -   **Custom Naming:** Define base filenames and output directories.
-   **User Experience:**
    -   **Presets:** Save and load your favorite voice and audio settings.
    -   **Audio Preview:** Quickly test voice and speed settings with a short preview.
    -   **UI Customization:** Adjustable interface scaling and theme (Dark/Light/System).
-   **Robust Processing:**
    -   **Text Cleaning:** Automatically strips HTML and formatting from EPUBs for clean reading.
    -   **Real-time Feedback:** Live progress tracking, time estimation, and status updates.

## Prerequisites

-   **Python 3.11+**

## Installation

1.  **Clone the repository:**
    ```bash
    git https://github.com/CoffeeMethod/KokoroGUI.git
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
    ```bash
    python main.py
    ```

2.  **Configure your conversion:**
    -   Choose your input method (Direct Text or Load File).
    -   Select a voice from the dropdown menu.
    -   (Optional) Adjust speed, volume, and pitch.
    -   (Optional) Use **Presets** to save or load configurations.
    -   Set your desired output directory and filename.
    -   Choose whether to keep separate chunks or merge them into one file.

3.  **Preview & Convert:**
    -   Click "Preview Audio" to hear a short sample of the current settings.
    -   Click "Start Conversion" to begin the full process. You can monitor progress via the status label and progress bar.

## Technologies Used

-   **[Kokoro](https://github.com/hexgrad/kokoro):** The core TTS engine.
-   **Customtkinter:** For the graphical user interface.
-   **PyTorch:** Deep learning backend for the TTS model.
-   **SoundFile:** For writing high-quality WAV files.
-   **PyPDF & EbookLib:** For parsing PDF and EPUB documents.
-   **BeautifulSoup4:** For cleaning text from EPUB/HTML sources.
