# Kokoro TTS GUI

A modern, high-quality Text-to-Speech (TTS) application built with Python, featuring a user-friendly graphical interface and powered by the [Kokoro](https://github.com/hexgrad/kokoro) library.

![Kokoro GUI Screenshot](https://github.com/user-attachments/assets/fb727aef-8637-4f86-bb00-787cdad55f14)


https://github.com/user-attachments/assets/df2eac14-3ba5-4608-a1b9-59fcff8955de


## Overview

Kokoro TTS GUI provides a convenient way to convert large amounts of text or entire books into natural-sounding speech. It leverages the power of the Kokoro TTS engine and provides advanced features like parallel processing, document parsing (PDF, EPUB, TXT), and customizable audio output.

## Features

-   **Multi-Source Input:**
    -   **Direct Text:** Paste text directly into the application.
    -   **File Support:** Load and process `.txt`, `.pdf`, and `.epub` files. Ideal for converting e-books to audiobooks.
-   **High-Quality Voices:** Choose from a wide variety of male and female voices (American English).
-   **Advanced Configuration:**
    -   **Parallel Processing:** Utilize multiple CPU/GPU threads to speed up generation (configurable number of processes).
    -   **Audio Speed:** Adjust playback speed from 0.5x to 2.0x.
    -   **Smart Splitting:** Split text by newlines, paragraphs, or sentences for optimal prosody.
-   **Flexible Output:**
    -   **Chunking:** Save speech segments as individual `.wav` files.
    -   **Automatic Merging:** Automatically combine all segments into a single high-quality audio file.
    -   **Custom Naming:** Define base filenames and output directories.
-   **Real-time Feedback:** Progress tracking and status updates during the conversion process.

## Prerequisites

-   **Python 3.11+**
-   **FFmpeg** (Recommended for audio processing)
-   **CUDA** (Optional, for GPU acceleration if using PyTorch with CUDA)

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
    -   (Optional) Adjust speed and splitting patterns.
    -   Set your desired output directory and filename.
    -   Choose whether to keep separate chunks or merge them into one file.

3.  **Start Conversion:** Click "Start Conversion" and wait for the process to complete. You can monitor progress via the status label and progress bar.

## Technologies Used

-   **[Kokoro](https://github.com/hexgrad/kokoro):** The core TTS engine.
-   **Tkinter:** For the graphical user interface.
-   **PyTorch:** Deep learning backend for the TTS model.
-   **SoundFile:** For writing high-quality WAV files.
-   **PyPDF & EbookLib:** For parsing PDF and EPUB documents.
-   **BeautifulSoup4:** For cleaning text from EPUB/HTML sources.