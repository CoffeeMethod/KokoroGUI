# KokoroGUI
# Text-to-Speech Application

## Overview

![image](https://github.com/user-attachments/assets/f19d08f3-aed5-4d29-b5f0-b5e391b08dec)


https://github.com/user-attachments/assets/df2eac14-3ba5-4608-a1b9-59fcff8955de


This is a simple Text-to-Speech (TTS) application built using Python and the `kokoro` library. It provides a graphical user interface (GUI) built with `tkinter` that allows users to convert text into speech using various voices. The application supports saving the generated audio to `.wav` files, selecting different voices, and customizing output filenames and directories.

## Features

-   **Text Input:** Allows users to enter the text they want to convert to speech.
-   **Voice Selection:** Provides a dropdown menu to select from a variety of available voices.
-   **Filename and Output Directory Customization:** Users can specify the filename and output directory for the generated audio files.
-   **Separate Audio Files:** Option to save each segment of the generated speech as a separate file.
-   **Timecode Formatting:** Option to customize the timecode format for the generated audio files.
-   **Combine Post-Processing:** Option to combine all generated audio segments into a single file after generation.
-   **Error Handling:** Displays error messages for common issues such as missing text or pipeline initialization failures.
-   **Status Updates:** Provides status updates during the conversion process.

## Requirements

-   Python 3.11
-   tkinter
-   kokoro
-   soundfile
-   torch
-   numpy

You can install the required packages using pip:

```bash
pip install tkinter kokoro soundfile torch numpy
