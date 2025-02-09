import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from kokoro import KPipeline
import soundfile as sf
import torch
import numpy as np
import time

class TTSApp:
  def __init__(self, master):
    self.master = master
    master.title("Text-to-Speech Application")

    # Variables
    self.text = tk.StringVar()
    self.voice = tk.StringVar(value="af_heart")  # Default voice
    self.filename = tk.StringVar(value="output")  # Default filename
    self.output_directory = tk.StringVar(value="audio_output")
    self.pipeline = None  # Initialize pipeline to None
    self.separate_files = tk.BooleanVar(value=True)  # Default: separate files
    self.timecode_format = tk.StringVar(
        value="%Y%m%d%H%M%S"
    )  # Default timecode format
    self.combine_post = tk.BooleanVar(
        value=False
    )  # Default: don't combine post

    # UI elements
    self.create_widgets()

  def create_widgets(self):
    # Text Input
    ttk.Label(self.master, text="Enter Text:").grid(
        row=0, column=0, padx=5, pady=5, sticky="w"
    )
    self.text_entry = tk.Text(self.master, height=5, width=40)
    self.text_entry.grid(row=0, column=1, padx=5, pady=5)

    # Voice Recommendation Label
    self.voice_recommendation_label = ttk.Label(
        self.master,
        text="Recommended voices: af_heart, am_michael, am_puck",
    )
    self.voice_recommendation_label.grid(
        row=1, column=1, padx=5, pady=5, sticky="w"
    )

    # Voice Selection
    ttk.Label(self.master, text="Select Voice:").grid(
        row=2, column=0, padx=5, pady=5, sticky="w"
    )
    self.voice_options = [
        "af_alloy",
        "af_aoede",
        "af_bella",
        "af_jessica",
        "af_kore",
        "af_nicole",
        "af_nova",
        "af_river",
        "af_sarah",
        "af_sky",
        "am_adam",
        "am_echo",
        "am_eric",
        "am_fenrir",
        "am_liam",
        "am_michael",
        "am_onyx",
        "am_puck",
        "am_santa",
        "af_heart",
    ]  # All voices
    self.voice_combo = ttk.Combobox(
        self.master, textvariable=self.voice, values=self.voice_options
    )
    self.voice_combo.grid(row=2, column=1, padx=5, pady=5, sticky="w")

    # Filename Input
    ttk.Label(self.master, text="Enter Filename:").grid(
        row=3, column=0, padx=5, pady=5, sticky="w"
    )
    self.filename_entry = ttk.Entry(self.master, textvariable=self.filename)
    self.filename_entry.grid(row=3, column=1, padx=5, pady=5, sticky="w")

    # Output Directory
    ttk.Label(self.master, text="Output Directory:").grid(
        row=4, column=0, padx=5, pady=5, sticky="w"
    )
    self.output_dir_entry = ttk.Entry(
        self.master, textvariable=self.output_directory, width=30
    )
    self.output_dir_entry.grid(row=4, column=1, padx=5, pady=5, sticky="w")

    self.browse_button = ttk.Button(
        self.master, text="Browse", command=self.browse_directory
    )
    self.browse_button.grid(row=4, column=2, padx=5, pady=5, sticky="w")

    # Separate Files Checkbox
    self.separate_files_check = ttk.Checkbutton(
        self.master,
        text="Separate Audio Files",
        variable=self.separate_files,
    )
    self.separate_files_check.grid(row=5, column=1, padx=5, pady=5, sticky="w")

    # Timecode Format Input
    ttk.Label(self.master, text="Timecode Format:").grid(
        row=6, column=0, padx=5, pady=5, sticky="w"
    )
    self.timecode_format_entry = ttk.Entry(
        self.master, textvariable=self.timecode_format, width=30
    )
    self.timecode_format_entry.grid(row=6, column=1, padx=5, pady=5, sticky="w")

    # Combine Post-Processing Checkbox
    self.combine_post_check = ttk.Checkbutton(
        self.master,
        text="Combine After Generation",
        variable=self.combine_post,
    )
    self.combine_post_check.grid(row=7, column=1, padx=5, pady=5, sticky="w")

    # Convert Button
    self.convert_button = ttk.Button(
        self.master, text="Convert to Speech", command=self.convert_text
    )
    self.convert_button.grid(row=8, column=1, padx=5, pady=10)

    # Status Label
    self.status_label = ttk.Label(self.master, text="")
    self.status_label.grid(row=9, column=0, columnspan=3, padx=5, pady=5)

  def browse_directory(self):
    directory = filedialog.askdirectory()
    if directory:
      self.output_directory.set(directory)

  def initialize_pipeline(self):
    # Initialize the pipeline only once
    if self.pipeline is None:
      try:
        self.pipeline = KPipeline(lang_code="a")
        self.status_label.config(text="Pipeline Initialized.")
      except Exception as e:
        messagebox.showerror("Error", f"Failed to initialize pipeline: {e}")
        self.status_label.config(text="Pipeline Initialization Failed.")
        self.pipeline = None  # Reset pipeline to None in case of failure
        return False  # Indicate failure
    return True  # Indicate success

  def convert_text(self):
    if not self.initialize_pipeline():
      return  # Exit if pipeline initialization failed

    text = self.text_entry.get("1.0", tk.END).strip()
    voice = self.voice.get()
    filename = self.filename.get()
    output_directory = self.output_directory.get()
    separate_files = self.separate_files.get()
    timecode_format = self.timecode_format.get()
    combine_post = self.combine_post.get()

    if not text:
      messagebox.showerror("Error", "Please enter text to convert.")
      return

    os.makedirs(output_directory, exist_ok=True)

    try:
      generator = self.pipeline(text, voice=voice, speed=1, split_pattern=r"\n+")

      # Get a unique time identifier
      time_identifier = time.strftime(timecode_format)

      audio_segments = []  # Store audio segments for post-processing

      for i, (gs, ps, audio) in enumerate(generator):
        print(f"Type of audio: {type(audio)}")

        if isinstance(audio, torch.Tensor):
          print("Audio is a torch.Tensor")
          audio_numpy = audio.cpu().numpy()
          print(f"Type of audio_numpy: {type(audio_numpy)}")
        else:
          print("Audio is NOT a torch.Tensor")
          audio_numpy = audio

        if not isinstance(audio_numpy, np.ndarray):
          audio_numpy = np.array(audio_numpy)
          print(f"Converted audio to numpy array: {type(audio_numpy)}")

        audio_bytes = audio_numpy.tobytes()
        print(f"Type of audio_bytes: {type(audio_bytes)}")

        # Save individual segments
        file_path = os.path.join(
            output_directory, f"{filename}_{time_identifier}_{i}.wav"
        )
        sf.write(file_path, audio_numpy, 24000)
        print(f"Saved audio to {file_path}")
        audio_segments.append(file_path)  # Store the file path

      if combine_post:
        # Post-processing: Combine all audio segments into a single file
        all_audio = []
        for file_path in audio_segments:
          # Load each audio segment from file
          audio_numpy, samplerate = sf.read(file_path)
          all_audio.append(audio_numpy)

        # Concatenate all audio segments
        combined_audio = np.concatenate(all_audio)

        # Save the combined audio to a single file
        combined_file_path = os.path.join(
            output_directory, f"{filename}_{time_identifier}_combined.wav"
        )
        sf.write(combined_file_path, combined_audio, 24000)
        print(f"Saved combined audio to {combined_file_path}")

      self.status_label.config(text="Conversion complete!")
      messagebox.showinfo("Success", "Text converted to speech successfully!")

    except Exception as e:
      messagebox.showerror("Error", f"Conversion failed: {e}")
      self.status_label.config(text=f"Conversion failed: {e}")


# Main application
if __name__ == "__main__":
  root = tk.Tk()
  app = TTSApp(root)
  root.mainloop()
