"""Generation tab: input source, voice/speed/output config, and the speaker
presets (`presets/*.json`) that snapshot that config.

Calls `gui.messagebox`, `gui.ctk.CTkInputDialog`, and reads `gui.PRESETS_DIR`
qualified, at call time, so tests can keep monkeypatching those names on the
`gui` module (the `tts_app` fixture redirects `PRESETS_DIR` into a tmp_path and
replaces `messagebox` with a `MagicMock()`).
"""
import json
import os
import re

import customtkinter as ctk

import gui


class GenerationTabMixin:
    def refresh_presets(self):
        presets = ["Select Preset..."]
        if os.path.exists(gui.PRESETS_DIR):
            files = [f for f in os.listdir(gui.PRESETS_DIR) if f.endswith(".json")]
            presets.extend([f[:-5] for f in files]) # Remove .json

        self.preset_combo.configure(values=presets)
        self.preset_combo.set("Select Preset...")

    def save_preset_dialog(self):
        dialog = gui.ctk.CTkInputDialog(text="Enter preset name:", title="Save Preset")
        name = dialog.get_input()
        if name:
            name = re.sub(r'[<>:"/\\|?*]', '', name).strip() # Sanitize
            if not name: return

            data = {
                "voice": self.voice_var.get(),
                "speed": self.speed_var.get(),
                "volume": self.volume_var.get(),
                "pitch": self.pitch_var.get(),
                "split_pattern": self.split_pattern_var.get(),
                "normalize": self.normalize_audio.get(),
                "trim": self.trim_silence.get(),
                "format": self.output_format_var.get(),
                "apply_fx": self.apply_fx_var.get(),
                "fx_preset": self.gen_fx_combo.get()
            }

            fpath = os.path.join(gui.PRESETS_DIR, f"{name}.json")
            try:
                with open(fpath, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4)
                gui.messagebox.showinfo("Saved", f"Preset '{name}' saved successfully.")
                self.refresh_presets()
                self.preset_combo.set(name)
            except Exception as e:
                gui.messagebox.showerror("Error", f"Failed to save preset: {e}")

    def load_preset(self, name):
        if name == "Select Preset...": return

        fpath = os.path.join(gui.PRESETS_DIR, f"{name}.json")
        if os.path.exists(fpath):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if "voice" in data: self.voice_var.set(data["voice"])
                if "speed" in data: self.speed_var.set(data["speed"])
                if "volume" in data: self.volume_var.set(data["volume"])
                if "pitch" in data: self.pitch_var.set(data["pitch"])
                if "split_pattern" in data: self.split_pattern_var.set(data["split_pattern"])
                if "normalize" in data: self.normalize_audio.set(data["normalize"])
                if "trim" in data: self.trim_silence.set(data["trim"])
                if "format" in data: self.output_format_var.set(data["format"])
                if "apply_fx" in data: self.apply_fx_var.set(data["apply_fx"])

                if "fx_preset" in data:
                    fx_name = data["fx_preset"]
                    if fx_name and fx_name != "Select FX Preset...":
                        self.load_fx_preset(fx_name)
                        # Ensure combo is updated (load_fx_preset does this, but being safe)
                        if hasattr(self, 'gen_fx_combo'): self.gen_fx_combo.set(fx_name)

                # Update UI labels manually since setting var triggers trace but maybe not UI update logic dependent on callbacks
                self.update_audio_labels(0)
                self.update_speed_label(self.speed_var.get())

                # Update split combo logic
                target_pat = self.split_pattern_var.get()
                for k, v in self.split_map.items():
                    if v == target_pat:
                        self.split_combo.set(k)
                        break

            except Exception as e:
                gui.messagebox.showerror("Error", f"Failed to load preset: {e}")

    def build_generation_tab(self, parent):
        parent.grid_columnconfigure(0, weight=1)

        # Move existing logic here
        main_frame = ctk.CTkScrollableFrame(parent)
        main_frame.pack(fill="both", expand=True, padx=5, pady=5)
        main_frame.grid_columnconfigure(0, weight=1)

        # --- 1. Input Section ---
        input_frame = ctk.CTkFrame(main_frame)
        input_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        input_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(input_frame, text="Input Source", font=("Roboto", 16, "bold")).grid(row=0, column=0, sticky="w", padx=10, pady=5)

        self.tab_view = ctk.CTkTabview(input_frame, height=150)
        self.tab_view.grid(row=1, column=0, sticky="ew", padx=10, pady=5)

        # Text Tab
        tab_text = self.tab_view.add("Direct Text")
        tab_text.grid_columnconfigure(0, weight=1)
        tab_text.grid_rowconfigure(0, weight=1)

        self.text_entry = ctk.CTkTextbox(tab_text, wrap="word")
        self.text_entry.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # File Tab
        tab_file = self.tab_view.add("Load File")
        tab_file.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(tab_file, text="File Path:").grid(row=0, column=0, padx=10, pady=20)
        ctk.CTkEntry(tab_file, textvariable=self.file_path_var).grid(row=0, column=1, sticky="ew", padx=5)
        ctk.CTkButton(tab_file, text="Browse", width=80, command=self.browse_file).grid(row=0, column=2, padx=10)
        ctk.CTkLabel(tab_file, text="Supported: .txt, .pdf, .epub", text_color="gray").grid(row=1, column=1, sticky="w", padx=5)

        # --- 2. Configuration ---
        config_frame = ctk.CTkFrame(main_frame)
        config_frame.grid(row=1, column=0, sticky="ew", pady=10)
        config_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(config_frame, text="Configuration", font=("Roboto", 16, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=5)

        # Presets Row
        preset_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        preset_frame.grid(row=0, column=1, sticky="ew", padx=10, pady=5)

        self.preset_combo = ctk.CTkComboBox(preset_frame, values=["Select Preset..."], command=self.load_preset, width=150)
        self.preset_combo.pack(side="left", padx=(0,5))

        ctk.CTkButton(preset_frame, text="💾", width=30, command=self.save_preset_dialog).pack(side="left", padx=2)
        ctk.CTkButton(preset_frame, text="🔄", width=30, command=self.refresh_presets).pack(side="left", padx=2)

        self.refresh_presets()

        # Language Selection
        ctk.CTkLabel(config_frame, text="Language:").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        # Reverse map for display
        lang_display_map = {v: k for k, v in self.LANGUAGES.items()}
        current_lang_code = self.lang_var.get()

        def on_lang_ui_change(choice):
            self.lang_var.set(self.LANGUAGES[choice])

        self.lang_combo = ctk.CTkComboBox(config_frame, values=list(self.LANGUAGES.keys()), command=on_lang_ui_change)

        # Set initial value
        if current_lang_code in lang_display_map:
            self.lang_combo.set(lang_display_map[current_lang_code])
        else:
            self.lang_combo.set("American English")

        self.lang_combo.grid(row=1, column=1, sticky="ew", padx=10)

        # Voice Selection
        ctk.CTkLabel(config_frame, text="Voice:").grid(row=2, column=0, sticky="w", padx=10, pady=5)
        self.voice_combo = ctk.CTkComboBox(config_frame, values=self.get_all_voices(), variable=self.voice_var)
        self.voice_combo.grid(row=2, column=1, sticky="ew", padx=10)

        # Output Dir
        ctk.CTkLabel(config_frame, text="Output Folder:").grid(row=3, column=0, sticky="w", padx=10, pady=5)
        dir_row = ctk.CTkFrame(config_frame, fg_color="transparent")
        dir_row.grid(row=3, column=1, sticky="ew", padx=10)
        dir_row.grid_columnconfigure(0, weight=1)
        ctk.CTkEntry(dir_row, textvariable=self.output_dir_var).grid(row=0, column=0, sticky="ew", padx=(0,5))
        ctk.CTkButton(dir_row, text="...", width=40, command=self.browse_directory).grid(row=0, column=1)

        # Filename
        ctk.CTkLabel(config_frame, text="Base Filename:").grid(row=4, column=0, sticky="w", padx=10, pady=5)

        file_row = ctk.CTkFrame(config_frame, fg_color="transparent")
        file_row.grid(row=4, column=1, sticky="ew", padx=10)
        file_row.grid_columnconfigure(0, weight=1)

        ctk.CTkEntry(file_row, textvariable=self.filename_var).grid(row=0, column=0, sticky="ew", padx=(0,5))

        self.format_combo = ctk.CTkComboBox(file_row, values=["wav", "flac", "mp3", "ogg"], width=70, variable=self.output_format_var)
        self.format_combo.grid(row=0, column=1)

        # Speed
        self.speed_label = ctk.CTkLabel(config_frame, text="Speed: 1.0x")
        self.speed_label.grid(row=5, column=0, sticky="w", padx=10, pady=5)
        self.speed_slider = ctk.CTkSlider(config_frame, from_=0.5, to=2.0, number_of_steps=15, variable=self.speed_var, command=self.update_speed_label)
        self.speed_slider.grid(row=5, column=1, sticky="ew", padx=10)

        # Split Pattern
        ctk.CTkLabel(config_frame, text="Split By:").grid(row=6, column=0, sticky="w", padx=10, pady=5)
        self.split_map = {
            "Natural (Newlines)": r"\n+",
            "Paragraphs (Double Newline)": r"\n\n+",
            "Sentences (.!?)": r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s"
        }
        self.split_combo = ctk.CTkComboBox(config_frame, values=list(self.split_map.keys()), command=self.update_split_pattern)

        # Determine initial selection based on loaded variable
        initial_pattern = self.split_pattern_var.get()
        initial_key = "Natural (Newlines)" # Default
        for k, v in self.split_map.items():
            if v == initial_pattern:
                initial_key = k
                break
        self.split_combo.set(initial_key)

        self.split_combo.grid(row=6, column=1, sticky="ew", padx=10, pady=5)

        # --- 3. Audio Control ---
        audio_frame = ctk.CTkFrame(main_frame)
        audio_frame.grid(row=2, column=0, sticky="ew", pady=10)
        audio_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(audio_frame, text="Audio Control", font=("Roboto", 16, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=5)

        # Volume
        self.vol_label = ctk.CTkLabel(audio_frame, text="Volume: 100%")
        self.vol_label.grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.vol_slider = ctk.CTkSlider(audio_frame, from_=0.1, to=2.0, number_of_steps=19, variable=self.volume_var, command=self.update_audio_labels)
        self.vol_slider.grid(row=1, column=1, sticky="ew", padx=10)

        # Pitch
        self.pitch_label = ctk.CTkLabel(audio_frame, text="Pitch: 0 st")
        self.pitch_label.grid(row=2, column=0, sticky="w", padx=10, pady=5)
        self.pitch_slider = ctk.CTkSlider(audio_frame, from_=-12, to=12, number_of_steps=24, variable=self.pitch_var, command=self.update_audio_labels)
        self.pitch_slider.grid(row=2, column=1, sticky="ew", padx=10)

        # FX Preset
        ctk.CTkLabel(audio_frame, text="FX Preset:").grid(row=3, column=0, sticky="w", padx=10, pady=5)
        fx_row = ctk.CTkFrame(audio_frame, fg_color="transparent")
        fx_row.grid(row=3, column=1, sticky="ew", padx=10)
        fx_row.grid_columnconfigure(0, weight=1)

        self.gen_fx_combo = ctk.CTkComboBox(fx_row, values=["Select FX Preset..."], command=self.load_fx_preset)
        self.gen_fx_combo.pack(side="left", fill="x", expand=True)
        ctk.CTkCheckBox(fx_row, text="Apply", variable=self.apply_fx_var, width=60).pack(side="left", padx=5)

        self.refresh_fx_presets() # Ensure values are populated

        # Toggles
        toggle_frame = ctk.CTkFrame(audio_frame, fg_color="transparent")
        toggle_frame.grid(row=4, column=0, columnspan=2, sticky="ew", padx=10, pady=5)

        ctk.CTkCheckBox(toggle_frame, text="Normalize", variable=self.normalize_audio).pack(side="left", padx=5)
        ctk.CTkCheckBox(toggle_frame, text="Trim Silence", variable=self.trim_silence).pack(side="left", padx=5)


        # --- 4. Advanced Options ---
        adv_frame = ctk.CTkFrame(main_frame)
        adv_frame.grid(row=3, column=0, sticky="ew", pady=10)

        ctk.CTkLabel(adv_frame, text="Processing Options", font=("Roboto", 14, "bold")).pack(anchor="w", padx=10, pady=5)

        chk_frame = ctk.CTkFrame(adv_frame, fg_color="transparent")
        chk_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkCheckBox(chk_frame, text="Keep Segments", variable=self.separate_files).pack(side="left", padx=5)
        ctk.CTkCheckBox(chk_frame, text="Combine Output", variable=self.combine_post).pack(side="left", padx=5)
        ctk.CTkCheckBox(chk_frame, text="Export Subtitles (.srt)", variable=self.export_subtitles).pack(side="left", padx=5)

        # Threads
        thread_frame = ctk.CTkFrame(adv_frame, fg_color="transparent")
        thread_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(thread_frame, text="Parallel Threads:").pack(side="left", padx=(5, 10))

        self.thread_minus_btn = ctk.CTkButton(thread_frame, text="-", width=30, command=lambda: self.change_threads(-1))
        self.thread_minus_btn.pack(side="left", padx=2)

        self.thread_entry = ctk.CTkEntry(thread_frame, textvariable=self.num_threads_var, width=50, justify="center")
        self.thread_entry.pack(side="left", padx=2)

        self.thread_plus_btn = ctk.CTkButton(thread_frame, text="+", width=30, command=lambda: self.change_threads(1))
        self.thread_plus_btn.pack(side="left", padx=2)

        ctk.CTkLabel(thread_frame, text="(More threads = High RAM usage)", text_color="orange").pack(side="left", padx=10)
