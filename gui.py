import os
import time
import json
import re
import winsound
import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading
from kokoro_engine import KokoroEngine

# Set Default Appearance (will be overridden by settings)
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

CONFIG_FILE = "config.json"
PRESETS_DIR = "presets"

class TTSApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Kokoro TTS GUI (Modern)")
        self.geometry("700x900")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Ensure presets dir exists
        if not os.path.exists(PRESETS_DIR):
            os.makedirs(PRESETS_DIR)

        # Load Settings
        self.settings = self.load_settings()
        self.apply_settings()

        # Initialize Engine
        self.engine = KokoroEngine()
        self.engine.on_progress = self.on_engine_progress
        self.engine.on_status = self.on_engine_status
        self.engine.on_finish = self.on_engine_finish
        
        # Auto-save timer
        self.save_timer = None
        
        # Variables
        self.file_path_var = ctk.StringVar()
        self.voice_var = ctk.StringVar(value=self.settings.get("voice", "af_heart"))
        self.filename_var = ctk.StringVar(value=self.settings.get("filename", "output"))
        self.output_dir_var = ctk.StringVar(value=self.settings.get("out_dir", "audio_output"))
        self.speed_var = ctk.DoubleVar(value=self.settings.get("speed", 1.0))
        self.volume_var = ctk.DoubleVar(value=self.settings.get("volume", 1.0))
        self.pitch_var = ctk.DoubleVar(value=self.settings.get("pitch", 0.0))
        self.num_threads_var = ctk.IntVar(value=self.settings.get("num_threads", 1))
        self.split_pattern_var = ctk.StringVar(value=self.settings.get("split_pattern", r"\n+"))
        
        self.separate_files = ctk.BooleanVar(value=self.settings.get("separate", True))
        self.combine_post = ctk.BooleanVar(value=self.settings.get("combine", True))
        self.export_subtitles = ctk.BooleanVar(value=self.settings.get("export_subtitles", False))
        self.normalize_audio = ctk.BooleanVar(value=self.settings.get("normalize", False))
        self.trim_silence = ctk.BooleanVar(value=self.settings.get("trim", False))
        self.timecode_format = "%Y%m%d%H%M%S"

        # Setup Auto-save Traces
        self.setup_autosave()

        self.create_widgets()
        
        # Init Pipeline
        self.status_label.configure(text="Initializing engine...")
        self.engine.worker.run_coro(self.engine.init_pipeline_async())

    def setup_autosave(self):
        vars_to_trace = [
            self.voice_var, self.filename_var, self.output_dir_var,
            self.speed_var, self.volume_var, self.pitch_var,
            self.num_threads_var, self.split_pattern_var,
            self.separate_files, self.combine_post, self.export_subtitles,
            self.normalize_audio, self.trim_silence
        ]
        for v in vars_to_trace:
            v.trace_add("write", self.schedule_save)

    def schedule_save(self, *args):
        if self.save_timer:
            self.after_cancel(self.save_timer)
        self.save_timer = self.after(1000, self.save_settings)

    def load_settings(self):
        defaults = {
            "appearance": "Dark", 
            "scaling": "100%",
            "voice": "af_heart",
            "filename": "output",
            "out_dir": "audio_output",
            "speed": 1.0,
            "volume": 1.0,
            "pitch": 0.0,
            "num_threads": 1,
            "split_pattern": r"\n+",
            "separate": True,
            "combine": True,
            "export_subtitles": False,
            "normalize": False,
            "trim": False
        }
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    return {**defaults, **json.load(f)}
            except:
                pass
        return defaults

    def save_settings(self):
        if self.save_timer:
            self.after_cancel(self.save_timer)
            self.save_timer = None
            
        if hasattr(self, 'voice_var'):
            self.settings['voice'] = self.voice_var.get()
            self.settings['filename'] = self.filename_var.get()
            self.settings['out_dir'] = self.output_dir_var.get()
            self.settings['speed'] = self.speed_var.get()
            self.settings['volume'] = self.volume_var.get()
            self.settings['pitch'] = self.pitch_var.get()
            self.settings['num_threads'] = self.num_threads_var.get()
            self.settings['split_pattern'] = self.split_pattern_var.get()
            self.settings['separate'] = self.separate_files.get()
            self.settings['combine'] = self.combine_post.get()
            self.settings['export_subtitles'] = self.export_subtitles.get()
            self.settings['normalize'] = self.normalize_audio.get()
            self.settings['trim'] = self.trim_silence.get()

        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(self.settings, f, indent=4)
        except Exception as e:
            print(f"Failed to save settings: {e}")

    def apply_settings(self):
        ctk.set_appearance_mode(self.settings["appearance"])
        
        # Parse scaling
        scale_str = self.settings["scaling"].replace("%", "")
        try:
            scale_float = float(scale_str) / 100
            ctk.set_widget_scaling(scale_float)
        except:
            ctk.set_widget_scaling(1.0)

    # --- Preset Management ---
    
    def refresh_presets(self):
        presets = ["Select Preset..."]
        if os.path.exists(PRESETS_DIR):
            files = [f for f in os.listdir(PRESETS_DIR) if f.endswith(".json")]
            presets.extend([f[:-5] for f in files]) # Remove .json
        
        self.preset_combo.configure(values=presets)
        self.preset_combo.set("Select Preset...")

    def save_preset_dialog(self):
        dialog = ctk.CTkInputDialog(text="Enter preset name:", title="Save Preset")
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
                "trim": self.trim_silence.get()
            }
            
            fpath = os.path.join(PRESETS_DIR, f"{name}.json")
            try:
                with open(fpath, "w") as f:
                    json.dump(data, f, indent=4)
                messagebox.showinfo("Saved", f"Preset '{name}' saved successfully.")
                self.refresh_presets()
                self.preset_combo.set(name)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save preset: {e}")

    def load_preset(self, name):
        if name == "Select Preset...": return
        
        fpath = os.path.join(PRESETS_DIR, f"{name}.json")
        if os.path.exists(fpath):
            try:
                with open(fpath, "r") as f:
                    data = json.load(f)
                    
                if "voice" in data: self.voice_var.set(data["voice"])
                if "speed" in data: self.speed_var.set(data["speed"])
                if "volume" in data: self.volume_var.set(data["volume"])
                if "pitch" in data: self.pitch_var.set(data["pitch"])
                if "split_pattern" in data: self.split_pattern_var.set(data["split_pattern"])
                if "normalize" in data: self.normalize_audio.set(data["normalize"])
                if "trim" in data: self.trim_silence.set(data["trim"])
                
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
                messagebox.showerror("Error", f"Failed to load preset: {e}")

    def create_widgets(self):
        # Main Container
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1) # Main content expands
        self.grid_rowconfigure(2, weight=0) # Action bar fixed
        
        # Header
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10,0))
        
        ctk.CTkLabel(header_frame, text="Kokoro TTS", font=("Roboto", 20, "bold")).pack(side="left", padx=5)
        ctk.CTkButton(header_frame, text="⚙ Settings", width=80, height=28, command=self.open_settings).pack(side="right")
        
        main_frame = ctk.CTkScrollableFrame(self)
        main_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
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

        # Voice Selection
        ctk.CTkLabel(config_frame, text="Voice:").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.voice_options = [
            "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", 
            "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", 
            "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir", 
            "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa"
        ]
        ctk.CTkComboBox(config_frame, values=self.voice_options, variable=self.voice_var).grid(row=1, column=1, sticky="ew", padx=10)

        # Output Dir
        ctk.CTkLabel(config_frame, text="Output Folder:").grid(row=2, column=0, sticky="w", padx=10, pady=5)
        dir_row = ctk.CTkFrame(config_frame, fg_color="transparent")
        dir_row.grid(row=2, column=1, sticky="ew", padx=10)
        dir_row.grid_columnconfigure(0, weight=1)
        ctk.CTkEntry(dir_row, textvariable=self.output_dir_var).grid(row=0, column=0, sticky="ew", padx=(0,5))
        ctk.CTkButton(dir_row, text="...", width=40, command=self.browse_directory).grid(row=0, column=1)

        # Filename
        ctk.CTkLabel(config_frame, text="Base Filename:").grid(row=3, column=0, sticky="w", padx=10, pady=5)
        ctk.CTkEntry(config_frame, textvariable=self.filename_var).grid(row=3, column=1, sticky="ew", padx=10)

        # Speed
        self.speed_label = ctk.CTkLabel(config_frame, text="Speed: 1.0x")
        self.speed_label.grid(row=4, column=0, sticky="w", padx=10, pady=5)
        self.speed_slider = ctk.CTkSlider(config_frame, from_=0.5, to=2.0, number_of_steps=15, variable=self.speed_var, command=self.update_speed_label)
        self.speed_slider.grid(row=4, column=1, sticky="ew", padx=10)

        # Split Pattern
        ctk.CTkLabel(config_frame, text="Split By:").grid(row=5, column=0, sticky="w", padx=10, pady=5)
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
        
        self.split_combo.grid(row=5, column=1, sticky="ew", padx=10, pady=5)

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

        # Toggles
        toggle_frame = ctk.CTkFrame(audio_frame, fg_color="transparent")
        toggle_frame.grid(row=3, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
        
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

        # --- 4. Actions & Status ---
        action_frame = ctk.CTkFrame(self)
        action_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        
        self.status_label = ctk.CTkLabel(action_frame, text="Ready", text_color="gray", anchor="w")
        self.status_label.pack(fill="x", padx=10, pady=(5,0))
        
        self.detail_label = ctk.CTkLabel(action_frame, text="...", font=("Consolas", 10), text_color="gray", anchor="w")
        self.detail_label.pack(fill="x", padx=10, pady=(0,5))

        self.progress_bar = ctk.CTkProgressBar(action_frame)
        self.progress_bar.set(0)
        self.progress_bar.pack(fill="x", padx=10, pady=5)
        
        self.info_label = ctk.CTkLabel(action_frame, text="Time: 00:00 / ETA: --:-- | 0%")
        self.info_label.pack(pady=2)

        btn_frame = ctk.CTkFrame(action_frame, fg_color="transparent")
        btn_frame.pack(fill="x", pady=10)

        self.preview_btn = ctk.CTkButton(btn_frame, text="Preview Audio", command=self.preview_conversion, height=40, fg_color="#2B719E", hover_color="#205578")
        self.preview_btn.pack(side="left", fill="x", expand=True, padx=5)
        
        self.start_btn = ctk.CTkButton(btn_frame, text="Start Generation", command=self.start_conversion, height=40, font=("Roboto", 14, "bold"))
        self.start_btn.pack(side="left", fill="x", expand=True, padx=5)
        
        self.cancel_btn = ctk.CTkButton(btn_frame, text="Cancel", command=self.cancel_conversion, height=40, fg_color="#c42b1c", hover_color="#8a1f14", state="disabled")
        self.cancel_btn.pack(side="left", fill="x", expand=True, padx=5)

    def open_settings(self):
        toplevel = ctk.CTkToplevel(self)
        toplevel.title("Settings")
        toplevel.geometry("400x380")
        toplevel.grab_set() # Modal
        
        # Center the window
        toplevel.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() // 2) - (toplevel.winfo_width() // 2)
        y = self.winfo_y() + (self.winfo_height() // 2) - (toplevel.winfo_height() // 2)
        toplevel.geometry(f"400x380+{x}+{y}")

        frame = ctk.CTkFrame(toplevel)
        frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Appearance
        ctk.CTkLabel(frame, text="Appearance Mode:", font=("Roboto", 14, "bold")).pack(anchor="w", pady=(10, 5))
        app_menu = ctk.CTkOptionMenu(frame, values=["System", "Dark", "Light"], command=self.change_appearance)
        app_menu.set(self.settings["appearance"])
        app_menu.pack(fill="x", pady=5)
        
        # Scaling
        ctk.CTkLabel(frame, text="UI Scaling:", font=("Roboto", 14, "bold")).pack(anchor="w", pady=(15, 5))
        scale_menu = ctk.CTkOptionMenu(frame, values=["80%", "90%", "100%", "110%", "120%"], command=self.change_scaling)
        scale_menu.set(self.settings["scaling"])
        scale_menu.pack(fill="x", pady=5)
        
        ctk.CTkLabel(frame, text="Note: Restart may be required for optimal scaling.", text_color="gray", font=("Arial", 10)).pack(pady=20)

        ctk.CTkButton(frame, text="Close", command=toplevel.destroy).pack(side="bottom", pady=10)

    def change_appearance(self, new_val):
        self.settings["appearance"] = new_val
        ctk.set_appearance_mode(new_val)
        self.save_settings()

    def change_scaling(self, new_val):
        self.settings["scaling"] = new_val
        scale_float = float(new_val.replace("%", "")) / 100
        ctk.set_widget_scaling(scale_float)
        self.save_settings()

    # --- Logic ---

    def update_audio_labels(self, value):
        self.vol_label.configure(text=f"Volume: {int(self.volume_var.get() * 100)}%")
        self.pitch_label.configure(text=f"Pitch: {int(self.pitch_var.get())} st")

    def update_speed_label(self, value):
        self.speed_label.configure(text=f"Speed: {value:.1f}x")

    def change_threads(self, delta):
        try:
            current = int(self.num_threads_var.get())
        except:
            current = 1
        new_val = max(1, min(16, current + delta))
        self.num_threads_var.set(new_val)

    def update_split_pattern(self, choice):
        self.split_pattern_var.set(self.split_map[choice])

    def browse_directory(self):
        d = filedialog.askdirectory()
        if d: self.output_dir_var.set(d)

    def browse_file(self):
        f = filedialog.askopenfilename(filetypes=[("Documents", "*.txt *.pdf *.epub")])
        if f: self.file_path_var.set(f)

    def on_engine_status(self, msg, is_error):
        color = "#ff5555" if is_error else "gray" # Red or Gray
        # Schedule update on main thread
        self.after(0, lambda: self.status_label.configure(text=msg, text_color=color))

    def on_engine_progress(self, percent, elapsed, eta, detail):
        # Schedule update
        def _update():
            self.progress_bar.set(percent / 100.0)
            elapsed_str = time.strftime('%M:%S', time.gmtime(elapsed))
            self.info_label.configure(text=f"Time: {elapsed_str} / ETA: {eta} | {int(percent)}%")
            self.detail_label.configure(text=detail)
        self.after(0, _update)

    def on_engine_finish(self):
        self.after(0, lambda: self.set_ui_state(False))

    def set_ui_state(self, is_running):
        state = "disabled" if is_running else "normal"
        cancel_state = "normal" if is_running else "disabled"
        
        self.start_btn.configure(state=state)
        self.preview_btn.configure(state=state)
        self.cancel_btn.configure(state=cancel_state)
        self.thread_minus_btn.configure(state=state)
        self.thread_plus_btn.configure(state=state)
        self.thread_entry.configure(state=state)
        self.vol_slider.configure(state=state)
        self.pitch_slider.configure(state=state)
        
        if not is_running:
            self.progress_bar.set(0 if self.engine.cancel_event.is_set() else 1)

    def preview_conversion(self):
        # 1. Get Text
        current_tab = self.tab_view.get()
        text_data = ""
        
        if current_tab == "Direct Text":
            text_data = self.text_entry.get("1.0", "end").strip()
        else:
            fpath = self.file_path_var.get()
            if os.path.exists(fpath):
                try:
                    text_data = self.engine.extract_text_from_file(fpath)
                except:
                    pass
        
        if not text_data:
            text_data = "This is a sample audio preview using the Koh-koh-ro Tea-Tea-S engine. It demonstrates the voice quality and speed settings."
            
        # Truncate to 2 sentences
        sentences = re.split(r'(?<=[.!?])\s+', text_data)
        preview_text = " ".join(sentences[:2])
        if len(preview_text) > 500: # Safety cap
             preview_text = preview_text[:500]
             
        # Config
        voice = self.voice_var.get()
        speed = self.speed_var.get()
        
        extra_config = {
            'volume': self.volume_var.get(),
            'pitch': self.pitch_var.get(),
            'normalize': self.normalize_audio.get(),
            'trim_silence': self.trim_silence.get()
        }
        
        # Temp file
        import tempfile
        tmp_path = os.path.join(tempfile.gettempdir(), "kokoro_preview.wav")
        
        self.status_label.configure(text="Generating preview...", text_color="blue")
        
        def _on_preview_done(future):
            def _ui_update():
                try:
                    success = future.result()
                    if success:
                        self.status_label.configure(text="Playing preview...", text_color="green")
                        winsound.PlaySound(tmp_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                        self.after(3000, lambda: self.status_label.configure(text="Ready", text_color="gray"))
                    else:
                        self.status_label.configure(text="Preview failed.", text_color="red")
                except Exception as e:
                    self.status_label.configure(text=f"Preview error: {e}", text_color="red")
            
            self.after(0, _ui_update)
        
        future = self.engine.worker.run_coro(self.engine.generate_preview(preview_text, voice, speed, tmp_path, extra_config))
        future.add_done_callback(_on_preview_done)

    def start_conversion(self):
        # 0. Validate Threads
        try:
            val = int(self.num_threads_var.get())
            if val < 1: val = 1
            self.num_threads_var.set(val)
        except:
            self.num_threads_var.set(1)

        # 1. Get Text
        current_tab = self.tab_view.get()
        text_data = ""
        
        if current_tab == "Direct Text":
            text_data = self.text_entry.get("1.0", "end").strip()
        else:
            fpath = self.file_path_var.get()
            if not os.path.exists(fpath):
                messagebox.showerror("Error", "File not found.")
                return
            try:
                text_data = self.engine.extract_text_from_file(fpath)
            except Exception as e:
                messagebox.showerror("Error", f"Read failed: {e}")
                return

        if not text_data:
            messagebox.showwarning("Empty", "No text to process.")
            return

        if not self.engine.pipeline:
             messagebox.showinfo("Wait", "Engine is initializing... please wait 2 seconds and try again.")
             return

        # 2. Config
        config = {
            'voice': self.voice_var.get(),
            'speed': self.speed_var.get(),
            'split_pattern': self.split_pattern_var.get(),
            'filename': self.filename_var.get(),
            'out_dir': self.output_dir_var.get(),
            'separate': self.separate_files.get(),
            'combine': self.combine_post.get(),
            'export_subtitles': self.export_subtitles.get(),
            'time_id': time.strftime(self.timecode_format),
            'num_threads': self.num_threads_var.get(),
            'volume': self.volume_var.get(),
            'pitch': self.pitch_var.get(),
            'normalize': self.normalize_audio.get(),
            'trim_silence': self.trim_silence.get()
        }

        # 3. Start
        self.set_ui_state(True)
        self.progress_bar.set(0)
        self.engine.start_conversion(text_data, config)

    def cancel_conversion(self):
        self.engine.cancel()
        self.status_label.configure(text="Cancelling... waiting for workers...", text_color="orange")

    def on_close(self):
        self.save_settings()
        self.destroy()

if __name__ == "__main__":
    app = TTSApp()
    app.mainloop()
