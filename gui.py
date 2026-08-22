import os
import time
import json
import playback
import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading
from kokoro_engine import KokoroEngine

from kokoro_gui.ui import FXTabMixin, GenerationTabMixin, LexiconTabMixin, MixingTabMixin

# Set Default Appearance (will be overridden by settings)
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

CONFIG_FILE = "config.json"
PRESETS_DIR = "presets"
FX_PRESETS_DIR = os.path.join(PRESETS_DIR, "fx")

class TTSApp(FXTabMixin, GenerationTabMixin, LexiconTabMixin, MixingTabMixin, ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Kokoro TTS GUI")
        self.geometry("700x900")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Ensure presets dirs exist
        if not os.path.exists(PRESETS_DIR):
            os.makedirs(PRESETS_DIR)
        if not os.path.exists(FX_PRESETS_DIR):
            os.makedirs(FX_PRESETS_DIR)

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
        
        self.LANGUAGES = {
            "American English": "a",
            "British English": "b",
            "Spanish": "e",
            "French": "f",
            "Italian": "i",
            "Portuguese": "p",
            "Japanese": "j",
            "Chinese": "z",
        }
        
        self.VOICE_DB = {
            "a": ["af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa"],
            "b": ["bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis"],
            "e": ["ef_dora", "em_alex", "em_santa"],
            "f": ["ff_siwis"],
            "i": ["if_sara", "im_nicola"],
            "p": ["pf_dora", "pm_alex"],
            "j": ["jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro"],
            "z": ["zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zm_yunjian"]
        }
        
        self.lang_var = ctk.StringVar(value=self.settings.get("lang_code", "a"))
        
        # Determine initial standard voices based on lang
        self.standard_voices = self.VOICE_DB.get(self.lang_var.get(), [])
        if not self.standard_voices: # Fallback
             self.standard_voices = self.VOICE_DB["a"]

        self.voice_var = ctk.StringVar(value=self.settings.get("voice", "af_heart"))
        self.filename_var = ctk.StringVar(value=self.settings.get("filename", "output"))
        self.output_format_var = ctk.StringVar(value=self.settings.get("format", "wav"))
        self.output_dir_var = ctk.StringVar(value=self.settings.get("out_dir", "audio_output"))
        self.speed_var = ctk.DoubleVar(value=self.settings.get("speed", 1.0))
        self.volume_var = ctk.DoubleVar(value=self.settings.get("volume", 1.0))
        self.pitch_var = ctk.DoubleVar(value=self.settings.get("pitch", 0.0))
        self.num_threads_var = ctk.IntVar(value=self.settings.get("num_threads", 1))
        self.split_pattern_var = ctk.StringVar(value=self.settings.get("split_pattern", r"\n+"))
        
        self.separate_files = ctk.BooleanVar(value=self.settings.get("separate", True))
        self.combine_post = ctk.BooleanVar(value=self.settings.get("combine", True))
        self.export_subtitles = ctk.BooleanVar(value=self.settings.get("export_subtitles", False))
        self.caching_enabled = ctk.BooleanVar(value=self.settings.get("caching", True))
        self.jit_enabled = ctk.BooleanVar(value=self.settings.get("jit_enabled", False))
        self.normalize_audio = ctk.BooleanVar(value=self.settings.get("normalize", False))
        self.trim_silence = ctk.BooleanVar(value=self.settings.get("trim", False))
        self.apply_fx_var = ctk.BooleanVar(value=self.settings.get("apply_fx", True))
        self.timecode_format = "%Y%m%d%H%M%S"

        # FX Variables
        self.reverb_enabled = ctk.BooleanVar(value=self.settings.get("reverb_enabled", False))
        self.reverb_room_size = ctk.DoubleVar(value=self.settings.get("reverb_room_size", 0.5))
        self.reverb_wet_level = ctk.DoubleVar(value=self.settings.get("reverb_wet_level", 0.3))
        
        self.eq_bass = ctk.DoubleVar(value=self.settings.get("eq_bass", 0.0))
        self.eq_treble = ctk.DoubleVar(value=self.settings.get("eq_treble", 0.0))
        
        self.comp_enabled = ctk.BooleanVar(value=self.settings.get("comp_enabled", False))
        self.comp_threshold = ctk.DoubleVar(value=self.settings.get("comp_threshold", -20.0))
        self.comp_ratio = ctk.DoubleVar(value=self.settings.get("comp_ratio", 4.0))
        self.comp_attack = ctk.DoubleVar(value=self.settings.get("comp_attack", 1.0))
        self.comp_release = ctk.DoubleVar(value=self.settings.get("comp_release", 100.0))

        # Reverb Extended
        self.reverb_damping = ctk.DoubleVar(value=self.settings.get("reverb_damping", 0.5))
        self.reverb_dry_level = ctk.DoubleVar(value=self.settings.get("reverb_dry_level", 1.0))
        self.reverb_width = ctk.DoubleVar(value=self.settings.get("reverb_width", 1.0))

        # New FX
        # Guitar
        self.distortion_enabled = ctk.BooleanVar(value=self.settings.get("distortion_enabled", False))
        self.distortion_drive = ctk.DoubleVar(value=self.settings.get("distortion_drive", 25.0))
        
        self.chorus_enabled = ctk.BooleanVar(value=self.settings.get("chorus_enabled", False))
        self.chorus_rate = ctk.DoubleVar(value=self.settings.get("chorus_rate", 1.0))
        self.chorus_depth = ctk.DoubleVar(value=self.settings.get("chorus_depth", 0.25))
        self.chorus_mix = ctk.DoubleVar(value=self.settings.get("chorus_mix", 0.5))
        
        self.phaser_enabled = ctk.BooleanVar(value=self.settings.get("phaser_enabled", False))
        self.phaser_rate = ctk.DoubleVar(value=self.settings.get("phaser_rate", 1.0))
        self.phaser_depth = ctk.DoubleVar(value=self.settings.get("phaser_depth", 0.5))
        self.phaser_mix = ctk.DoubleVar(value=self.settings.get("phaser_mix", 0.5))
        
        self.clipping_enabled = ctk.BooleanVar(value=self.settings.get("clipping_enabled", False))
        self.clipping_thresh = ctk.DoubleVar(value=self.settings.get("clipping_thresh", -6.0))
        
        # Quality
        self.bitcrush_enabled = ctk.BooleanVar(value=self.settings.get("bitcrush_enabled", False))
        self.bitcrush_depth = ctk.DoubleVar(value=self.settings.get("bitcrush_depth", 8.0))
        
        self.gsm_enabled = ctk.BooleanVar(value=self.settings.get("gsm_enabled", False))
        
        # Filters
        self.highpass_enabled = ctk.BooleanVar(value=self.settings.get("highpass_enabled", False))
        self.highpass_freq = ctk.DoubleVar(value=self.settings.get("highpass_freq", 50.0))
        
        self.lowpass_enabled = ctk.BooleanVar(value=self.settings.get("lowpass_enabled", False))
        self.lowpass_freq = ctk.DoubleVar(value=self.settings.get("lowpass_freq", 10000.0))

        # Spatial
        self.delay_enabled = ctk.BooleanVar(value=self.settings.get("delay_enabled", False))
        self.delay_time = ctk.DoubleVar(value=self.settings.get("delay_time", 0.5))
        self.delay_feedback = ctk.DoubleVar(value=self.settings.get("delay_feedback", 0.0))
        self.delay_mix = ctk.DoubleVar(value=self.settings.get("delay_mix", 0.5))
        
        # Pitch
        self.pitch_shift_enabled = ctk.BooleanVar(value=self.settings.get("pitch_shift_enabled", False))
        self.pitch_shift_semitones = ctk.DoubleVar(value=self.settings.get("pitch_shift_semitones", 0.0))
        
        # Dynamics
        self.limiter_enabled = ctk.BooleanVar(value=self.settings.get("limiter_enabled", False))
        self.limiter_threshold = ctk.DoubleVar(value=self.settings.get("limiter_threshold", -1.0))
        self.limiter_release = ctk.DoubleVar(value=self.settings.get("limiter_release", 100.0))
        
        self.gain_enabled = ctk.BooleanVar(value=self.settings.get("gain_enabled", False))
        self.gain_db = ctk.DoubleVar(value=self.settings.get("gain_db", 0.0))

        # Mixing Variables
        self.mix_lang_a_var = ctk.StringVar(value="a")
        self.mix_lang_b_var = ctk.StringVar(value="a")
        self.preview_lang_var = ctk.StringVar(value="a")
        
        self.mix_voice_a_var = ctk.StringVar(value=self.VOICE_DB["a"][0])
        self.mix_voice_b_var = ctk.StringVar(value=self.VOICE_DB["a"][1])
        self.mix_ratio_var = ctk.DoubleVar(value=0.5)
        self.mix_op_var = ctk.StringVar(value="mix")
        self.mix_name_var = ctk.StringVar()

        # Setup Auto-save Traces
        self.setup_autosave()

        self.create_widgets()
        
        # Init Pipeline
        self.status_label.configure(text="Initializing engine...")
        self.engine.worker.run_coro(self.engine.init_pipeline_async(self.lang_var.get()))

    def get_all_voices(self, lang_code=None):
        if lang_code is None:
            lang_code = self.lang_var.get()
        
        standard = self.VOICE_DB.get(lang_code, [])
        custom = []
        if os.path.exists("custom_voices"):
            custom = [f[:-3] for f in os.listdir("custom_voices") if f.endswith(".pt")]
        return sorted(standard + custom)


    def setup_autosave(self):
        vars_to_trace = [
            self.lang_var,
            self.voice_var, self.filename_var, self.output_format_var, self.output_dir_var,
            self.speed_var, self.volume_var, self.pitch_var,
            self.num_threads_var, self.split_pattern_var,
            self.separate_files, self.combine_post, self.export_subtitles, self.caching_enabled,
            self.normalize_audio, self.trim_silence, self.apply_fx_var,
            self.reverb_enabled, self.reverb_room_size, self.reverb_wet_level, self.reverb_damping, self.reverb_dry_level, self.reverb_width,
            self.eq_bass, self.eq_treble,
            self.comp_enabled, self.comp_threshold, self.comp_ratio, self.comp_attack, self.comp_release,
            self.distortion_enabled, self.distortion_drive,
            self.chorus_enabled, self.chorus_rate, self.chorus_depth, self.chorus_mix,
            self.phaser_enabled, self.phaser_rate, self.phaser_depth, self.phaser_mix,
            self.clipping_enabled, self.clipping_thresh,
            self.bitcrush_enabled, self.bitcrush_depth,
            self.gsm_enabled,
            self.highpass_enabled, self.highpass_freq,
            self.lowpass_enabled, self.lowpass_freq,
            self.delay_enabled, self.delay_time, self.delay_feedback, self.delay_mix,
            self.pitch_shift_enabled, self.pitch_shift_semitones,
            self.limiter_enabled, self.limiter_threshold, self.limiter_release,
            self.gain_enabled, self.gain_db
        ]
        for v in vars_to_trace:
            v.trace_add("write", self.schedule_save)
            
        # Also trigger voice list update when lang changes
        self.lang_var.trace_add("write", self.on_lang_change)
        
        # Mix tab traces
        self.mix_lang_a_var.trace_add("write", self.on_mix_lang_a_change)
        self.mix_lang_b_var.trace_add("write", self.on_mix_lang_b_change)

    def on_lang_change(self, *args):
        code = self.lang_var.get()
        self.standard_voices = self.VOICE_DB.get(code, self.VOICE_DB["a"])
        # Update generation combo
        if hasattr(self, 'voice_combo'):
            self.voice_combo.configure(values=self.get_all_voices(code))
        
        # Set default voice for this language if current voice is invalid
        if self.voice_var.get() not in self.VOICE_DB.get(code, []):
            if self.VOICE_DB.get(code, []):
                self.voice_var.set(self.VOICE_DB[code][0])

    def schedule_save(self, *args):
        if self.save_timer:
            self.after_cancel(self.save_timer)
        self.save_timer = self.after(1000, self.save_settings)

    def load_settings(self):
        defaults = {
            "appearance": "Dark", 
            "scaling": "100%",
            "lang_code": "a",
            "voice": "af_heart",
            "filename": "output",
            "format": "wav",
            "out_dir": "audio_output",
            "speed": 1.0,
            "volume": 1.0,
            "pitch": 0.0,
            "num_threads": 1,
            "split_pattern": r"\n+",
            "separate": True,
            "combine": True,
            "export_subtitles": False,
            "caching": True,
            "jit_enabled": False,
            "normalize": False,
            "trim": False,
            "apply_fx": True,
            "reverb_enabled": False,
            "reverb_room_size": 0.5,
            "reverb_wet_level": 0.3,
            "reverb_damping": 0.5,
            "reverb_dry_level": 1.0,
            "reverb_width": 1.0,
            "eq_bass": 0.0,
            "eq_treble": 0.0,
            "comp_enabled": False,
            "comp_threshold": -20.0,
            "comp_ratio": 4.0,
            "comp_attack": 1.0,
            "comp_release": 100.0,
            "distortion_enabled": False,
            "distortion_drive": 25.0,
            "chorus_enabled": False,
            "chorus_rate": 1.0,
            "chorus_depth": 0.25,
            "chorus_mix": 0.5,
            "phaser_enabled": False,
            "phaser_rate": 1.0,
            "phaser_depth": 0.5,
            "phaser_mix": 0.5,
            "clipping_enabled": False,
            "clipping_thresh": -6.0,
            "bitcrush_enabled": False,
            "bitcrush_depth": 8.0,
            "gsm_enabled": False,
            "highpass_enabled": False,
            "highpass_freq": 50.0,
            "lowpass_enabled": False,
            "lowpass_freq": 10000.0,
            "delay_enabled": False,
            "delay_time": 0.5,
            "delay_feedback": 0.0,
            "delay_mix": 0.5,
            "pitch_shift_enabled": False,
            "pitch_shift_semitones": 0.0,
            "limiter_enabled": False,
            "limiter_threshold": -1.0,
            "limiter_release": 100.0,
            "gain_enabled": False,
            "gain_db": 0.0,
            "lexicon": {}
        }
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    return {**defaults, **json.load(f)}
            except Exception:
                pass
        return defaults

    def save_settings(self):
        if self.save_timer:
            self.after_cancel(self.save_timer)
            self.save_timer = None
            
        if hasattr(self, 'voice_var'):
            self.settings['lang_code'] = self.lang_var.get()
            self.settings['voice'] = self.voice_var.get()
            self.settings['filename'] = self.filename_var.get()
            self.settings['format'] = self.output_format_var.get()
            self.settings['out_dir'] = self.output_dir_var.get()
            self.settings['speed'] = self.speed_var.get()
            self.settings['volume'] = self.volume_var.get()
            self.settings['pitch'] = self.pitch_var.get()
            self.settings['num_threads'] = self.num_threads_var.get()
            self.settings['split_pattern'] = self.split_pattern_var.get()
            self.settings['separate'] = self.separate_files.get()
            self.settings['combine'] = self.combine_post.get()
            self.settings['export_subtitles'] = self.export_subtitles.get()
            self.settings['caching'] = self.caching_enabled.get()
            self.settings['jit_enabled'] = self.jit_enabled.get()
            self.settings['normalize'] = self.normalize_audio.get()
            self.settings['trim'] = self.trim_silence.get()
            self.settings['apply_fx'] = self.apply_fx_var.get()
            self.settings['reverb_enabled'] = self.reverb_enabled.get()
            self.settings['reverb_room_size'] = self.reverb_room_size.get()
            self.settings['reverb_wet_level'] = self.reverb_wet_level.get()
            self.settings['reverb_damping'] = self.reverb_damping.get()
            self.settings['reverb_dry_level'] = self.reverb_dry_level.get()
            self.settings['reverb_width'] = self.reverb_width.get()
            
            self.settings['eq_bass'] = self.eq_bass.get()
            self.settings['eq_treble'] = self.eq_treble.get()
            
            self.settings['comp_enabled'] = self.comp_enabled.get()
            self.settings['comp_threshold'] = self.comp_threshold.get()
            self.settings['comp_ratio'] = self.comp_ratio.get()
            self.settings['comp_attack'] = self.comp_attack.get()
            self.settings['comp_release'] = self.comp_release.get()
            
            self.settings['distortion_enabled'] = self.distortion_enabled.get()
            self.settings['distortion_drive'] = self.distortion_drive.get()
            
            self.settings['chorus_enabled'] = self.chorus_enabled.get()
            self.settings['chorus_rate'] = self.chorus_rate.get()
            self.settings['chorus_depth'] = self.chorus_depth.get()
            self.settings['chorus_mix'] = self.chorus_mix.get()
            
            self.settings['phaser_enabled'] = self.phaser_enabled.get()
            self.settings['phaser_rate'] = self.phaser_rate.get()
            self.settings['phaser_depth'] = self.phaser_depth.get()
            self.settings['phaser_mix'] = self.phaser_mix.get()
            
            self.settings['clipping_enabled'] = self.clipping_enabled.get()
            self.settings['clipping_thresh'] = self.clipping_thresh.get()
            
            self.settings['bitcrush_enabled'] = self.bitcrush_enabled.get()
            self.settings['bitcrush_depth'] = self.bitcrush_depth.get()
            
            self.settings['gsm_enabled'] = self.gsm_enabled.get()
            
            self.settings['highpass_enabled'] = self.highpass_enabled.get()
            self.settings['highpass_freq'] = self.highpass_freq.get()
            
            self.settings['lowpass_enabled'] = self.lowpass_enabled.get()
            self.settings['lowpass_freq'] = self.lowpass_freq.get()
            
            self.settings['delay_enabled'] = self.delay_enabled.get()
            self.settings['delay_time'] = self.delay_time.get()
            self.settings['delay_feedback'] = self.delay_feedback.get()
            self.settings['delay_mix'] = self.delay_mix.get()
            
            self.settings['pitch_shift_enabled'] = self.pitch_shift_enabled.get()
            self.settings['pitch_shift_semitones'] = self.pitch_shift_semitones.get()
            
            self.settings['limiter_enabled'] = self.limiter_enabled.get()
            self.settings['limiter_threshold'] = self.limiter_threshold.get()
            self.settings['limiter_release'] = self.limiter_release.get()
            
            self.settings['gain_enabled'] = self.gain_enabled.get()
            self.settings['gain_db'] = self.gain_db.get()

        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
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
        except Exception:
            ctk.set_widget_scaling(1.0)

    def create_widgets(self):
        # Header
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)

        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10,0))
        
        ctk.CTkLabel(header_frame, text="Kokoro TTS", font=("Roboto", 20, "bold")).pack(side="left", padx=5)
        ctk.CTkButton(header_frame, text="⚙ Settings", width=80, height=28, command=self.open_settings).pack(side="right")

        # Main Tabs
        self.main_tabs = ctk.CTkTabview(self)
        self.main_tabs.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        
        gen_tab = self.main_tabs.add("Generate Audio")
        self.build_generation_tab(gen_tab)
        
        mix_tab = self.main_tabs.add("Custom Voice")
        self.build_mixing_tab(mix_tab)

        fx_tab = self.main_tabs.add("Audio FX")
        self.build_fx_tab(fx_tab)

        lex_tab = self.main_tabs.add("Lexicon")
        self.build_lexicon_tab(lex_tab)

        # Actions (Global)
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
        
        btn_txt = "Start Real-time JIT" if self.jit_enabled.get() else "Start Generation"
        self.start_btn = ctk.CTkButton(btn_frame, text=btn_txt, command=self.start_conversion, height=40, font=("Roboto", 14, "bold"))
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
        
        # Caching
        ctk.CTkLabel(frame, text="Generation Cache:", font=("Roboto", 14, "bold")).pack(anchor="w", pady=(15, 5))
        ctk.CTkCheckBox(frame, text="Enable Generation Caching", variable=self.caching_enabled).pack(anchor="w", pady=5)
        
        # JIT
        ctk.CTkLabel(frame, text="Real-time / JIT:", font=("Roboto", 14, "bold")).pack(anchor="w", pady=(15, 5))
        ctk.CTkCheckBox(frame, text="Enable JIT Generation (Streaming)", variable=self.jit_enabled, command=self.on_jit_toggle).pack(anchor="w", pady=5)
        
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

    def on_jit_toggle(self):
        if self.jit_enabled.get():
            self.start_btn.configure(text="Start Real-time JIT")
        else:
            self.start_btn.configure(text="Start Generation")
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
        except Exception:
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
        self.after(0, lambda: self.status_label.configure(text=msg.split('\n')[0], text_color=color))
        
        if is_error and "pip install" in msg:
            self.after(0, lambda: messagebox.showerror("Missing Dependencies", msg))

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
        if not self.engine.pipeline:
            messagebox.showinfo("Wait", "Engine is initializing... please wait 2 seconds and try again.")
            return

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
                except Exception:
                    pass
        
        if not text_data:
            text_data = "This is a sample audio preview using the Koh-koh-ro Tea-Tea-S engine. It demonstrates the voice quality and speed settings."
            
        preview_text = text_data
        if len(preview_text) > 1000: # Slightly larger cap for raw text before engine handles it
             preview_text = preview_text[:1000]
             
        # Config
        voice = self.voice_var.get()
        speed = self.speed_var.get()
        
        extra_config = {
            'volume': self.volume_var.get(),
            'pitch': self.pitch_var.get(),
            'normalize': self.normalize_audio.get(),
            'trim_silence': self.trim_silence.get(),
            'lexicon': self.settings.get('lexicon', {})
        }
        
        if self.apply_fx_var.get():
            extra_config.update({
                'reverb_enabled': self.reverb_enabled.get(),
                'reverb_room_size': self.reverb_room_size.get(),
                'reverb_wet_level': self.reverb_wet_level.get(),
                'reverb_damping': self.reverb_damping.get(),
                'reverb_dry_level': self.reverb_dry_level.get(),
                'reverb_width': self.reverb_width.get(),
                'eq_bass': self.eq_bass.get(),
                'eq_treble': self.eq_treble.get(),
                'comp_enabled': self.comp_enabled.get(),
                'comp_threshold': self.comp_threshold.get(),
                'comp_ratio': self.comp_ratio.get(),
                'comp_attack': self.comp_attack.get(),
                'comp_release': self.comp_release.get(),
                'distortion_enabled': self.distortion_enabled.get(),
                'distortion_drive': self.distortion_drive.get(),
                'chorus_enabled': self.chorus_enabled.get(),
                'chorus_rate': self.chorus_rate.get(),
                'chorus_depth': self.chorus_depth.get(),
                'chorus_mix': self.chorus_mix.get(),
                'phaser_enabled': self.phaser_enabled.get(),
                'phaser_rate': self.phaser_rate.get(),
                'phaser_depth': self.phaser_depth.get(),
                'phaser_mix': self.phaser_mix.get(),
                'clipping_enabled': self.clipping_enabled.get(),
                'clipping_thresh': self.clipping_thresh.get(),
                'bitcrush_enabled': self.bitcrush_enabled.get(),
                'bitcrush_depth': self.bitcrush_depth.get(),
                'gsm_enabled': self.gsm_enabled.get(),
                'highpass_enabled': self.highpass_enabled.get(),
                'highpass_freq': self.highpass_freq.get(),
                'lowpass_enabled': self.lowpass_enabled.get(),
                'lowpass_freq': self.lowpass_freq.get(),
                'delay_enabled': self.delay_enabled.get(),
                'delay_time': self.delay_time.get(),
                'delay_feedback': self.delay_feedback.get(),
                'delay_mix': self.delay_mix.get(),
                'pitch_shift_enabled': self.pitch_shift_enabled.get(),
                'pitch_shift_semitones': self.pitch_shift_semitones.get(),
                'limiter_enabled': self.limiter_enabled.get(),
                'limiter_threshold': self.limiter_threshold.get(),
                'limiter_release': self.limiter_release.get(),
                'gain_enabled': self.gain_enabled.get(),
                'gain_db': self.gain_db.get()
            })
        
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
                        playback.play(tmp_path)
                        self.after(3000, lambda: self.status_label.configure(text="Ready", text_color="gray"))
                    else:
                        self.status_label.configure(text="Preview failed.", text_color="red")
                except Exception as e:
                    self.status_label.configure(text=f"Preview error: {e}", text_color="red")
            
            self.after(0, _ui_update)
        
        future = self.engine.worker.run_coro(self.engine.generate_preview(preview_text, voice, speed, tmp_path, extra_config, lang_code=self.lang_var.get()))
        future.add_done_callback(_on_preview_done)

    def start_conversion(self):
        # 0. Validate Threads
        try:
            val = int(self.num_threads_var.get())
            if val < 1: val = 1
            self.num_threads_var.set(val)
        except Exception:
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
            'lang_code': self.lang_var.get(),
            'voice': self.voice_var.get(),
            'speed': self.speed_var.get(),
            'split_pattern': self.split_pattern_var.get(),
            'filename': self.filename_var.get(),
            'format': self.output_format_var.get(),
            'out_dir': self.output_dir_var.get(),
            'separate': self.separate_files.get(),
            'combine': self.combine_post.get(),
            'export_subtitles': self.export_subtitles.get(),
            'caching': self.caching_enabled.get(),
            'time_id': time.strftime(self.timecode_format),
            'num_threads': self.num_threads_var.get(),
            'volume': self.volume_var.get(),
            'pitch': self.pitch_var.get(),
            'normalize': self.normalize_audio.get(),
            'trim_silence': self.trim_silence.get(),
            'lexicon': self.settings.get('lexicon', {})
        }
        
        if self.apply_fx_var.get():
            config.update({
                'reverb_enabled': self.reverb_enabled.get(),
                'reverb_room_size': self.reverb_room_size.get(),
                'reverb_wet_level': self.reverb_wet_level.get(),
                'reverb_damping': self.reverb_damping.get(),
                'reverb_dry_level': self.reverb_dry_level.get(),
                'reverb_width': self.reverb_width.get(),
                'eq_bass': self.eq_bass.get(),
                'eq_treble': self.eq_treble.get(),
                'comp_enabled': self.comp_enabled.get(),
                'comp_threshold': self.comp_threshold.get(),
                'comp_ratio': self.comp_ratio.get(),
                'comp_attack': self.comp_attack.get(),
                'comp_release': self.comp_release.get(),
                'distortion_enabled': self.distortion_enabled.get(),
                'distortion_drive': self.distortion_drive.get(),
                'chorus_enabled': self.chorus_enabled.get(),
                'chorus_rate': self.chorus_rate.get(),
                'chorus_depth': self.chorus_depth.get(),
                'chorus_mix': self.chorus_mix.get(),
                'phaser_enabled': self.phaser_enabled.get(),
                'phaser_rate': self.phaser_rate.get(),
                'phaser_depth': self.phaser_depth.get(),
                'phaser_mix': self.phaser_mix.get(),
                'clipping_enabled': self.clipping_enabled.get(),
                'clipping_thresh': self.clipping_thresh.get(),
                'bitcrush_enabled': self.bitcrush_enabled.get(),
                'bitcrush_depth': self.bitcrush_depth.get(),
                'gsm_enabled': self.gsm_enabled.get(),
                'highpass_enabled': self.highpass_enabled.get(),
                'highpass_freq': self.highpass_freq.get(),
                'lowpass_enabled': self.lowpass_enabled.get(),
                'lowpass_freq': self.lowpass_freq.get(),
                'delay_enabled': self.delay_enabled.get(),
                'delay_time': self.delay_time.get(),
                'delay_feedback': self.delay_feedback.get(),
                'delay_mix': self.delay_mix.get(),
                'pitch_shift_enabled': self.pitch_shift_enabled.get(),
                'pitch_shift_semitones': self.pitch_shift_semitones.get(),
                'limiter_enabled': self.limiter_enabled.get(),
                'limiter_threshold': self.limiter_threshold.get(),
                'limiter_release': self.limiter_release.get(),
                'gain_enabled': self.gain_enabled.get(),
                'gain_db': self.gain_db.get()
            })

        # 3. Start
        self.set_ui_state(True)
        self.progress_bar.set(0)
        
        if self.jit_enabled.get():
            self.engine.start_jit_conversion(text_data, config)
        else:
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
