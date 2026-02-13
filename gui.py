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
FX_PRESETS_DIR = os.path.join(PRESETS_DIR, "fx")

class TTSApp(ctk.CTk):
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

    def on_mix_lang_a_change(self, *args):
        code = self.mix_lang_a_var.get()
        if hasattr(self, 'mix_combo_a'):
            voices = self.get_all_voices(code)
            self.mix_combo_a.configure(values=voices)
            if self.mix_voice_a_var.get() not in voices:
                self.mix_voice_a_var.set(voices[0])

    def on_mix_lang_b_change(self, *args):
        code = self.mix_lang_b_var.get()
        if hasattr(self, 'mix_combo_b'):
            voices = self.get_all_voices(code)
            self.mix_combo_b.configure(values=voices)
            if self.mix_voice_b_var.get() not in voices:
                self.mix_voice_b_var.set(voices[0])

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
                "trim": self.trim_silence.get(),
                "format": self.output_format_var.get(),
                "apply_fx": self.apply_fx_var.get(),
                "fx_preset": self.gen_fx_combo.get()
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
                messagebox.showerror("Error", f"Failed to load preset: {e}")

    # --- FX Preset Management ---

    def refresh_fx_presets(self):
        presets = ["Select FX Preset..."]
        if os.path.exists(FX_PRESETS_DIR):
            files = [f for f in os.listdir(FX_PRESETS_DIR) if f.endswith(".json")]
            presets.extend([f[:-5] for f in files]) # Remove .json
        
        # Update FX Tab Combo
        if hasattr(self, 'fx_preset_combo'):
            self.fx_preset_combo.configure(values=presets)
            self.fx_preset_combo.set("Select FX Preset...")
            
        # Update Gen Tab Combo
        if hasattr(self, 'gen_fx_combo'):
            self.gen_fx_combo.configure(values=presets)
            self.gen_fx_combo.set("Select FX Preset...")

    def save_fx_preset_dialog(self):
        dialog = ctk.CTkInputDialog(text="Enter FX preset name:", title="Save FX Preset")
        name = dialog.get_input()
        if name:
            name = re.sub(r'[<>:"/\\|?*]', '', name).strip()
            if not name: return
            
            data = {
                "reverb_enabled": self.reverb_enabled.get(),
                "reverb_room_size": self.reverb_room_size.get(),
                "reverb_wet_level": self.reverb_wet_level.get(),
                "reverb_damping": self.reverb_damping.get(),
                "reverb_dry_level": self.reverb_dry_level.get(),
                "reverb_width": self.reverb_width.get(),
                "eq_bass": self.eq_bass.get(),
                "eq_treble": self.eq_treble.get(),
                "comp_enabled": self.comp_enabled.get(),
                "comp_threshold": self.comp_threshold.get(),
                "comp_ratio": self.comp_ratio.get(),
                "comp_attack": self.comp_attack.get(),
                "comp_release": self.comp_release.get(),
                "distortion_enabled": self.distortion_enabled.get(),
                "distortion_drive": self.distortion_drive.get(),
                "chorus_enabled": self.chorus_enabled.get(),
                "chorus_rate": self.chorus_rate.get(),
                "chorus_depth": self.chorus_depth.get(),
                "chorus_mix": self.chorus_mix.get(),
                "phaser_enabled": self.phaser_enabled.get(),
                "phaser_rate": self.phaser_rate.get(),
                "phaser_depth": self.phaser_depth.get(),
                "phaser_mix": self.phaser_mix.get(),
                "clipping_enabled": self.clipping_enabled.get(),
                "clipping_thresh": self.clipping_thresh.get(),
                "bitcrush_enabled": self.bitcrush_enabled.get(),
                "bitcrush_depth": self.bitcrush_depth.get(),
                "gsm_enabled": self.gsm_enabled.get(),
                "highpass_enabled": self.highpass_enabled.get(),
                "highpass_freq": self.highpass_freq.get(),
                "lowpass_enabled": self.lowpass_enabled.get(),
                "lowpass_freq": self.lowpass_freq.get(),
                "delay_enabled": self.delay_enabled.get(),
                "delay_time": self.delay_time.get(),
                "delay_feedback": self.delay_feedback.get(),
                "delay_mix": self.delay_mix.get(),
                "pitch_shift_enabled": self.pitch_shift_enabled.get(),
                "pitch_shift_semitones": self.pitch_shift_semitones.get(),
                "limiter_enabled": self.limiter_enabled.get(),
                "limiter_threshold": self.limiter_threshold.get(),
                "limiter_release": self.limiter_release.get(),
                "gain_enabled": self.gain_enabled.get(),
                "gain_db": self.gain_db.get()
            }
            
            fpath = os.path.join(FX_PRESETS_DIR, f"{name}.json")
            try:
                with open(fpath, "w") as f:
                    json.dump(data, f, indent=4)
                messagebox.showinfo("Saved", f"FX Preset '{name}' saved.")
                self.refresh_fx_presets()
                if hasattr(self, 'fx_preset_combo'): self.fx_preset_combo.set(name)
                if hasattr(self, 'gen_fx_combo'): self.gen_fx_combo.set(name)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save FX preset: {e}")

    def load_fx_preset(self, name):
        if name == "Select FX Preset...": return
        
        fpath = os.path.join(FX_PRESETS_DIR, f"{name}.json")
        if os.path.exists(fpath):
            try:
                with open(fpath, "r") as f:
                    data = json.load(f)
                
                if "reverb_enabled" in data: self.reverb_enabled.set(data["reverb_enabled"])
                if "reverb_room_size" in data: self.reverb_room_size.set(data["reverb_room_size"])
                if "reverb_wet_level" in data: self.reverb_wet_level.set(data["reverb_wet_level"])
                if "reverb_damping" in data: self.reverb_damping.set(data["reverb_damping"])
                if "reverb_dry_level" in data: self.reverb_dry_level.set(data["reverb_dry_level"])
                if "reverb_width" in data: self.reverb_width.set(data["reverb_width"])
                
                if "eq_bass" in data: self.eq_bass.set(data["eq_bass"])
                if "eq_treble" in data: self.eq_treble.set(data["eq_treble"])
                
                if "comp_enabled" in data: self.comp_enabled.set(data["comp_enabled"])
                if "comp_threshold" in data: self.comp_threshold.set(data["comp_threshold"])
                if "comp_ratio" in data: self.comp_ratio.set(data["comp_ratio"])
                if "comp_attack" in data: self.comp_attack.set(data["comp_attack"])
                if "comp_release" in data: self.comp_release.set(data["comp_release"])

                if "distortion_enabled" in data: self.distortion_enabled.set(data["distortion_enabled"])
                if "distortion_drive" in data: self.distortion_drive.set(data["distortion_drive"])
                
                if "chorus_enabled" in data: self.chorus_enabled.set(data["chorus_enabled"])
                if "chorus_rate" in data: self.chorus_rate.set(data["chorus_rate"])
                if "chorus_depth" in data: self.chorus_depth.set(data["chorus_depth"])
                if "chorus_mix" in data: self.chorus_mix.set(data["chorus_mix"])
                
                if "phaser_enabled" in data: self.phaser_enabled.set(data["phaser_enabled"])
                if "phaser_rate" in data: self.phaser_rate.set(data["phaser_rate"])
                if "phaser_depth" in data: self.phaser_depth.set(data["phaser_depth"])
                if "phaser_mix" in data: self.phaser_mix.set(data["phaser_mix"])
                
                if "clipping_enabled" in data: self.clipping_enabled.set(data["clipping_enabled"])
                if "clipping_thresh" in data: self.clipping_thresh.set(data["clipping_thresh"])
                
                if "bitcrush_enabled" in data: self.bitcrush_enabled.set(data["bitcrush_enabled"])
                if "bitcrush_depth" in data: self.bitcrush_depth.set(data["bitcrush_depth"])
                
                if "gsm_enabled" in data: self.gsm_enabled.set(data["gsm_enabled"])
                
                if "highpass_enabled" in data: self.highpass_enabled.set(data["highpass_enabled"])
                if "highpass_freq" in data: self.highpass_freq.set(data["highpass_freq"])
                
                if "lowpass_enabled" in data: self.lowpass_enabled.set(data["lowpass_enabled"])
                if "lowpass_freq" in data: self.lowpass_freq.set(data["lowpass_freq"])
                
                if "delay_enabled" in data: self.delay_enabled.set(data["delay_enabled"])
                if "delay_time" in data: self.delay_time.set(data["delay_time"])
                if "delay_feedback" in data: self.delay_feedback.set(data["delay_feedback"])
                if "delay_mix" in data: self.delay_mix.set(data["delay_mix"])
                
                if "pitch_shift_enabled" in data: self.pitch_shift_enabled.set(data["pitch_shift_enabled"])
                if "pitch_shift_semitones" in data: self.pitch_shift_semitones.set(data["pitch_shift_semitones"])
                
                if "limiter_enabled" in data: self.limiter_enabled.set(data["limiter_enabled"])
                if "limiter_threshold" in data: self.limiter_threshold.set(data["limiter_threshold"])
                if "limiter_release" in data: self.limiter_release.set(data["limiter_release"])
                
                if "gain_enabled" in data: self.gain_enabled.set(data["gain_enabled"])
                if "gain_db" in data: self.gain_db.set(data["gain_db"])
                
                self.update_fx_labels()
                
                # Sync Combos
                if hasattr(self, 'fx_preset_combo'): self.fx_preset_combo.set(name)
                if hasattr(self, 'gen_fx_combo'): self.gen_fx_combo.set(name)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load FX preset: {e}")

    def refresh_voice_lists(self):
        # Update Gen Tab Combo
        if hasattr(self, 'voice_combo'):
            self.voice_combo.configure(values=self.get_all_voices(self.lang_var.get()))
            
        # Update Mix Tab Combos
        self.on_mix_lang_a_change()
        self.on_mix_lang_b_change()
            
        # Update Custom List
        if hasattr(self, 'custom_list_frame'):
            for widget in self.custom_list_frame.winfo_children():
                widget.destroy()
                
            all_voices = self.get_all_voices(self.lang_var.get())
            custom = [f[:-3] for f in os.listdir("custom_voices") if f.endswith(".pt")]
            if not custom:
                ctk.CTkLabel(self.custom_list_frame, text="No custom voices found.", text_color="gray").pack(pady=5)
            else:
                for cv in sorted(custom):
                    row = ctk.CTkFrame(self.custom_list_frame)
                    row.pack(fill="x", pady=2)
                    ctk.CTkLabel(row, text=cv).pack(side="left", padx=5)
                    ctk.CTkButton(row, text="X", width=30, fg_color="#c42b1c", command=lambda v=cv: self.delete_custom_voice(v)).pack(side="right", padx=5)

    def delete_custom_voice(self, name):
        if messagebox.askyesno("Confirm", f"Delete voice '{name}'?"):
            try:
                path = os.path.join("custom_voices", f"{name}.pt")
                if os.path.exists(path):
                    os.remove(path)
                    self.refresh_voice_lists()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete: {e}")

    def preview_mix(self):
        v1 = self.mix_voice_a_var.get()
        v2 = self.mix_voice_b_var.get()
        ratio = self.mix_ratio_var.get()
        op = self.mix_op_var.get()
        preview_lang = self.preview_lang_var.get()
        
        preview_text = "This is a preview of your custom mixed voice."
        if preview_lang == 'f': preview_text = "Ceci est un aperçu de votre voix personnalisée."
        elif preview_lang == 'e': preview_text = "Esta es una vista previa de su voz personalizada."
        elif preview_lang == 'i': preview_text = "Questa è un'anteprima della tua voce personalizzata."
        elif preview_lang == 'p': preview_text = "Esta é uma prévia da sua voz personalizada."
        elif preview_lang == 'j': preview_text = "これはカスタム合成音声のプレビューです。"
        elif preview_lang == 'z': preview_text = "这是您的自定义混合语音预览。"
        
        # Temp voice name and file
        import tempfile
        tmp_voice_name = "_tmp_mix_preview"
        tmp_audio_path = os.path.join(tempfile.gettempdir(), "kokoro_mix_preview.wav")
        
        self.mix_status_label.configure(text="Generating preview...", text_color="blue")
        
        async def _run_preview():
            # 1. Mix to a temporary file (we ignore the file for preview, use tensor)
            success, msg, tensor = await self.engine.mix_voices(v1, v2, ratio, tmp_voice_name, op=op)
            if not success:
                return False, msg
            
            # 2. Generate audio using that mixed voice tensor and target preview language
            success = await self.engine.generate_preview(preview_text, tmp_voice_name, 1.0, tmp_audio_path, voice_tensor=tensor, lang_code=preview_lang)
            
            # 3. Cleanup temp voice file
            try:
                p = os.path.join("custom_voices", f"{tmp_voice_name}.pt")
                if os.path.exists(p): os.remove(p)
            except: pass
            
            return success, ""

        def _on_done(future):
            try:
                success, err = future.result()
                if success:
                    self.after(0, lambda: self.mix_status_label.configure(text="Playing preview...", text_color="green"))
                    winsound.PlaySound(tmp_audio_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                else:
                    self.after(0, lambda: self.mix_status_label.configure(text=f"Preview failed: {err}", text_color="red"))
            except Exception as e:
                self.after(0, lambda: self.mix_status_label.configure(text=f"Error: {e}", text_color="red"))

        future = self.engine.worker.run_coro(_run_preview())
        future.add_done_callback(_on_done)

    def mix_voice_action(self):
        v1 = self.mix_voice_a_var.get()
        v2 = self.mix_voice_b_var.get()
        ratio = self.mix_ratio_var.get()
        op = self.mix_op_var.get()
        name = self.mix_name_var.get().strip()
        
        if not name:
            messagebox.showwarning("Error", "Please enter a name for the new voice.")
            return
        
        if not re.match(r'^[a-zA-Z0-9_-]+$', name):
             messagebox.showwarning("Error", "Invalid name. Use alphanumeric, _, - only.")
             return
             
        if name in self.get_all_voices():
            if not messagebox.askyesno("Overwrite", f"Voice '{name}' exists. Overwrite?"):
                return
        
        self.mix_status_label.configure(text="Mixing...", text_color="blue")
        self.set_ui_state(True) # Reuse existing lock
        
        def _done(future):
            self.after(0, lambda: self.set_ui_state(False))
            try:
                success, msg, _ = future.result()
                if success:
                    self.after(0, lambda: self.mix_status_label.configure(text=f"Saved: {name}", text_color="green"))
                    self.after(0, self.refresh_voice_lists)
                else:
                    self.after(0, lambda: self.mix_status_label.configure(text=f"Error: {msg}", text_color="red"))
            except Exception as e:
                self.after(0, lambda: self.mix_status_label.configure(text=f"Error: {e}", text_color="red"))

        future = self.engine.worker.run_coro(self.engine.mix_voices(v1, v2, ratio, name, op=op))
        future.add_done_callback(_done)

    def build_mixing_tab(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        
        lang_display_map = {v: k for k, v in self.LANGUAGES.items()}
        
        # 1. Selection
        sel_frame = ctk.CTkFrame(parent)
        sel_frame.pack(fill="x", padx=10, pady=10)
        sel_frame.grid_columnconfigure(1, weight=1)
        sel_frame.grid_columnconfigure(2, weight=1)
        
        # Voice A Row
        ctk.CTkLabel(sel_frame, text="Voice A:").grid(row=0, column=0, padx=10, pady=5)
        
        def on_lang_a_ui(c): self.mix_lang_a_var.set(self.LANGUAGES[c])
        mix_lang_a_combo = ctk.CTkComboBox(sel_frame, values=list(self.LANGUAGES.keys()), command=on_lang_a_ui, width=150)
        mix_lang_a_combo.set(lang_display_map.get(self.mix_lang_a_var.get(), "American English"))
        mix_lang_a_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        self.mix_combo_a = ctk.CTkComboBox(sel_frame, variable=self.mix_voice_a_var)
        self.mix_combo_a.grid(row=0, column=2, sticky="ew", padx=5, pady=5)
        
        # Voice B Row
        ctk.CTkLabel(sel_frame, text="Voice B:").grid(row=1, column=0, padx=10, pady=5)
        
        def on_lang_b_ui(c): self.mix_lang_b_var.set(self.LANGUAGES[c])
        mix_lang_b_combo = ctk.CTkComboBox(sel_frame, values=list(self.LANGUAGES.keys()), command=on_lang_b_ui, width=150)
        mix_lang_b_combo.set(lang_display_map.get(self.mix_lang_b_var.get(), "American English"))
        mix_lang_b_combo.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        
        self.mix_combo_b = ctk.CTkComboBox(sel_frame, variable=self.mix_voice_b_var)
        self.mix_combo_b.grid(row=1, column=2, sticky="ew", padx=5, pady=5)
        
        # 2. Ratio & Operation
        ratio_frame = ctk.CTkFrame(parent)
        ratio_frame.pack(fill="x", padx=10, pady=10)
        
        op_frame = ctk.CTkFrame(ratio_frame, fg_color="transparent")
        op_frame.pack(fill="x", padx=20, pady=(10, 0))
        ctk.CTkLabel(op_frame, text="Operation:").pack(side="left", padx=5)
        
        def update_ratio_label(val=None):
            if val is None: val = self.mix_ratio_var.get()
            p = int(float(val) * 100)
            op = self.mix_op_var.get()
            if op == 'mix':
                self.ratio_label.configure(text=f"Mix: {100-p}% A / {p}% B", text_color=("black", "white"))
            elif op == 'divide':
                self.ratio_label.configure(text=f"Op: Divide | Influence: {p}%\n(Results are more likely to be unstable and VERY LOUD)", text_color="#E57373")
            else:
                self.ratio_label.configure(text=f"Op: {op.capitalize()} | Influence: {p}%", text_color=("black", "white"))

        ctk.CTkComboBox(op_frame, values=["mix", "add", "subtract", "multiply", "divide"], variable=self.mix_op_var, command=lambda _: update_ratio_label()).pack(side="left", padx=5)

        self.ratio_label = ctk.CTkLabel(ratio_frame, text="Mix: 50% A / 50% B")
        self.ratio_label.pack(pady=5)
        
        slider = ctk.CTkSlider(ratio_frame, from_=0.0, to=1.0, number_of_steps=100, variable=self.mix_ratio_var, command=update_ratio_label)
        slider.pack(fill="x", padx=20, pady=10)
        
        update_ratio_label()
        
        # 3. Preview Lang & Actions
        act_frame = ctk.CTkFrame(parent)
        act_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(act_frame, text="Preview Language:").grid(row=0, column=0, padx=10, pady=5)
        
        def on_prev_lang_ui(c): self.preview_lang_var.set(self.LANGUAGES[c])
        prev_lang_combo = ctk.CTkComboBox(act_frame, values=list(self.LANGUAGES.keys()), command=on_prev_lang_ui, width=150)
        prev_lang_combo.set(lang_display_map.get(self.preview_lang_var.get(), "American English"))
        prev_lang_combo.grid(row=0, column=1, padx=5, pady=5)
        
        ctk.CTkButton(act_frame, text="🔊 Preview", width=100, fg_color="#2B719E", command=self.preview_mix).grid(row=0, column=2, padx=10)
        
        # Save Row
        save_frame = ctk.CTkFrame(parent)
        save_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(save_frame, text="New Voice Name:").pack(side="left", padx=10)
        ctk.CTkEntry(save_frame, textvariable=self.mix_name_var).pack(side="left", fill="x", expand=True, padx=5)
        ctk.CTkButton(save_frame, text="Create & Save", command=self.mix_voice_action).pack(side="left", padx=10)
        
        self.mix_status_label = ctk.CTkLabel(parent, text="", text_color="gray")
        self.mix_status_label.pack(pady=5)
        
        # 4. List
        ctk.CTkLabel(parent, text="Custom Voices:", font=("Roboto", 14, "bold")).pack(anchor="w", padx=10, pady=(20,5))
        self.custom_list_frame = ctk.CTkScrollableFrame(parent, height=200)
        self.custom_list_frame.pack(fill="x", padx=10, pady=5)
        
        self.refresh_voice_lists()

    def build_fx_tab(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        
        # --- Preset Controls ---
        pre_frame = ctk.CTkFrame(parent, fg_color="transparent")
        pre_frame.pack(fill="x", padx=10, pady=(10,5))
        
        self.fx_preset_combo = ctk.CTkComboBox(pre_frame, values=["Select FX Preset..."], command=self.load_fx_preset, width=200)
        self.fx_preset_combo.pack(side="left", padx=(0,5))
        
        ctk.CTkButton(pre_frame, text="💾 Save", width=60, command=self.save_fx_preset_dialog).pack(side="left", padx=2)
        ctk.CTkButton(pre_frame, text="🔄", width=30, command=self.refresh_fx_presets).pack(side="left", padx=2)
        
        scroll = ctk.CTkScrollableFrame(parent)
        scroll.pack(fill="both", expand=True, padx=5, pady=5)
        scroll.grid_columnconfigure(0, weight=1)

        # Helper to create rows
        def _create_slider(parent, label_text, variable, from_, to_, steps=100, label_attr=None):
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.pack(fill="x", padx=5, pady=2)
            lbl = ctk.CTkLabel(row, text=label_text, width=120, anchor="w")
            lbl.pack(side="left")
            if label_attr: setattr(self, label_attr, lbl)
            
            ctk.CTkSlider(row, from_=from_, to=to_, number_of_steps=steps, variable=variable, 
                          command=lambda v: self.update_fx_labels()).pack(side="left", fill="x", expand=True, padx=5)

        # --- 1. Dynamics ---
        dyn_frame = ctk.CTkFrame(scroll)
        dyn_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(dyn_frame, text="Dynamics", font=("Roboto", 14, "bold")).pack(anchor="w", padx=10, pady=5)
        
        # Compressor
        c_head = ctk.CTkFrame(dyn_frame, fg_color="transparent")
        c_head.pack(fill="x", padx=5)
        ctk.CTkCheckBox(c_head, text="Compressor", variable=self.comp_enabled, font=("Roboto", 12, "bold")).pack(side="left")
        
        c_body = ctk.CTkFrame(dyn_frame)
        c_body.pack(fill="x", padx=10, pady=2)
        _create_slider(c_body, "Threshold", self.comp_threshold, -60, 0, 60, 'comp_thresh_label')
        _create_slider(c_body, "Ratio", self.comp_ratio, 1, 20, 19, 'comp_ratio_label')
        
        # Limiter
        l_head = ctk.CTkFrame(dyn_frame, fg_color="transparent")
        l_head.pack(fill="x", padx=5, pady=(5,0))
        ctk.CTkCheckBox(l_head, text="Limiter", variable=self.limiter_enabled, font=("Roboto", 12, "bold")).pack(side="left")
        
        l_body = ctk.CTkFrame(dyn_frame)
        l_body.pack(fill="x", padx=10, pady=2)
        _create_slider(l_body, "Threshold", self.limiter_threshold, -12, 0, 24, 'lim_thresh_label')
        
        # Gain
        g_head = ctk.CTkFrame(dyn_frame, fg_color="transparent")
        g_head.pack(fill="x", padx=5, pady=(5,0))
        ctk.CTkCheckBox(g_head, text="Gain", variable=self.gain_enabled, font=("Roboto", 12, "bold")).pack(side="left")
        _create_slider(dyn_frame, "dB", self.gain_db, -20, 20, 80, 'gain_label')

        # --- 2. EQ & Filters ---
        eq_frame = ctk.CTkFrame(scroll)
        eq_frame.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(eq_frame, text="EQ & Filters", font=("Roboto", 14, "bold")).pack(anchor="w", padx=10, pady=5)
        
        _create_slider(eq_frame, "Bass (LowShelf)", self.eq_bass, -20, 20, 40, 'bass_label')
        _create_slider(eq_frame, "Treble (HighShelf)", self.eq_treble, -20, 20, 40, 'treble_label')
        
        # HPF
        h_head = ctk.CTkFrame(eq_frame, fg_color="transparent")
        h_head.pack(fill="x", padx=5, pady=(5,0))
        ctk.CTkCheckBox(h_head, text="HighPass Filter", variable=self.highpass_enabled).pack(side="left")
        _create_slider(eq_frame, "Freq (Hz)", self.highpass_freq, 20, 1000, 100, 'hpf_label')
        
        # LPF
        lpf_head = ctk.CTkFrame(eq_frame, fg_color="transparent")
        lpf_head.pack(fill="x", padx=5, pady=(5,0))
        ctk.CTkCheckBox(lpf_head, text="LowPass Filter", variable=self.lowpass_enabled).pack(side="left")
        _create_slider(eq_frame, "Freq (Hz)", self.lowpass_freq, 1000, 20000, 100, 'lpf_label')

        # --- 3. Spatial & Time ---
        sp_frame = ctk.CTkFrame(scroll)
        sp_frame.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(sp_frame, text="Spatial & Time", font=("Roboto", 14, "bold")).pack(anchor="w", padx=10, pady=5)
        
        # Reverb
        r_head = ctk.CTkFrame(sp_frame, fg_color="transparent")
        r_head.pack(fill="x", padx=5)
        ctk.CTkCheckBox(r_head, text="Reverb", variable=self.reverb_enabled, font=("Roboto", 12, "bold")).pack(side="left")
        
        r_body = ctk.CTkFrame(sp_frame)
        r_body.pack(fill="x", padx=10, pady=2)
        _create_slider(r_body, "Room Size", self.reverb_room_size, 0, 1, 100, 'rev_room_label')
        _create_slider(r_body, "Wet Level", self.reverb_wet_level, 0, 1, 100, 'rev_wet_label')
        _create_slider(r_body, "Damping", self.reverb_damping, 0, 1, 100, None)
        _create_slider(r_body, "Width", self.reverb_width, 0, 1, 100, None)

        # Delay
        d_head = ctk.CTkFrame(sp_frame, fg_color="transparent")
        d_head.pack(fill="x", padx=5, pady=(5,0))
        ctk.CTkCheckBox(d_head, text="Delay", variable=self.delay_enabled, font=("Roboto", 12, "bold")).pack(side="left")
        
        d_body = ctk.CTkFrame(sp_frame)
        d_body.pack(fill="x", padx=10, pady=2)
        _create_slider(d_body, "Time (s)", self.delay_time, 0, 2, 100, 'dly_time_label')
        _create_slider(d_body, "Feedback", self.delay_feedback, 0, 1, 100, None)
        _create_slider(d_body, "Mix", self.delay_mix, 0, 1, 100, 'dly_mix_label')

        # --- 4. Guitar / Modulation ---
        mod_frame = ctk.CTkFrame(scroll)
        mod_frame.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(mod_frame, text="Guitar / Modulation", font=("Roboto", 14, "bold")).pack(anchor="w", padx=10, pady=5)

        # Chorus
        ch_head = ctk.CTkFrame(mod_frame, fg_color="transparent")
        ch_head.pack(fill="x", padx=5)
        ctk.CTkCheckBox(ch_head, text="Chorus", variable=self.chorus_enabled).pack(side="left")
        _create_slider(mod_frame, "Rate (Hz)", self.chorus_rate, 0.1, 10, 50, 'chorus_rate_label')
        _create_slider(mod_frame, "Depth", self.chorus_depth, 0, 1, 50, None)
        
        # Distortion
        di_head = ctk.CTkFrame(mod_frame, fg_color="transparent")
        di_head.pack(fill="x", padx=5, pady=(5,0))
        ctk.CTkCheckBox(di_head, text="Distortion", variable=self.distortion_enabled).pack(side="left")
        _create_slider(mod_frame, "Drive (dB)", self.distortion_drive, 0, 60, 60, 'dist_drive_label')
        
        # Phaser
        ph_head = ctk.CTkFrame(mod_frame, fg_color="transparent")
        ph_head.pack(fill="x", padx=5, pady=(5,0))
        ctk.CTkCheckBox(ph_head, text="Phaser", variable=self.phaser_enabled).pack(side="left")
        _create_slider(mod_frame, "Rate (Hz)", self.phaser_rate, 0.1, 10, 50, 'phaser_rate_label')
        
        # Clipping
        cl_head = ctk.CTkFrame(mod_frame, fg_color="transparent")
        cl_head.pack(fill="x", padx=5, pady=(5,0))
        ctk.CTkCheckBox(cl_head, text="Clipping", variable=self.clipping_enabled).pack(side="left")
        _create_slider(mod_frame, "Threshold (dB)", self.clipping_thresh, -20, 0, 40, 'clip_thresh_label')

        # --- 5. Quality & Pitch ---
        q_frame = ctk.CTkFrame(scroll)
        q_frame.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(q_frame, text="Quality / Pitch", font=("Roboto", 14, "bold")).pack(anchor="w", padx=10, pady=5)
        
        # Pitch Shift
        ps_head = ctk.CTkFrame(q_frame, fg_color="transparent")
        ps_head.pack(fill="x", padx=5)
        ctk.CTkCheckBox(ps_head, text="Pitch Shift (High Quality)", variable=self.pitch_shift_enabled).pack(side="left")
        _create_slider(q_frame, "Semitones", self.pitch_shift_semitones, -12, 12, 48, 'pitch_shift_label')
        
        # Bitcrush
        bc_head = ctk.CTkFrame(q_frame, fg_color="transparent")
        bc_head.pack(fill="x", padx=5, pady=(5,0))
        ctk.CTkCheckBox(bc_head, text="Bitcrush", variable=self.bitcrush_enabled).pack(side="left")
        _create_slider(q_frame, "Bit Depth", self.bitcrush_depth, 2, 16, 28, 'bit_depth_label')
        
        # GSM
        ctk.CTkCheckBox(q_frame, text="GSM Compressor (Phone Quality)", variable=self.gsm_enabled).pack(anchor="w", padx=10, pady=5)
        
        # Init labels
        self.update_fx_labels()
        self.refresh_fx_presets()

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

    def build_lexicon_tab(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(1, weight=1) # List area expands

        # 1. Add New Entry
        add_frame = ctk.CTkFrame(parent)
        add_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        
        ctk.CTkLabel(add_frame, text="Original Text:").pack(side="left", padx=5)
        self.lex_orig_var = ctk.StringVar()
        ctk.CTkEntry(add_frame, textvariable=self.lex_orig_var, width=150).pack(side="left", padx=5)
        
        ctk.CTkLabel(add_frame, text="Replacement:").pack(side="left", padx=5)
        self.lex_replace_var = ctk.StringVar()
        ctk.CTkEntry(add_frame, textvariable=self.lex_replace_var, width=150).pack(side="left", padx=5)
        
        ctk.CTkButton(add_frame, text="Add Rule", command=self.add_lexicon_rule).pack(side="left", padx=10)

        # 2. List
        self.lex_list_frame = ctk.CTkScrollableFrame(parent)
        self.lex_list_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        
        self.refresh_lexicon_list()
        
        # 3. Help Text
        ctk.CTkLabel(parent, text="Note: Replacements are case-insensitive. Applied before generation.", text_color="gray").grid(row=2, column=0, pady=5)

    def add_lexicon_rule(self):
        orig = self.lex_orig_var.get().strip()
        rep = self.lex_replace_var.get().strip()
        
        if not orig:
            messagebox.showwarning("Error", "Original text cannot be empty.")
            return
            
        if "lexicon" not in self.settings:
            self.settings["lexicon"] = {}
            
        self.settings["lexicon"][orig] = rep
        self.lex_orig_var.set("")
        self.lex_replace_var.set("")
        self.save_settings()
        self.refresh_lexicon_list()

    def delete_lexicon_rule(self, key):
        if key in self.settings.get("lexicon", {}):
            del self.settings["lexicon"][key]
            self.save_settings()
            self.refresh_lexicon_list()

    def refresh_lexicon_list(self):
        for widget in self.lex_list_frame.winfo_children():
            widget.destroy()
            
        lexicon = self.settings.get("lexicon", {})
        if not lexicon:
            ctk.CTkLabel(self.lex_list_frame, text="No rules defined.", text_color="gray").pack(pady=10)
            return

        for i, (orig, rep) in enumerate(lexicon.items()):
            row = ctk.CTkFrame(self.lex_list_frame)
            row.pack(fill="x", pady=2)
            
            ctk.CTkLabel(row, text=orig, width=150, anchor="w", font=("Consolas", 12)).pack(side="left", padx=10)
            ctk.CTkLabel(row, text="->", width=30).pack(side="left")
            ctk.CTkLabel(row, text=rep, width=150, anchor="w", font=("Consolas", 12)).pack(side="left", padx=10)
            
            ctk.CTkButton(row, text="X", width=30, fg_color="#c42b1c", command=lambda k=orig: self.delete_lexicon_rule(k)).pack(side="right", padx=5)

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

    def update_fx_labels(self):
        # EQ
        if hasattr(self, 'bass_label'): self.bass_label.configure(text=f"Bass: {self.eq_bass.get():.1f} dB")
        if hasattr(self, 'treble_label'): self.treble_label.configure(text=f"Treble: {self.eq_treble.get():.1f} dB")
        if hasattr(self, 'hpf_label'): self.hpf_label.configure(text=f"Freq: {int(self.highpass_freq.get())} Hz")
        if hasattr(self, 'lpf_label'): self.lpf_label.configure(text=f"Freq: {int(self.lowpass_freq.get())} Hz")
        
        # Comp / Dynamics
        if hasattr(self, 'comp_thresh_label'): self.comp_thresh_label.configure(text=f"Thresh: {self.comp_threshold.get():.1f} dB")
        if hasattr(self, 'comp_ratio_label'): self.comp_ratio_label.configure(text=f"Ratio: {self.comp_ratio.get():.1f}:1")
        if hasattr(self, 'lim_thresh_label'): self.lim_thresh_label.configure(text=f"Thresh: {self.limiter_threshold.get():.1f} dB")
        if hasattr(self, 'gain_label'): self.gain_label.configure(text=f"Gain: {self.gain_db.get():.1f} dB")
        
        # Reverb
        if hasattr(self, 'rev_room_label'): self.rev_room_label.configure(text=f"Size: {self.reverb_room_size.get():.2f}")
        if hasattr(self, 'rev_wet_label'): self.rev_wet_label.configure(text=f"Wet: {self.reverb_wet_level.get():.2f}")
        
        # Delay
        if hasattr(self, 'dly_time_label'): self.dly_time_label.configure(text=f"Time: {self.delay_time.get():.2f} s")
        if hasattr(self, 'dly_mix_label'): self.dly_mix_label.configure(text=f"Mix: {self.delay_mix.get():.2f}")
        
        # Guitar
        if hasattr(self, 'dist_drive_label'): self.dist_drive_label.configure(text=f"Drive: {self.distortion_drive.get():.1f} dB")
        if hasattr(self, 'chorus_rate_label'): self.chorus_rate_label.configure(text=f"Rate: {self.chorus_rate.get():.1f} Hz")
        if hasattr(self, 'phaser_rate_label'): self.phaser_rate_label.configure(text=f"Rate: {self.phaser_rate.get():.1f} Hz")
        if hasattr(self, 'clip_thresh_label'): self.clip_thresh_label.configure(text=f"Thresh: {self.clipping_thresh.get():.1f} dB")
        
        # Quality / Pitch
        if hasattr(self, 'bit_depth_label'): self.bit_depth_label.configure(text=f"Depth: {self.bitcrush_depth.get():.1f}")
        if hasattr(self, 'pitch_shift_label'): self.pitch_shift_label.configure(text=f"Shift: {self.pitch_shift_semitones.get():.1f} st")

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
                        winsound.PlaySound(tmp_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
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
