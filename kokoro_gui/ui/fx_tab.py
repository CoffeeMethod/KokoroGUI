"""Audio FX tab: builds the FX sliders/toggles and loads/saves FX presets under
`presets/fx/`.

Calls `gui.messagebox`, `gui.ctk.CTkInputDialog`, and reads `gui.FX_PRESETS_DIR`
qualified, at call time, so tests can keep monkeypatching those names on the
`gui` module (the `tts_app` fixture redirects `FX_PRESETS_DIR` into a tmp_path
and replaces `messagebox` with a `MagicMock()`; `test_gui_handlers.py` patches
`gui.ctk.CTkInputDialog` with a fake dialog).
"""
import json
import os
import re

import customtkinter as ctk

import gui


class FXTabMixin:
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

    def refresh_fx_presets(self):
        presets = ["Select FX Preset..."]
        if os.path.exists(gui.FX_PRESETS_DIR):
            files = [f for f in os.listdir(gui.FX_PRESETS_DIR) if f.endswith(".json")]
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
        dialog = gui.ctk.CTkInputDialog(text="Enter FX preset name:", title="Save FX Preset")
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

            fpath = os.path.join(gui.FX_PRESETS_DIR, f"{name}.json")
            try:
                with open(fpath, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4)
                gui.messagebox.showinfo("Saved", f"FX Preset '{name}' saved.")
                self.refresh_fx_presets()
                if hasattr(self, 'fx_preset_combo'): self.fx_preset_combo.set(name)
                if hasattr(self, 'gen_fx_combo'): self.gen_fx_combo.set(name)
            except Exception as e:
                gui.messagebox.showerror("Error", f"Failed to save FX preset: {e}")

    def load_fx_preset(self, name):
        if name == "Select FX Preset...": return

        safe_name = os.path.basename(name)
        if not safe_name: return
        fpath = os.path.join(gui.FX_PRESETS_DIR, f"{safe_name}.json")
        if os.path.exists(fpath):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
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
                gui.messagebox.showerror("Error", f"Failed to load FX preset: {e}")

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
