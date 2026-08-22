"""Custom Voice (mixing) tab: blends two voice tensors via `self.engine.mix_voices`
and previews/saves the result.

Calls `gui.messagebox` qualified, at call time, so tests can keep monkeypatching
that name on the `gui` module (the `tts_app` fixture replaces it with a
`MagicMock()`). `playback` is not a `gui`-level monkeypatch target in the test
suite, so it's imported normally here, same as the original `gui.py`.
"""
import os
import re
import tempfile

import customtkinter as ctk
import playback

import gui


class MixingTabMixin:
    def _update_mix_voice_list(self, lang_var, combo_attr, voice_var):
        code = lang_var.get()
        if hasattr(self, combo_attr):
            combo = getattr(self, combo_attr)
            voices = self.get_all_voices(code)
            combo.configure(values=voices)
            if voice_var.get() not in voices:
                voice_var.set(voices[0])

    def on_mix_lang_a_change(self, *args):
        self._update_mix_voice_list(self.mix_lang_a_var, 'mix_combo_a', self.mix_voice_a_var)

    def on_mix_lang_b_change(self, *args):
        self._update_mix_voice_list(self.mix_lang_b_var, 'mix_combo_b', self.mix_voice_b_var)

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
        if gui.messagebox.askyesno("Confirm", f"Delete voice '{name}'?"):
            try:
                path = os.path.join("custom_voices", f"{name}.pt")
                if os.path.exists(path):
                    os.remove(path)
                    self.refresh_voice_lists()
            except Exception as e:
                gui.messagebox.showerror("Error", f"Failed to delete: {e}")

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
            except Exception: pass

            return success, ""

        def _on_done(future):
            try:
                success, err = future.result()
                if success:
                    self.after(0, lambda: self.mix_status_label.configure(text="Playing preview...", text_color="green"))
                    playback.play(tmp_audio_path)
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
            gui.messagebox.showwarning("Error", "Please enter a name for the new voice.")
            return

        if not re.match(r'^[a-zA-Z0-9_-]+$', name):
             gui.messagebox.showwarning("Error", "Invalid name. Use alphanumeric, _, - only.")
             return

        if name in self.get_all_voices():
            if not gui.messagebox.askyesno("Overwrite", f"Voice '{name}' exists. Overwrite?"):
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
