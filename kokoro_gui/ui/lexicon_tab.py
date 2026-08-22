"""Lexicon tab: find/replace rules stored in `self.settings["lexicon"]`.

Calls `gui.messagebox` qualified, at call time, so tests can keep monkeypatching
that name on the `gui` module (the `tts_app` fixture replaces it with a
`MagicMock()`).
"""
import customtkinter as ctk

import gui


class LexiconTabMixin:
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
            gui.messagebox.showwarning("Error", "Original text cannot be empty.")
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
