"""Lexicon (find/replace) substitution, applied to text before synthesis."""
import re


class LexiconMixin:
    def apply_lexicon(self, text, lexicon):
        """
        Applies a dictionary of replacements to the text.
        Case-insensitive finding, preserves case of replacement.
        """
        if not lexicon:
            return text

        for src, dest in lexicon.items():
            if not src: continue
            try:
                # Use cached pattern if available to avoid repeated recompilation overhead
                if src not in self._lexicon_cache:
                    # Escape the search term to treat it as literal text
                    self._lexicon_cache[src] = re.compile(re.escape(src), re.IGNORECASE)

                pattern = self._lexicon_cache[src]
                text = pattern.sub(dest, text)
            except Exception as e:
                print(f"Lexicon error for '{src}': {e}")

        return text
