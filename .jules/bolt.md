## 2024-05-24 - Lexicon Compilation Caching
**Learning:** In text-processing intensive workflows like `apply_lexicon`, using `re.compile()` repeatedly on user-defined text replacements introduces measurable overhead, especially with large lexicons inside a processing loop.
**Action:** Always maintain a compilation cache (like `_lexicon_cache`) when repeatedly applying dynamically-created regexes in core loops.
