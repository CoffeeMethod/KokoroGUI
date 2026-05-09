## 2024-06-25 - Regex Compilation Overhead
**Learning:** Compiling regexes in tight loops (e.g. `re.compile()` in `apply_lexicon` which is called per text segment) adds unnecessary overhead.
**Action:** Always cache compiled regex patterns when applying dictionary replacements in a loop or across multiple segments to avoid recompilation overhead.
