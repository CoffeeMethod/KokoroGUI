## 2024-05-24 - [Path Traversal in Parsed Multi-speaker Syntax]
**Vulnerability:** User-controlled filenames generated from parsed multi-speaker text (e.g. `[speaker_name]:`) or direct inputs to engine mix/load endpoints lacked sanitization in `kokoro_engine.py`, resulting in Path Traversal vulnerabilities when fetching presets, voices, and fx files using `os.path.join()`.
**Learning:** Due to the complex text parsing mechanism, what initially seems like clean preset names parsed out of input strings might contain relative traversal directories, making the system unexpectedly vulnerable.
**Prevention:** Always use `os.path.basename()` or equivalent file name sanitization before passing parsed components or user inputs to `os.path.join()`.
