## 2025-02-14 - Fix Path Traversal in Kokoro Engine File Operations
**Vulnerability:** User inputs (voice names, mix names, preset names) were being passed directly into `os.path.join()` without sanitization in `kokoro_engine.py` (e.g., `os.path.join(CUSTOM_VOICES_DIR, f"{voice_name}.pt")`).
**Learning:** Python's `os.path.join()` allows path traversal if an absolute path or relative path traversing upward (like `../`) is passed as a subsequent argument. This allows users to read/write arbitrary files outside the intended directories (`CUSTOM_VOICES_DIR`, `presets/`). This application accepts unvalidated input from GUI elements like preset saves and multispeaker texts.
**Prevention:** Always sanitize user-provided filename strings using `os.path.basename()` before passing them to `os.path.join()`.
## 2025-05-14 - Path Traversal Prevention Limitations
**Vulnerability:** Path traversal fixes using os.path.basename() can break functionality if applied to absolute paths.
**Learning:** In desktop GUI applications, fields like 'filename' might be expected to contain relative or absolute paths, and sanitizing them unconditionally with os.path.basename() can break user-specified save locations.
**Prevention:** Only sanitize input strings that are meant to be pure filenames (e.g. preset names, voice names) and NOT inputs that can legitimately be paths.
