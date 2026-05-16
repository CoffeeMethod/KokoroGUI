## 2025-02-14 - Fix Path Traversal in Kokoro Engine File Operations
**Vulnerability:** User inputs (voice names, mix names, preset names) were being passed directly into `os.path.join()` without sanitization in `kokoro_engine.py` (e.g., `os.path.join(CUSTOM_VOICES_DIR, f"{voice_name}.pt")`).
**Learning:** Python's `os.path.join()` allows path traversal if an absolute path or relative path traversing upward (like `../`) is passed as a subsequent argument. This allows users to read/write arbitrary files outside the intended directories (`CUSTOM_VOICES_DIR`, `presets/`). This application accepts unvalidated input from GUI elements like preset saves and multispeaker texts.
**Prevention:** Always sanitize user-provided filename strings using `os.path.basename()` before passing them to `os.path.join()`.
## 2024-05-16 - Prevent Path Traversal in GUI Presets
**Vulnerability:** The GUI allowed unsanitized names for saving and loading preset, fx preset, and deleting custom voices, which were directly passed to `os.path.join()`. This creates a path traversal risk where arbitrary `.json` or `.pt` files could be read, written, or deleted outside intended directories.
**Learning:** Even internal GUI-driven file operations need sanitization if the name comes from a GUI text input dialog or arbitrary text parsed from other places.
**Prevention:** Always use `os.path.basename()` to sanitize filenames before joining them to a base directory in GUI operations.
