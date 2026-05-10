## 2024-05-18 - [CRITICAL] Path Traversal in Resource Loading
**Vulnerability:** User-provided inputs (like preset names, voice names, and output filenames) were passed directly to `os.path.join` to construct file paths.
**Learning:** This codebase frequently constructs paths dynamically using identifiers from user configs or multi-speaker text parsing. Because `os.path.join` evaluates absolute paths and `..` traverses directories, unsanitized inputs could write to or read from arbitrary locations.
**Prevention:** Always sanitize any filename input from external sources using `os.path.basename()` before combining it with a trusted directory using `os.path.join()`.
