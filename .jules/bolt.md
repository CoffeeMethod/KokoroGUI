## 2024-05-24 - [Regex Compilation Overhead]
**Learning:** Python`s `re.compile()` has some internal caching, but doing it in a tight loop across multiple text chunks or items still introduces measurable overhead.
**Action:** When a method applies dictionary/lexicon replacements via regular expressions, explicitly cache the compiled regex objects in a class attribute or closure, rather than compiling them on every call.
