## 2024-05-24 - [Regex Compilation Overhead]
**Learning:** Python`s `re.compile()` has some internal caching, but doing it in a tight loop across multiple text chunks or items still introduces measurable overhead.
**Action:** When a method applies dictionary/lexicon replacements via regular expressions, explicitly cache the compiled regex objects in a class attribute or closure, rather than compiling them on every call.

## 2024-05-24 - [Inefficient String Concatenation]
**Learning:** Using `+=` for string concatenation inside loops (e.g., when parsing large PDFs or EPUBs) leads to O(N²) time complexity because strings are immutable and a new string must be allocated each time. This creates a severe performance bottleneck for large documents.
**Action:** Always accumulate strings in a list using `.append()` and then combine them once at the end using `"".join(list)`. This reduces the complexity to O(N).
