## 2024-05-24 - [Regex Compilation Overhead]
**Learning:** Python`s `re.compile()` has some internal caching, but doing it in a tight loop across multiple text chunks or items still introduces measurable overhead.
**Action:** When a method applies dictionary/lexicon replacements via regular expressions, explicitly cache the compiled regex objects in a class attribute or closure, rather than compiling them on every call.
## 2024-05-24 - [O(N²) String Concatenation Bottleneck]
**Learning:** Using the `+=` operator for string concatenation inside a loop, particularly when reading long texts like PDFs/EPUBs or aggregating large JIT segment remnants, creates an O(N²) time complexity bottleneck due to continuous reallocation and copying of immutable strings in Python.
**Action:** Always accumulate individual text segments in a list using `.append()` and combine them at the end using `"".join(list)`.
