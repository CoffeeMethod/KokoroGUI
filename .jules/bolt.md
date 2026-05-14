## 2024-05-24 - [Regex Compilation Overhead]
**Learning:** Python`s `re.compile()` has some internal caching, but doing it in a tight loop across multiple text chunks or items still introduces measurable overhead.
**Action:** When a method applies dictionary/lexicon replacements via regular expressions, explicitly cache the compiled regex objects in a class attribute or closure, rather than compiling them on every call.

## 2024-05-24 - [O(N²) String Concatenation Bottleneck]
**Learning:** Using `+=` for string concatenation inside loops (such as parsing text from PDFs, EPUBs, or concatenating JIT remaining text) in Python creates an O(N²) time complexity bottleneck due to string immutability.
**Action:** Always accumulate strings in a list using `.append()` and combine them once at the end using `"".join(list)` instead of using `+=` inside loops.
