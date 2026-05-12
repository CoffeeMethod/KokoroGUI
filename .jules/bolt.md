## 2024-05-24 - [Regex Compilation Overhead]
**Learning:** Python`s `re.compile()` has some internal caching, but doing it in a tight loop across multiple text chunks or items still introduces measurable overhead.
**Action:** When a method applies dictionary/lexicon replacements via regular expressions, explicitly cache the compiled regex objects in a class attribute or closure, rather than compiling them on every call.
## 2024-05-24 - [String Concatenation Bottleneck]
**Learning:** Using `+=` for string concatenation inside loops (like iterating over PDF pages or large document segments) forces Python to create a new string object each time, leading to O(N²) time complexity.
**Action:** Always accumulate strings in a list using `.append()` and combine them once at the end using `"".join(list)` to maintain O(N) performance.
