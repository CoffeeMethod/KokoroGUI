## 2024-05-24 - [Regex Compilation Overhead]
**Learning:** Python`s `re.compile()` has some internal caching, but doing it in a tight loop across multiple text chunks or items still introduces measurable overhead.
**Action:** When a method applies dictionary/lexicon replacements via regular expressions, explicitly cache the compiled regex objects in a class attribute or closure, rather than compiling them on every call.

## 2024-05-24 - [String Concatenation Bottlenecks]
**Learning:** Using `+=` for string accumulation inside loops leads to O(N²) time complexity bottlenecks during large text generation, which is a known pattern to avoid.
**Action:** Always accumulate strings in a list using `.append()` and combine them once at the end using `"".join(list)` instead of using `+=` inside loops.
