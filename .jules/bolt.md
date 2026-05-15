## 2024-05-24 - [Regex Compilation Overhead]
**Learning:** Python`s `re.compile()` has some internal caching, but doing it in a tight loop across multiple text chunks or items still introduces measurable overhead.
**Action:** When a method applies dictionary/lexicon replacements via regular expressions, explicitly cache the compiled regex objects in a class attribute or closure, rather than compiling them on every call.

## 2024-05-15 - Optimize O(N²) String Concatenation to O(N) Array Joins
**Learning:** Found significant O(N²) bottlenecks in Python loops when processing large amounts of text using `+=` string concatenation (especially in `extract_text_from_file` across many document pages and `_process_jit_async` over multiple text segments).
**Action:** Always accumulate text using `.append()` on a list and return `"".join(list_name)` outside the loop to optimize from O(N²) to O(N) complexity for text-heavy operations.
