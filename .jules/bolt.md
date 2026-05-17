## 2024-05-24 - [Regex Compilation Overhead]
**Learning:** Python`s `re.compile()` has some internal caching, but doing it in a tight loop across multiple text chunks or items still introduces measurable overhead.
**Action:** When a method applies dictionary/lexicon replacements via regular expressions, explicitly cache the compiled regex objects in a class attribute or closure, rather than compiling them on every call.

## 2024-05-24 - [O(N²) String Concatenation Bottleneck]
**Learning:** Using `+=` for string concatenation inside loops (especially for large documents like PDFs and EPUBs) leads to O(N²) time complexity because a new string must be allocated and copied each time. This creates a significant performance bottleneck in Python when processing large text payloads.
**Action:** When accumulating strings in a loop, always use a list to append chunks (e.g., `text_list.append(chunk)`) and then join them at the end using `"".join(text_list)`.
