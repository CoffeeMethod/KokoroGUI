## What

<!-- One or two sentences. What changes for a user, or for a developer if it's internal. -->

## Why

<!-- The bug or gap. Link the issue if there is one. -->

## How to check it

<!-- What you ran or clicked. `pytest` output counts; a screenshot for anything visual. -->

## Checklist

- [ ] `pytest` passes locally (the fast suite; CI runs it on Windows and Linux)
- [ ] New settings are threaded through `_assemble_config` and `tests/gui_qt/test_qt_config_assembly.py`
- [ ] README's "What's new" (one line) and `docs/changes.html` (full entry) updated if a user can see the change
