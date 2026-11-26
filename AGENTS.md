# AGENTS.md

Instructions for AI coding agents working on KokoroGUI. People should read [CONTRIBUTING.md](CONTRIBUTING.md),
which has the full setup, test conventions and the steps for adding an engine. This file is the short version, plus
the rules an agent tends to break.

If a `CLAUDE.md` or a `Claude/` folder exists in your checkout, those are the maintainer's local notes. Read them
first; they go into more detail than this file and win where the two differ.

## What this is

A PySide6 desktop text to speech editor. The transcript is the project: each run of text belongs to a character,
each character has a voice and an engine, and the timeline is built from the text. `main.py` starts the only window.
Projects save as one `.tbaw` file.

| Folder | Holds |
| --- | --- |
| `kokoro_gui/qt/` | The window, its docks (`qt/docks/`), dialogs and project open/save (`qt/project.py`) |
| `kokoro_gui/daw/` | The document model: transcript, clips, tracks, characters, staleness, undo |
| `kokoro_gui/audio/` | Transport, mixer and the FX applied at playback |
| `kokoro_gui/core/` | What the app does with engines, clips and configs, with no widgets |
| `kokoro_gui/headless/` | The command line and the MCP server over `core/` |
| `kokoro_gui/engine/` | The shared synthesis runner and its mixins |
| `kokoro_gui/engines/` | Backend adapters (Kokoro, Audio8, LuxTTS, Dummy), the registry and voice stores |
| `kokoro_engine.py`, `playback.py` | Kokoro's model, and the preview playback boundary |
| `docs/` | The GitHub Pages site |

## Commands

```bash
pip install -r requirements.txt -r requirements-test.txt
python main.py
pytest                         # full fast suite, mocked, no model or eSpeak needed (several minutes)
pytest tests/test_caching.py   # one file
ruff check .                   # the only lint step; there is no formatter
python scripts/i18n.py extract # after adding or changing any user-facing string
```

Run focused test files while you work and plain `pytest` before you push. CI runs the suite without `tests/gui_qt`
on Windows, Linux and macOS, so a local run is the only thing that covers the GUI tests.

## Rules

**Ask first:**

- Never change how a `.tbaw` is saved or laid out (the zip entries, the manifest, `document.json` fields, the open and
  save code in `qt/project.py`) without the maintainer's explicit approval.
- Never update `tests/test_segment_key_pins.py` to make a test pass. A segment key change marks every clip in every
  existing project stale.
- Don't settle open design questions yourself. Ask the maintainer, one question at a time.

**Git:**

- `main` only changes when a release merges into it. Branch from the latest release branch (the highest version, such
  as `4.0.0-beta.3`) and target it with your pull request. CONTRIBUTING.md shows how to find it.
- Never commit secrets, machine paths, user projects, caches or anything else `.gitignore` excludes. Don't force-add
  ignored files.

**Code:**

- Imports at the top level are flat (`import kokoro_engine`). Nothing in `kokoro_gui/engine/` imports `kokoro_engine`
  or `kokoro` at module level, and `engines/` never imports the document model.
- `core/` and `headless/` never import PySide6, `qt/app.py`, `qt/docks/`, `qt/subprojects.py` or `audio/transport.py`.
  Logic that isn't about widgets goes in `core/`.
- Engine config is a plain `dict`. A new setting goes through `QtTTSApp._assemble_config`, `core/config.py` if a clip
  or the command line uses it, and `tests/gui_qt/test_qt_config_assembly.py`.
- Clips store raw audio. FX run at playback and export, so an FX change never marks a clip stale.
- Edit the model by assignment, an undo command or `document.touch()`. Mutating a list or dict in place skips the
  revision counters, and the suite fails with `StaleCacheError`. Fix the edit; don't turn the check off.
- Keep generation, book import and project IO off the GUI thread.
- Preview audio goes through `playback.py`. Never call `sounddevice` directly from an engine.
- Book files, `.tbaw` and `.kgshare` bundles, templates and `config_qt.json` are untrusted input. A name that becomes
  a file path gets `os.path.basename()` or the existing containment check. Reuse the size limits already there.
- Only `main.py` sets up logging and the exception hooks.
- M4B export is the one place the app runs another program (`ffmpeg`), always with an argument list, never a shell.

**Text a person reads:**

- Wrap it in `tr("...")` with `{placeholders}`, never an f-string. Use `ntr(...)` for counts and `N_(...)` in tables
  built at import. Text stored in a project, sent to an engine or exported is not translated.
- A new button, field or menu entry gets a tooltip.
- Use the app's words: Project, Transcript, Clip, Character, Voice, Engine, FX, Generate, Preview, Export. A clip's
  audio is "stale"; a project is "unsaved".

**Tests:**

- Never touch real voice stores, caches, projects or audio devices. Use the fixtures in `tests/conftest.py`
  (`make_config`, `isolated_dirs`, `engine`, `fake_pipeline`) and patch storage and playback on
  `kokoro_gui.engine.runtime`. Only `tests/test_caching.py` turns caching on.
- GUI tests use the `qt_app` fixture. Call `qt_app.wait_for_project_io()` after Save or Open and
  `qt_app.flush_updates()` after typing.
- Never open a real ffmpeg, microphone or model in a test. Mock them.

**Docs:**

- A change a user can see gets a one-line bullet in the README's "What's new" and a full entry in `docs/changes.html`,
  under one "Unreleased" heading for everything since the latest tag. Keep the README short; the detail belongs on
  the site.
- Write plain, short prose. No em dashes, no marketing words.
