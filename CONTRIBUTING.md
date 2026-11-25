# Contributing

Bug reports and pull requests are welcome. Questions go in Discussions, security reports through
the Security tab (see [SECURITY.md](SECURITY.md)).

## Which branch to work on

`main` only changes when a release is merged into it. Don't push to `main`, and don't open a pull request against it.

Work happens on a branch named after the next release, such as `4.0.0-beta.3`. Start from the one with the highest version
number and open your pull request against it. Your change reaches `main` when that release is merged. This lists the release
branches, highest version first:

```bash
git fetch origin
git branch -r --list "origin/[0-9]*" --sort=-version:refname
```

Check again before you open the pull request, since a newer release branch may have appeared. If the highest one is already
released (it has a tag by that name) and there's no newer one, ask in Discussions where your change should go.

## Setup

```bash
git clone https://github.com/CoffeeMethod/KokoroGUI.git
cd KokoroGUI
git switch 4.0.0-beta.3                            # the latest release branch, see above
python -m venv .venv && . .venv/Scripts/activate   # or .venv/bin/activate
pip install -r requirements.txt -r requirements-test.txt
python main.py
```

Python 3.11 or newer. eSpeak NG is only needed to actually synthesize with Kokoro; the fast test
suite runs without it.

## Reporting a bug

`python main.py` writes a log to `cache/logs/kokorogui.log` (Help > Open Log Folder opens it).
Attach the end of it, or the traceback from the crash dialog's Copy button. It can hold file paths
from your machine, so read it first.

## Tests

```bash
pytest
```

That's the fast suite: the Kokoro pipeline is mocked, playback is mocked, no model download. The Qt tests run headless (`tests/gui_qt/conftest.py` sets
`QT_QPA_PLATFORM=offscreen`), so no display is needed. CI runs the suite on `windows-latest`,
`ubuntu-latest` and `macos-latest` without `tests/gui_qt/` (`pytest --ignore=tests/gui_qt -p no:pytest-qt`). The GUI
tests run in a separate `gui` job on Ubuntu, which is not a required check, so a red `gui` job doesn't block a merge.
Run plain `pytest` before you push, `tests/gui_qt/` included (several minutes).

Three conventions the suite enforces, all from `tests/conftest.py`:

- Test configs come from the `make_config` fixture and have `caching: False`. Only
  `tests/test_caching.py` turns caching on; `tests/test_meta_caching_policy.py` fails the run if
  another file does.
- Tests never touch the real `custom_voices/` or `cache/` directories or a real audio device. Use
  the `isolated_dirs`, `engine` and `fake_pipeline` fixtures rather than patching around them.
  The storage dirs and `playback` live in `kokoro_gui/engine/runtime.py`; patch them there, not on
  `kokoro_engine`.
- Verify mode is on for every test. The panels cache what they derive from a project (the run
  index, which clips are stale, clip placement), and in the suite each cached answer is also
  computed from scratch and compared. If your change edits the model in a way that doesn't
  reach `kokoro_gui/daw/revision.py`'s counters, a test fails with `StaleCacheError`. Set the
  attribute instead of mutating a list or dict in place, go through an undo command, or call
  `document.touch()`.

GUI tests build a real `QtTTSApp` through the `qt_app` fixture in `tests/gui_qt/conftest.py`,
with the engine replaced by `StubEngine`. Save and Open run on a thread; call
`qt_app.wait_for_project_io()` before asserting on the result. Typing defers the timeline
refresh; call `qt_app.flush_updates()` before reading the timeline after a keystroke.

The MCP server's tests (`tests/headless/test_mcp_server.py`) skip unless the SDK is there. CI doesn't install it, so run them
yourself when you touch `kokoro_gui/headless/`: `pip install -r requirements-mcp.txt`. Tests in `tests/headless/` use the
`fake` and `offline` fixtures, never a real model, network or audio device.

The integration suite (`pytest -m integration tests/integration -s`) does real synthesis and is
opt-in. It isn't run in CI.

## Pull requests

- Branch from the latest release branch and target it, never `main` (see [Which branch to work on](#which-branch-to-work-on)).
  One change per PR.
- `pytest` green locally before you push (that includes `tests/gui_qt/`; CI runs it in a non-required job). CI has
  to pass on Windows, Linux and macOS to merge.
- A new setting is threaded through `QtTTSApp._assemble_config` and covered in
  `tests/gui_qt/test_qt_config_assembly.py`.
- Logic that isn't about widgets goes in `kokoro_gui/core/`, not in a dock or `app.py`. A dock reads its widgets into plain
  values and calls the core, and the command line calls the same function. `core/` and `headless/` never import PySide6,
  `qt/app.py`, `qt/subprojects.py`, `qt/docks/` or `audio/transport.py`, and `tests/core/test_imports.py` and
  `tests/headless/test_imports.py` check that.
- If a user can see the change, update two files: a one-line bullet in the README's "What's new"
  (and its Features list if a line there is now wrong), and the full entry in the top version of
  `docs/changes.html`. If that version already has a tag, start an "Unreleased" one above it.
- Run `ruff check .` before a PR (`pip install -r requirements-test.txt` has it). It checks for
  syntax errors, undefined names, unused imports and variables, and mutable defaults, and CI runs
  it. There is no formatter: match the style of the file you're in.

## Adding an engine

An engine is one module with two classes, and nothing under `kokoro_gui/qt/`, `daw/` or `audio/`
changes. `tests/plugins/toneclone.py` is a complete small example.

1. A model: subclass `ModelBase` from `kokoro_gui/engine/runner.py`. Set `engine_id`,
   `sample_rate` and `concurrency` (`"per_thread"` if the model keeps its own per-thread state,
   `"shared"` for one instance the runner serializes), and implement `synthesize(text, voice,
   speed, lang_code, params)`, returning a `Synthesis` (mono float32 audio, and word timings if
   the model has them). Override `load`, `engine_version`, `cache_key_extra` or
   `resolve_voice_path` when the defaults don't fit. Wrap it with `EngineRunner(model)`.
2. An adapter: subclass `BackendHooksMixin` from `kokoro_gui/engines/base.py`. Give it `id`,
   `display_name`, `capabilities`, a `get_config_schema()` classmethod (a `lang_code` field with
   its languages as choices, a `voice` field, then `common_fields(...)` and your own fields), an
   `engine` property, and `cancel()`. Set `voice_kind`: `"named"` (override `builtin_voices`),
   `"embedding"` (a `.ext` file per voice, `voice_store = EmbeddingStore(id, ext)`) or
   `"reference"` (a wav and transcript per voice, `voice_store = ReferenceStore(id)`, which also
   gets you the Voice Reference editor). The store gives you voice listing and bundling.
   Reference engines assign saved voices through Voices (Save or Use); Settings shows the
   assignment read-only. Use `ReferenceStore(id, requires_transcript=False)` when text is
   optional, and consume reviewed text when supplied. Override
   `reference_transcription_duration(config)` if the model uses only an excerpt: the default
   is `None` for the full WAV, and the editor transcribes the reported duration on its worker.
3. Register it. In this repo, add a line to `BUILTIN_ENGINES` in `kokoro_gui/engines/__init__.py`
   and call `register_engine(...)` at the bottom of the module. As a separate package, declare an
   entry point in the `kokorogui.engines` group naming the module or the adapter class. An engine
   whose import fails is listed as "(not installed)" instead of stopping the app.
4. Add it to `tests/test_engine_contract.py`: its id in `ENGINES`, and either a fake for its model
   in the `contract_backend` fixture or a `make_contract_engine()` classmethod on the adapter that
   builds an engine without weights.

## Layout

- `kokoro_engine.py` is Kokoro's model and engine; `kokoro_gui/engine/` is the shared synthesis
  core: the `EngineRunner`, mixins per feature area, and `runtime.py`.
- `kokoro_gui/engines/` is the backend interface, the registry, the voice stores and the built-in
  backends (Kokoro, Audio8, LuxTTS, Dummy). Nothing there imports the document model.
- `kokoro_gui/daw/` is the document model: text, clips, tracks, characters, arrangement, dirty
  tracking, undo.
- `kokoro_gui/audio/` is the transport, mixer and read-time FX stage.
- `kokoro_gui/core/` is what the app does with engines, clips and configs, without widgets.
- `kokoro_gui/headless/` is the command line and the MCP server over `core/`.
- `kokoro_gui/qt/` is the PySide6 shell; every panel is a dock under `kokoro_gui/qt/docks/`.
- `docs/` is the GitHub Pages site.
