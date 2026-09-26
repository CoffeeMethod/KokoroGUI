# Contributing

Bug reports and pull requests are welcome. Questions go in Discussions, security reports through
the Security tab (see [SECURITY.md](SECURITY.md)).

## Setup

```bash
git clone https://github.com/CoffeeMethod/KokoroGUI.git
cd KokoroGUI
python -m venv .venv && . .venv/Scripts/activate   # or .venv/bin/activate
pip install -r requirements.txt -r requirements-test.txt
python main.py
```

Python 3.11 or newer. eSpeak NG is only needed to actually synthesize with Kokoro; the fast test
suite runs without it.

## Tests

```bash
pytest
```

That's the fast suite: the Kokoro pipeline is mocked, playback is mocked, no model download, runs
in well under a minute. The Qt tests run headless (`tests/gui_qt/conftest.py` sets
`QT_QPA_PLATFORM=offscreen`), so no display is needed. CI runs the suite on `windows-latest` and
`ubuntu-latest` without `tests/gui_qt/` (`pytest --ignore=tests/gui_qt -p no:pytest-qt`), so the
GUI tests only run on your machine. Run plain `pytest` before you push.

Two conventions the suite enforces, both from `tests/conftest.py`:

- Test configs come from the `make_config` fixture and have `caching: False`. Only
  `tests/test_caching.py` turns caching on; `tests/test_meta_caching_policy.py` fails the run if
  another file does.
- Tests never touch the real `custom_voices/` or `cache/` directories or a real audio device. Use
  the `isolated_dirs`, `engine` and `fake_pipeline` fixtures rather than patching around them.
  The storage dirs and `playback` live in `kokoro_gui/engine/runtime.py`; patch them there, not on
  `kokoro_engine`.

GUI tests build a real `QtTTSApp` through the `qt_app` fixture in `tests/gui_qt/conftest.py`,
with the engine replaced by `StubEngine`. Save and Open run on a thread; call
`qt_app.wait_for_project_io()` before asserting on the result.

The integration suite (`pytest -m integration tests/integration -s`) does real synthesis and is
opt-in. It isn't run in CI.

## Pull requests

- Branch from `main`, one change per PR.
- `pytest` green locally before you push (that includes `tests/gui_qt/`, which CI skips). CI has
  to pass on both OSes to merge.
- A new setting is threaded through `QtTTSApp._assemble_config` and covered in
  `tests/gui_qt/test_qt_config_assembly.py`.
- If a user can see the change, update two files: a one-line bullet in the README's "What's new"
  (and its Features list if a line there is now wrong), and the full entry under "Unreleased" in
  `docs/changes.html`.
- No formatter or linter is configured. Match the style of the file you're in.

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
- `kokoro_gui/engines/` is the backend interface, the registry, the voice stores and the three
  built-in backends (Kokoro, Audio8, Dummy). Nothing there imports the document model.
- `kokoro_gui/daw/` is the document model: text, clips, tracks, characters, arrangement, dirty
  tracking, undo.
- `kokoro_gui/audio/` is the transport, mixer and read-time FX stage.
- `kokoro_gui/qt/` is the PySide6 shell; every panel is a dock under `kokoro_gui/qt/docks/`.
- `docs/` is the GitHub Pages site.
