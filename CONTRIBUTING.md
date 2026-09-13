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
in well under a minute. CI runs the same command on `windows-latest` and `ubuntu-latest` with
`QT_QPA_PLATFORM=offscreen`, so the Qt tests need no display.

Two conventions the suite enforces, both from `tests/conftest.py`:

- Test configs come from the `make_config` fixture and have `caching: False`. Only
  `tests/test_caching.py` turns caching on; `tests/test_meta_caching_policy.py` fails the run if
  another file does.
- Tests never touch the real `custom_voices/` or `cache/` directories or a real audio device. Use
  the `isolated_dirs`, `engine` and `fake_pipeline` fixtures rather than patching around them.

GUI tests build a real `QtTTSApp` through the `qt_app` fixture in `tests/gui_qt/conftest.py`,
with the engine replaced by `StubEngine`. Save and Open run on a thread; call
`qt_app.wait_for_project_io()` before asserting on the result.

The integration suite (`pytest -m integration tests/integration -s`) does real synthesis and is
opt-in. It isn't run in CI.

## Pull requests

- Branch from `main`, one change per PR.
- `pytest` green locally before you push. CI has to pass on both OSes to merge.
- A new setting is threaded through `QtTTSApp._assemble_config` and covered in
  `tests/gui_qt/test_qt_config_assembly.py`.
- If a user can see the change, update the README: the Features list, and a bullet under the
  current "New in" heading.
- No formatter or linter is configured. Match the style of the file you're in.

## Layout

- `kokoro_engine.py` and `kokoro_gui/engine/` are the synthesis core (mixins per feature area).
- `kokoro_gui/engines/` is the backend interface and the three registered backends (Kokoro,
  Audio8, Dummy). Nothing there imports the document model.
- `kokoro_gui/daw/` is the document model: text, clips, tracks, characters, arrangement, dirty
  tracking, undo.
- `kokoro_gui/audio/` is the transport, mixer and read-time FX stage.
- `kokoro_gui/qt/` is the PySide6 shell; every panel is a dock under `kokoro_gui/qt/docks/`.
- `docs/` is the GitHub Pages site.
