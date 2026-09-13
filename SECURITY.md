# Security

## Supported versions

Only the latest 4.x release gets fixes. 3.x is unsupported.

## Reporting

Use the "Report a vulnerability" button on the repository's Security tab (GitHub private
vulnerability reporting). Don't open a public issue for anything you think is exploitable. You
should hear back within a week.

## What counts

KokoroGUI is a desktop app that reads files you give it. The parts worth a second look:

- `.tbaw` project bundles are zip files from anywhere. `kokoro_gui/qt/project.py` refuses
  absolute, drive-relative, `..` and symlink entries before extracting, and refuses a bundle
  carrying a `.pt` voice file unless `torch >= 2.6`. A way past either of those is a bug we want
  to hear about.
- Voice, preset, mix and reference names come from the GUI and end up in file paths. Every
  resolver sanitizes with `os.path.basename()`. A name that escapes its directory is a bug.
- Text input: `.txt`, `.pdf` (`pypdf`) and `.epub` (`ebooklib` + BeautifulSoup) go through
  `kokoro_gui/engine/text_extraction.py`.
- The Audio8 TTS and ASR models load from Hugging Face with `trust_remote_code=True`. That's a
  property of those models, not something the app can turn off; the app never downloads anything
  else at runtime.

Out of scope: anything that needs the attacker to already run code as your user, and denial of
service by feeding the app an enormous document.
