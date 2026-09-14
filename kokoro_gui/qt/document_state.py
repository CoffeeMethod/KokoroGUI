"""Load-or-create logic for wiring a `kokoro_gui.daw.models.Document` into
`QtTTSApp`. Pure functions, no Qt imports - mirrors `kokoro_gui/qt/settings.py`'s
own split (app.py owns file-path constants and *when* to save; this module
just knows *how* to produce a `Document` to start from).
"""
from kokoro_gui.daw.migration import migrate_legacy_settings_to_document
from kokoro_gui.daw.serialization import load_document


def load_or_create_document(document_path: str, settings: dict, presets_dir: str):
    """Loads `document_path` if it exists and parses; otherwise migrates
    today's presets/settings into a fresh `Document` (see
    `migrate_legacy_settings_to_document`). Migration is a one-time
    bootstrap, not an ongoing sync - once a `document.json` exists, it's
    trusted as-is even if `presets_dir`'s contents have since changed."""
    doc = load_document(document_path)
    if doc is not None:
        return doc
    return migrate_legacy_settings_to_document(settings, presets_dir)
