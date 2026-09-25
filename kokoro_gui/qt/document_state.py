"""Load-or-create logic for wiring a `kokoro_gui.daw.models.Document` into
`QtTTSApp`. Pure functions, no Qt imports - mirrors `kokoro_gui/qt/settings.py`'s
own split (app.py owns file-path constants and *when* to save; this module
just knows *how* to produce a `Document` to start from).
"""
from kokoro_gui.daw.migration import migrate_legacy_settings_to_document
from kokoro_gui.daw.serialization import load_document


def load_or_create_document(document_path: str, settings: dict, library):
    """Loads `document_path` if it exists and parses; otherwise starts a
    fresh `Document` whose characters come from the character library
    (`library`, a `kokoro_gui.daw.library.CharacterLibrary`; see
    `migrate_legacy_settings_to_document`). A loaded document is trusted
    as-is; its linked characters are refreshed later, by the app's
    `resolve_characters` call."""
    doc = load_document(document_path)
    if doc is not None:
        return doc
    return migrate_legacy_settings_to_document(settings, library)
