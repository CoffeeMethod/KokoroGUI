"""Tests for kokoro_gui/qt/document_state.py's load-or-create logic. Plain
tmp_path/monkeypatch - no need for the full qt_app fixture since this module
has no Qt imports."""
import json

from kokoro_gui.daw.serialization import save_document
from kokoro_gui.daw.migration import DEFAULT_CHARACTER_NAME
from kokoro_gui.daw.models import Character, Document
from kokoro_gui.qt.document_state import load_or_create_document


def test_migrates_when_no_document_file_exists(tmp_path):
    presets_dir = tmp_path / "presets"
    presets_dir.mkdir()
    (presets_dir / "Alice.json").write_text(json.dumps({"voice": "af_bella"}), encoding="utf-8")

    doc = load_or_create_document(str(tmp_path / "document.json"), {}, str(presets_dir))

    assert [c.name for c in doc.characters] == ["Alice"]
    assert doc.text == ""


def test_migrates_seeding_default_character_when_no_presets(tmp_path):
    doc = load_or_create_document(
        str(tmp_path / "document.json"), {"voice": "af_bella"}, str(tmp_path / "presets")
    )
    assert doc.characters[0].name == DEFAULT_CHARACTER_NAME


def test_loads_existing_document_verbatim_without_remigrating(tmp_path):
    document_path = tmp_path / "document.json"
    existing = Document.from_plain_text("hello world", characters=[Character.from_preset_dict("Saved", {})])
    save_document(existing, str(document_path))

    presets_dir = tmp_path / "presets"
    presets_dir.mkdir()
    (presets_dir / "SomeoneNew.json").write_text(json.dumps({"voice": "af_bella"}), encoding="utf-8")

    doc = load_or_create_document(str(document_path), {}, str(presets_dir))

    assert doc.text == "hello world"
    assert [c.name for c in doc.characters] == ["Saved"]
