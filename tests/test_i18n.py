"""The translation layer (`kokoro_gui/i18n.py`) and its language files."""
import ast
import importlib.util
import json
import os

import pytest

from kokoro_gui import i18n

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def _english():
    yield
    i18n.set_language("en")


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("i18n_tool", os.path.join(ROOT, "scripts", "i18n.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(folder, code, strings=None, plurals=None, name="Test"):
    path = folder / f"{code}.json"
    path.write_text(json.dumps({"language": code, "name": name, "strings": strings or {}, "plurals": plurals or {}}),
                    encoding="utf-8")
    return path


# -- lookups --------------------------------------------------------------------

def test_english_is_the_source_text():
    assert i18n.language() == "en"
    assert i18n.tr("Play") == "Play"
    assert i18n.tr("Saved {name}.", name="a") == "Saved a."
    assert i18n.tr("a lone { brace") == "a lone { brace"
    assert i18n.ntr("{n} clip", "{n} clips", 1) == "1 clip"
    assert i18n.ntr("{n} clip", "{n} clips", 0) == "0 clips"


def test_a_language_file_is_used_and_gaps_stay_english(tmp_path):
    _write(tmp_path, "de", {"Play": "Abspielen", "Stop": "", "Saved {name}.": "{name} gespeichert."},
           {"{n} clips": {"one": "{n} Clip", "other": "{n} Clips"}})
    assert i18n.set_language("de", str(tmp_path)) == "de"
    assert i18n.tr("Play") == "Abspielen"
    assert i18n.tr("Stop") == "Stop"
    assert i18n.tr("Not in the file") == "Not in the file"
    assert i18n.tr("Saved {name}.", name="x") == "x gespeichert."
    assert i18n.ntr("{n} clip", "{n} clips", 1) == "1 Clip"
    assert i18n.ntr("{n} clip", "{n} clips", 3) == "3 Clips"
    i18n.set_language("en")
    assert i18n.tr("Play") == "Play"


@pytest.mark.parametrize("bad", [
    "{nmae} gespeichert.", "gespeichert.", "{name gespeichert", "{name.__class__}", "{name[0]}", "{} gespeichert",
    "{name} {other}", "{name:{width}}", 5, None, ["{name}"],
])
def test_a_translation_with_other_placeholders_is_left_out(tmp_path, bad, capsys):
    _write(tmp_path, "de", {"Saved {name}.": bad})
    i18n.set_language("de", str(tmp_path))
    assert i18n.tr("Saved {name}.", name="x") == "Saved x."
    assert "bad entries" in capsys.readouterr().out


def test_a_format_the_value_cannot_take_falls_back_to_english(tmp_path):
    _write(tmp_path, "de", {"Saved {name}.": "{name:d} gespeichert."})
    i18n.set_language("de", str(tmp_path))
    assert i18n.tr("Saved {name}.", name="x") == "Saved x."


@pytest.mark.parametrize("content", ["", "[]", "{", '{"strings": []}', '{"plurals": 3}', "[" * 100000],
                         ids=["empty", "list", "cut", "strings", "plurals", "deep"])
def test_a_damaged_file_gives_english(tmp_path, content):
    (tmp_path / "de.json").write_text(content, encoding="utf-8")
    assert i18n.set_language("de", str(tmp_path)) == "en"
    assert i18n.available_languages(str(tmp_path)) == [("en", "English")]


@pytest.mark.parametrize("code", ["../de", "de/../en", "DE", "", "de.json", "c:de"])
def test_a_code_never_becomes_another_path(tmp_path, code):
    with pytest.raises(ValueError):
        i18n.catalog_path(code, str(tmp_path))
    assert i18n.set_language(code, str(tmp_path)) == "en"


def test_matching_the_system_languages(tmp_path):
    _write(tmp_path, "pt")
    _write(tmp_path, "zh_Hans")
    folder = str(tmp_path)
    assert i18n.match_language(["pt-BR", "en-US"], folder) == "pt"
    assert i18n.match_language(["zh-Hans-CN"], folder) == "zh_Hans"
    assert i18n.match_language(["en-GB", "pt"], folder) == "en"
    assert i18n.match_language(["fi"], folder) == "en"


@pytest.mark.parametrize("code, counts", [
    ("en", {1: "one", 0: "other", 2: "other"}),
    ("fr", {0: "one", 1: "one", 2: "other"}),
    ("ja", {1: "other", 5: "other"}),
    ("ru", {1: "one", 2: "few", 5: "many", 11: "many", 12: "many", 21: "one", 22: "few"}),
    ("pl", {1: "one", 2: "few", 5: "many", 21: "many", 22: "few"}),
    ("cs", {1: "one", 3: "few", 5: "other"}),
    ("ar", {0: "zero", 1: "one", 2: "two", 3: "few", 11: "many", 100: "other"}),
    ("pt_BR", {1: "one", 2: "other"}),
])
def test_plural_forms(code, counts):
    for n, form in counts.items():
        assert i18n.plural_form(code, n) == form, n
        assert form in i18n.plural_forms(code)


# -- the files and the source ---------------------------------------------------

def test_en_json_matches_the_source(tool):
    """Run `python scripts/i18n.py extract` when this fails."""
    assert i18n.read_catalog(i18n.catalog_path("en")) == tool.template()


def test_every_language_file_is_sound(tool):
    source = tool.template()
    for code in tool.language_codes():
        assert tool.problems(code, i18n.read_catalog(i18n.catalog_path(code)), source) == []


def test_every_source_string_is_a_format_string_or_plain(tool):
    source = tool.template()
    for text in (*source["strings"], *source["plurals"]):
        assert i18n.placeholders(text) is not None, text


def test_tr_is_never_called_while_a_module_loads(tool):
    """The language isn't known yet, so a table marks its strings with N_."""
    early = []
    for path in tool.source_files():
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        inside = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                inside.update(id(n) for statement in node.body for n in ast.walk(statement))
            elif isinstance(node, ast.Lambda):
                inside.update(id(n) for n in ast.walk(node.body))
        early += [f"{os.path.basename(path)}:{n.lineno}" for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("tr", "ntr")
                  and id(n) not in inside]
    assert early == []


def test_extract_refuses_an_f_string(tool, tmp_path):
    path = os.path.join(str(tmp_path), "x.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write('tr(f"Saved {name}")\n')
    with pytest.raises(tool.ExtractError):
        tool.scan(path, {}, {})


def test_merging_keeps_translations_and_drops_old_keys(tool):
    source = {"strings": {"Play": "Play", "Stop": "Stop"},
              "plurals": {"{n} clips": {"one": "{n} clip", "other": "{n} clips"}}}
    old = {"strings": {"Play": "Lire", "Gone": "Parti"}, "plurals": {"{n} clips": {"one": "{n} extrait"}}}
    new = tool.merged(source, "fr", "Français", old)
    assert new["strings"] == {"Play": "Lire", "Stop": ""}
    assert new["plurals"] == {"{n} clips": {"one": "{n} extrait", "other": ""}}
    assert tool.missing_count(new) == 2
    assert tool.problems("fr", new, source) == []


def test_check_reports_lost_edges_and_wrong_forms(tool):
    source = {"strings": {" (missing)": " (missing)"},
              "plurals": {"{n} clips": {"one": "{n} clip", "other": "{n} clips"}}}
    data = {"strings": {" (missing)": "(fehlt)"}, "plurals": {"{n} clips": {"other": "{n}"}}}
    found = tool.problems("de", data, source)
    assert len(found) == 2 and "ends differ" in found[0] and "forms" in found[1]


def test_i18n_has_no_qt_import():
    with open(i18n.__file__, encoding="utf-8") as f:
        assert "PySide6" not in f.read()
