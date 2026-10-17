"""Share files in the Characters dialog and the Audio FX dock (plan 40): the
export and import dialogs, and the buttons that run them. The format itself is
tested in tests/daw/test_share.py. The modal dialogs are answered by patching
`exec`, the file dialogs by passing `path`."""
import io
import json
import os
import wave
import zipfile

import pytest
from PySide6.QtWidgets import QDialog, QMessageBox

from kokoro_gui.daw import share
from kokoro_gui.daw.models import Character
from kokoro_gui.qt.characters_dialog import CharactersDialog
from kokoro_gui.qt.share_dialogs import ExportShareDialog, ImportShareDialog

ACCEPT = QDialog.DialogCode.Accepted
REJECT = QDialog.DialogCode.Rejected


def wav_bytes(frames: int = 800) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\x00\x00" * frames)
    return buf.getvalue()


@pytest.fixture
def messages(monkeypatch):
    """What the app told the user: `[(kind, text)]`."""
    seen = []
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda p, t, text, *a, **k: seen.append(("info", text))))
    monkeypatch.setattr(QMessageBox, "critical",
                        staticmethod(lambda p, t, text, *a, **k: seen.append(("error", text))))
    return seen


@pytest.fixture
def accept_dialogs(monkeypatch):
    monkeypatch.setattr(ExportShareDialog, "exec", lambda self: ACCEPT)
    monkeypatch.setattr(ImportShareDialog, "exec", lambda self: ACCEPT)


def fx_folder(qt_app):
    from kokoro_gui.qt import app as qt_app_module

    return qt_app_module.FX_PRESETS_DIR


def add_fx(qt_app, name="Radio", values=None, ir=None):
    folder = fx_folder(qt_app)
    os.makedirs(os.path.join(folder, "ir"), exist_ok=True)
    values = values or {"gain_db": 3, "reverb_enabled": True}
    if ir:
        values = {**values, "convolution_ir": ir}
        with open(os.path.join(folder, "ir", f"{ir}.wav"), "wb") as f:
            f.write(wav_bytes())
    with open(os.path.join(folder, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(values, f)


def make_share(path, characters=(), entries=None):
    manifest = {"format": share.FORMAT, "version": share.VERSION, "characters": list(characters)}
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("share.json", json.dumps(manifest))
        for name, data in (entries or {}).items():
            zf.writestr(name, data)
    return str(path)


# -- the export dialog ---------------------------------------------------------------


def test_the_export_dialog_ticks_the_preselected_character_only(qt_app):
    a, b = Character(name="A"), Character(name="B")
    dialog = ExportShareDialog(qt_app, [a, b], preselect={b.id})
    assert [c.name for c in dialog.selected_characters()] == ["B"]
    assert dialog.options() == (True, True)
    assert not dialog.fx_list.isVisibleTo(dialog)


def test_the_export_dialog_ticks_everything_with_no_preselection(qt_app):
    dialog = ExportShareDialog(qt_app, [Character(name="A"), Character(name="B")])
    assert [c.name for c in dialog.selected_characters()] == ["A", "B"]


def test_the_export_dialog_needs_something_ticked(qt_app):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QDialogButtonBox

    dialog = ExportShareDialog(qt_app, [Character(name="A")])
    ok = dialog.buttons.button(QDialogButtonBox.StandardButton.Ok)
    assert ok.isEnabled()
    dialog.character_list.item(0).setCheckState(Qt.CheckState.Unchecked)
    assert not ok.isEnabled()


def test_the_fx_only_export_dialog_hides_the_character_options(qt_app):
    dialog = ExportShareDialog(qt_app, fx_names=["Radio", "Hall"], preselect={"Hall"})
    assert dialog.selected_fx() == ["Hall"] and dialog.selected_characters() == []
    assert not dialog.include_voices.isVisibleTo(dialog) and not dialog.character_list.isVisibleTo(dialog)


# -- the Characters dialog -----------------------------------------------------------


def test_exporting_a_character_writes_a_share_file(qt_app, tmp_path, accept_dialogs, messages):
    add_fx(qt_app, "Radio", ir="Hall")
    dialog = CharactersDialog(qt_app)
    character = dialog._current
    character.preset_data["fx_preset"] = "Radio"
    character.name = "Alice"
    os.makedirs(tmp_path / "out")
    path = str(tmp_path / "out" / "alice")

    report = dialog.export_characters(path=path)

    assert report.characters == 1 and report.fx == ["Radio"] and report.irs == ["Hall"]
    assert os.path.isfile(path + ".kgshare")  # the extension is added
    assert messages and messages[-1][0] == "info" and "alice.kgshare" in messages[-1][1]
    with zipfile.ZipFile(path + ".kgshare") as zf:
        assert set(zf.namelist()) == {"share.json", "fx/Radio.json", "fx/ir/Hall.wav"}


def test_cancelling_the_export_writes_nothing(qt_app, tmp_path, monkeypatch):
    monkeypatch.setattr(ExportShareDialog, "exec", lambda self: REJECT)
    assert CharactersDialog(qt_app).export_characters(path=str(tmp_path / "x.kgshare")) is None
    assert not (tmp_path / "x.kgshare").exists()


def test_importing_adds_characters_to_the_library_and_reloads_the_list(qt_app, tmp_path, accept_dialogs, messages):
    path = make_share(tmp_path / "in.kgshare", [{"name": "Zed", "preset_data": {"speed": 1.5}}],
                      {"fx/Echo.json": json.dumps({"gain_db": 2})})
    dialog = CharactersDialog(qt_app)
    result = dialog.import_characters(path=path)
    assert sorted(result.imported) == ["Character Zed", "FX Echo"]
    (entry,) = [e for e in qt_app.character_library.list() if e.name == "Zed"]
    assert entry.preset_data == {"speed": 1.5}
    assert os.path.isfile(os.path.join(fx_folder(qt_app), "Echo.json"))
    # Not in the project until the user adds it.
    assert "Zed" not in [c.name for c in qt_app.document.characters]
    assert "Imported 2 items" in messages[-1][1]


def test_a_refused_file_says_why_and_writes_nothing(qt_app, tmp_path, accept_dialogs, messages):
    path = make_share(tmp_path / "bad.kgshare", entries={"../evil.json": "{}"})
    assert CharactersDialog(qt_app).import_characters(path=path) is None
    assert messages[-1][0] == "error" and "unsafe entry name" in messages[-1][1]
    assert not os.path.exists(os.path.join(fx_folder(qt_app), "evil.json"))


def test_cancelling_the_import_writes_nothing(qt_app, tmp_path, monkeypatch):
    monkeypatch.setattr(ImportShareDialog, "exec", lambda self: REJECT)
    before = len(qt_app.character_library.list())
    path = make_share(tmp_path / "in.kgshare", [{"name": "Zed"}])
    assert CharactersDialog(qt_app).import_characters(path=path) is None
    assert len(qt_app.character_library.list()) == before


# -- the import dialog ---------------------------------------------------------------


def test_the_import_dialog_offers_a_choice_only_where_a_name_is_taken(qt_app, tmp_path):
    add_fx(qt_app, "Radio", {"gain_db": -1})
    add_fx(qt_app, "Same", {"gain_db": 2})
    path = make_share(tmp_path / "in.kgshare", [{"name": "Zed"}], {
        "fx/Radio.json": json.dumps({"gain_db": 5}), "fx/Same.json": json.dumps({"gain_db": 2}),
        "fx/New.json": json.dumps({"gain_db": 1}), "voices/ghost/v.wav": wav_bytes()})
    info = share.inspect_share(path)
    items = share.plan_import(info, library=qt_app.character_library, fx_dir=fx_folder(qt_app))
    dialog = ImportShareDialog(qt_app, info, items)

    assert set(dialog._combos) == {"fx:Radio"}
    assert [dialog._combos["fx:Radio"].itemText(i) for i in range(3)] == ["Keep both", "Replace", "Skip"]
    # Ticked: the character, the clash (Keep both by default) and the new FX. Not the identical one, and not the
    # voice for an engine that isn't installed.
    assert dialog.choices() == {"character:0": share.KEEP, "fx:Radio": share.KEEP, "fx:New": share.KEEP}


def test_the_import_dialog_choice_reaches_the_import(qt_app, tmp_path, monkeypatch, messages):
    add_fx(qt_app, "Radio", {"gain_db": -1})
    path = make_share(tmp_path / "in.kgshare", entries={"fx/Radio.json": json.dumps({"gain_db": 5})})

    def answer(self):
        self._combos["fx:Radio"].setCurrentIndex(1)  # Replace
        return ACCEPT

    monkeypatch.setattr(ImportShareDialog, "exec", answer)
    result = qt_app.fx_dock.import_fx(path=path)
    assert result.imported == ["FX Radio"] and result.renamed == {}
    with open(os.path.join(fx_folder(qt_app), "Radio.json"), encoding="utf-8") as f:
        assert json.load(f)["gain_db"] == 5


def test_nothing_ticked_means_nothing_to_import(qt_app, tmp_path):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QDialogButtonBox

    path = make_share(tmp_path / "in.kgshare", [{"name": "Zed"}])
    info = share.inspect_share(path)
    items = share.plan_import(info, library=qt_app.character_library, fx_dir=fx_folder(qt_app))
    dialog = ImportShareDialog(qt_app, info, items)
    ok = dialog.buttons.button(QDialogButtonBox.StandardButton.Ok)
    assert ok.isEnabled()
    dialog.table.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
    assert dialog.choices() == {} and not ok.isEnabled()


# -- the Audio FX dock ---------------------------------------------------------------


def test_the_fx_dock_exports_the_fx_and_its_impulse_response(qt_app, tmp_path, accept_dialogs, messages):
    add_fx(qt_app, "Radio", ir="Hall")
    add_fx(qt_app, "Other")
    qt_app.fx_dock.refresh_presets()
    path = str(tmp_path / "radio.kgshare")
    report = qt_app.fx_dock.export_fx(names=["Radio"], path=path)
    assert report.characters == 0 and report.fx == ["Radio"] and report.irs == ["Hall"]
    assert qt_app.fx_dock.export_action.text() == "Export FX..."


def test_the_fx_dock_import_leaves_characters_and_voices_out(qt_app, tmp_path, accept_dialogs, messages):
    path = make_share(tmp_path / "in.kgshare", [{"name": "Zed"}], {
        "fx/Echo.json": json.dumps({"gain_db": 2}), "fx/ir/Room.wav": wav_bytes()})
    result = qt_app.fx_dock.import_fx(path=path)
    assert sorted(result.imported) == ["FX Echo", "Impulse response Room"]
    assert all(e.name != "Zed" for e in qt_app.character_library.list())
    # The combo lists the new FX without a click on Refresh.
    assert qt_app.fx_dock.preset_combo.findData("Echo") >= 0


def test_the_fx_dock_says_when_a_file_holds_no_fx(qt_app, tmp_path, accept_dialogs, messages):
    path = make_share(tmp_path / "in.kgshare", [{"name": "Zed"}])
    assert qt_app.fx_dock.import_fx(path=path) is None
    assert messages[-1][0] == "info" and "nothing of that kind" in messages[-1][1]


def test_an_exported_fx_survives_the_trip_through_the_dock(qt_app, tmp_path, accept_dialogs, messages):
    add_fx(qt_app, "Radio", {"gain_db": 7}, ir="Hall")
    path = str(tmp_path / "radio.kgshare")
    qt_app.fx_dock.export_fx(names=["Radio"], path=path)
    os.remove(os.path.join(fx_folder(qt_app), "Radio.json"))
    os.remove(os.path.join(fx_folder(qt_app), "ir", "Hall.wav"))
    result = qt_app.fx_dock.import_fx(path=path)
    assert sorted(result.imported) == ["FX Radio", "Impulse response Hall"]
    with open(os.path.join(fx_folder(qt_app), "Radio.json"), encoding="utf-8") as f:
        assert json.load(f) == {"gain_db": 7, "convolution_ir": "Hall"}
