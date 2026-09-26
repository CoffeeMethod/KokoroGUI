"""Tests for the Audio FX tab's selection scoping (UI6, section 3 of
Claude/PLAN_ui_shell_redesign.md): project / character / clip modes, the
debounced clip override, the character-preset confirmation, and the
timeline FX button raising the tab."""
import json
import os

from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QMessageBox

from kokoro_gui.daw.models import Character


def _type(editor, text):
    cursor = editor.textCursor()
    cursor.select(QTextCursor.SelectionType.Document)
    cursor.insertText(text)


def _write_fx_preset(qt_app, name, values):
    import kokoro_gui.qt.app as qt_app_module

    os.makedirs(qt_app_module.FX_PRESETS_DIR, exist_ok=True)
    with open(os.path.join(qt_app_module.FX_PRESETS_DIR, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(values, f)
    qt_app.fx_dock.refresh_presets()


def _clip_for(qt_app, character, text="hello world"):
    _type(qt_app.editor, text)
    clip = qt_app.document.assign_character_to_range(0, len(text), character.id)
    qt_app.editor.rehighlight()
    return clip


def test_starts_in_project_mode_and_follows_selection(qt_app):
    fx = qt_app.fx_dock
    assert fx.mode == "none"
    alice = qt_app.document.characters[0]
    clip = _clip_for(qt_app, alice)

    qt_app.selection.select_clip(clip.id)
    assert fx.mode == "clip"
    qt_app.selection.select_character(alice.id)
    assert fx.mode == "character"
    qt_app.selection.clear()
    assert fx.mode == "none"


def test_project_mode_edits_stay_project_wide(qt_app):
    fx = qt_app.fx_dock
    fx._value_widgets["gain_db"].setValue(4.0)
    assert fx.project_fx_state()["gain_db"] == 4.0
    assert qt_app._assemble_config()["gain_db"] == 4.0


def test_clip_mode_shows_the_resolved_stack_and_edits_become_one_undoable_override(qt_app):
    fx = qt_app.fx_dock
    _write_fx_preset(qt_app, "Warm", {"eq_bass": 3.0})
    warm = Character.from_preset_dict("Warm", {"fx_preset": "Warm"})
    qt_app.document.characters.append(warm)
    clip = _clip_for(qt_app, warm)
    fx._value_widgets["gain_db"].setValue(2.0)  # project value, inherited below

    qt_app.selection.select_clip(clip.id)

    assert fx.mode == "clip"
    assert fx._value_widgets["eq_bass"].value() == 3.0  # from the character's preset
    assert fx._value_widgets["gain_db"].value() == 2.0  # from the project state
    assert fx.preset_combo.currentText() == "Warm"

    fx._value_widgets["eq_treble"].setValue(5.0)
    fx._value_widgets["eq_treble"].setValue(6.0)
    fx._flush_clip_edit()

    assert clip.fx_override["eq_treble"] == 6.0
    assert clip.fx_override["eq_bass"] == 3.0  # the whole resolved dict is the override
    assert fx.preset_combo.currentText() == "(custom)"
    assert fx.project_fx_state()["eq_treble"] == 0.0  # project untouched
    qt_app.undo()
    assert clip.fx_override is None


def test_character_mode_writes_the_preset_file_after_confirming_once(qt_app, monkeypatch):
    fx = qt_app.fx_dock
    _write_fx_preset(qt_app, "Warm", {"eq_bass": 3.0})
    warm = Character.from_preset_dict("Warm", {"fx_preset": "Warm"})
    qt_app.document.characters.append(warm)
    asked = []

    def _question(*args, **kwargs):
        asked.append(args[2])
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", staticmethod(_question))
    qt_app.selection.select_character(warm.id)
    assert fx.mode == "character"

    fx._value_widgets["eq_bass"].setValue(-2.0)
    fx._value_widgets["eq_treble"].setValue(1.5)

    import kokoro_gui.qt.app as qt_app_module

    with open(os.path.join(qt_app_module.FX_PRESETS_DIR, "Warm.json"), encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["eq_bass"] == -2.0 and saved["eq_treble"] == 1.5
    assert len(asked) == 1 and "Warm" in asked[0]  # confirmed once per session


def test_character_mode_declined_confirmation_reverts_the_widget(qt_app, monkeypatch):
    fx = qt_app.fx_dock
    _write_fx_preset(qt_app, "Warm", {"eq_bass": 3.0})
    warm = Character.from_preset_dict("Warm", {"fx_preset": "Warm"})
    qt_app.document.characters.append(warm)
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.No))
    qt_app.selection.select_character(warm.id)

    fx._value_widgets["eq_bass"].setValue(-2.0)

    assert fx._value_widgets["eq_bass"].value() == 3.0


def test_character_without_preset_gets_one_named_after_it(qt_app, monkeypatch):
    fx = qt_app.fx_dock
    bare = Character.from_preset_dict("Bare", {})
    qt_app.document.characters.append(bare)
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
    qt_app.selection.select_character(bare.id)

    fx._value_widgets["gain_db"].setValue(1.0)

    import kokoro_gui.qt.app as qt_app_module

    assert bare.preset_data["fx_preset"] == "Bare"
    assert os.path.exists(os.path.join(qt_app_module.FX_PRESETS_DIR, "Bare.json"))


def test_preset_combo_in_clip_mode_applies_a_named_override(qt_app):
    fx = qt_app.fx_dock
    _write_fx_preset(qt_app, "Telephone", {"highpass_enabled": True, "highpass_freq": 300.0})
    qt_app.engine.load_fx_preset.return_value = {"highpass_enabled": True, "highpass_freq": 300.0}
    alice = qt_app.document.characters[0]
    clip = _clip_for(qt_app, alice)
    qt_app.selection.select_clip(clip.id)

    index = fx.preset_combo.findText("Telephone")
    fx._on_preset_activated(index)

    assert clip.overrides["fx_preset"] == "Telephone"
    assert clip.fx_override["highpass_freq"] == 300.0
    assert fx.preset_combo.currentText() == "Telephone"


def test_timeline_fx_button_selects_the_clip_and_raises_the_tab(qt_app):
    _write_fx_preset(qt_app, "Telephone", {"highpass_enabled": True})
    qt_app.engine.load_fx_preset.return_value = {"highpass_enabled": True}
    alice = qt_app.document.characters[0]
    clip = _clip_for(qt_app, alice)
    raised = []
    qt_app.raise_fx_tab = lambda: raised.append(True)

    qt_app.timeline_dock.on_fx_preset_requested(clip.id, "Telephone")

    assert qt_app.selection.selected_clip_id == clip.id
    assert raised == [True]
    assert qt_app.fx_dock.mode == "clip"
    assert clip.overrides["fx_preset"] == "Telephone"


def test_get_state_reflects_widgets_while_project_state_survives_scope_changes(qt_app):
    fx = qt_app.fx_dock
    fx._value_widgets["gain_db"].setValue(7.0)
    alice = qt_app.document.characters[0]
    clip = _clip_for(qt_app, alice)
    clip.fx_override = {"gain_db": -3.0}
    qt_app.selection.select_clip(clip.id)

    assert fx.get_state()["gain_db"] == -3.0
    assert fx.project_fx_state()["gain_db"] == 7.0
    qt_app.selection.clear()
    assert fx.get_state()["gain_db"] == 7.0


# -- convolution reverb impulse response field (grill Q31) -----------------------

def _write_ir(path, value=1.0):
    import numpy as np
    import soundfile as sf

    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, np.array([value], dtype=np.float32), 24000, subtype="FLOAT")
    return path


def _global_ir_dir():
    import kokoro_gui.qt.app as qt_app_module

    return os.path.join(qt_app_module.FX_PRESETS_DIR, "ir")


def _combo_names(combo):
    return [combo.itemData(i) for i in range(combo.count())]


def test_ir_field_lists_project_and_global_names_under_none(qt_app):
    fx = qt_app.fx_dock
    _write_ir(os.path.join(_global_ir_dir(), "Hall.wav"))
    _write_ir(os.path.join(qt_app.project_dir, "fx", "ir", "Cave.wav"))
    _write_ir(os.path.join(qt_app.project_dir, "fx", "ir", "Hall.wav"))
    fx.refresh_presets()

    combo = fx._file_combos["convolution_ir"]
    assert combo.itemText(0) == "None"
    assert _combo_names(combo) == ["", "Cave", "Hall"]
    assert fx.get_state()["convolution_ir"] == ""
    assert "convolution_mix" in fx._value_widgets


def test_picking_an_ir_feeds_the_clip_post_config(qt_app):
    from kokoro_gui.audio import post

    fx = qt_app.fx_dock
    _write_ir(os.path.join(_global_ir_dir(), "Hall.wav"))
    fx.refresh_presets()
    combo = fx._file_combos["convolution_ir"]
    combo.setCurrentIndex(combo.findData("Hall"))
    fx._value_widgets["convolution_mix"].setValue(0.8)

    assert fx.project_fx_state()["convolution_ir"] == "Hall"
    clip = _clip_for(qt_app, qt_app.document.characters[0])
    config = qt_app.post_config_for_clip(clip)
    assert config["convolution_ir"] == "Hall"
    assert config["convolution_mix"] == 0.8
    assert config["project_dir"] == qt_app.project_dir
    assert post.post_key(config) != post.post_key(dict(config, convolution_ir=""))


def test_ir_picked_in_clip_mode_becomes_the_clip_override(qt_app, qtbot):
    fx = qt_app.fx_dock
    _write_ir(os.path.join(_global_ir_dir(), "Hall.wav"))
    fx.refresh_presets()
    clip = _clip_for(qt_app, qt_app.document.characters[0])
    qt_app.selection.select_clip(clip.id)

    combo = fx._file_combos["convolution_ir"]
    combo.setCurrentIndex(combo.findData("Hall"))
    fx._flush_clip_edit()

    assert clip.fx_override["convolution_ir"] == "Hall"
    assert fx.project_fx_state()["convolution_ir"] == ""


def test_an_ir_name_that_resolves_nowhere_is_kept_and_marked_missing(qt_app):
    fx = qt_app.fx_dock
    clip = _clip_for(qt_app, qt_app.document.characters[0])
    clip.fx_override = {"convolution_ir": "Gone"}
    qt_app.selection.select_clip(clip.id)

    combo = fx._file_combos["convolution_ir"]
    assert combo.currentText() == "Gone (missing)"
    assert fx.get_state()["convolution_ir"] == "Gone"


def test_add_copies_a_wav_into_the_global_store_under_a_clean_name(qt_app, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    fx = qt_app.fx_dock
    src_dir = tmp_path / "downloads"
    first = _write_ir(str(src_dir / "Big Hall.wav"), 1.0)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (first, "")))
    fx._add_ir_file("convolution_ir")

    assert os.path.isfile(os.path.join(_global_ir_dir(), "Big Hall.wav"))
    assert fx.get_state()["convolution_ir"] == "Big Hall"

    # A different file with the same stem gets a numbered name; the same
    # file again reuses the stored one.
    other = _write_ir(str(src_dir / "other" / "Big Hall.wav"), 0.5)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (other, "")))
    fx._add_ir_file("convolution_ir")
    assert fx.get_state()["convolution_ir"] == "Big Hall 2"
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (first, "")))
    fx._add_ir_file("convolution_ir")
    assert fx.get_state()["convolution_ir"] == "Big Hall"
    assert sorted(os.listdir(_global_ir_dir())) == ["Big Hall 2.wav", "Big Hall.wav"]


def test_add_refuses_a_file_that_is_not_audio(qt_app, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    bogus = tmp_path / "notes.wav"
    bogus.write_bytes(b"hello")
    errors = []
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(bogus), "")))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: errors.append(a)))

    qt_app.fx_dock._add_ir_file("convolution_ir")

    assert errors
    assert not os.path.exists(os.path.join(_global_ir_dir(), "notes.wav"))


def test_fx_presets_resolve_ir_prefers_the_project_copy(qt_app):
    from kokoro_gui.qt.fx_presets import resolve_ir

    global_hall = _write_ir(os.path.join(_global_ir_dir(), "Hall.wav"))
    assert resolve_ir("Hall", qt_app.project_dir) == os.path.realpath(global_hall)
    local_hall = _write_ir(os.path.join(qt_app.project_dir, "fx", "ir", "Hall.wav"))
    assert resolve_ir("Hall", qt_app.project_dir) == os.path.realpath(local_hall)
    assert resolve_ir("Hall", None) == os.path.realpath(global_hall)
    assert resolve_ir("Other", qt_app.project_dir) is None


def test_loading_a_preset_with_a_non_string_ir_ignores_it(qt_app):
    fx = qt_app.fx_dock
    _write_fx_preset(qt_app, "Bad", {"convolution_ir": {"path": "/etc/passwd"}, "gain_db": 2.0,
                                     "eq_bass": "loud"})
    fx.load_preset("Bad")

    assert fx.get_state()["convolution_ir"] == ""
    assert fx.get_state()["gain_db"] == 2.0
    assert fx.get_state()["eq_bass"] == 0.0


def test_import_ir_file_drops_control_characters_from_the_name(qt_app, tmp_path):
    import numpy as np
    import soundfile as sf

    from kokoro_gui.qt import fx_presets

    src = tmp_path / "Big\tHall.wav"
    sf.write(str(src), np.array([1.0], dtype=np.float32), 24000)

    name = fx_presets.import_ir_file(str(src))

    assert name == "BigHall"
    assert fx_presets.resolve_ir(name) is not None
    assert name in fx_presets.list_ir_names()
