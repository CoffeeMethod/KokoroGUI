"""Options > Settings... (kokoro_gui/qt/settings_window.py): the program and
project settings, staged until Apply or OK."""
import json

import pytest

from kokoro_gui.qt.settings_window import PAGES


@pytest.fixture
def window(qt_app):
    window = qt_app.open_settings_window()
    yield window
    window.reject()


def _make_clip(qt_app, start=0, end=5, text="hello world"):
    qt_app.document.text = text
    character = qt_app.document.characters[0]
    return qt_app.document.assign_character_to_range(start, end, character.id)


def test_the_options_menu_opens_it_on_the_first_page(qt_app):
    assert qt_app.options_menu.actions()[0] is qt_app.settings_window_action
    qt_app.settings_window_action.trigger()
    window = qt_app.settings_window
    assert window.isVisible()
    assert [window.page_list.item(i).text() for i in range(window.page_list.count())] == list(PAGES)
    assert window.pages.currentIndex() == 0
    window.reject()


def test_apply_is_off_until_something_changes_and_nothing_is_written_before_it(qt_app, window):
    assert not window.apply_button.isEnabled()
    window.widgets["gap_s"].setValue(0.5)
    assert window.apply_button.isEnabled()
    assert "gap_s" not in qt_app.document.settings

    window.apply()

    assert qt_app.document.settings["gap_s"] == 0.5
    assert not window.apply_button.isEnabled()


def test_cancel_drops_staged_edits(qt_app, window):
    window.widgets["ripple"].setChecked(False)
    window.generation_form.widget_for("segment_target_words").setValue(12)
    window.reject()
    assert "ripple" not in qt_app.document.settings
    assert qt_app.settings.get("segment_target_words", 40) == 40


def test_ok_applies_and_closes(qt_app, window):
    window.widgets["auto_crossfade"].setChecked(True)
    window.buttons.accepted.emit()
    assert qt_app.document.settings["auto_crossfade"] is True
    assert not window.isVisible()


def test_the_settings_tab_has_only_voice_fields(qt_app):
    form = qt_app.settings_dock.schema_form
    for key in ("segment_target_words", "segment_at_paragraphs", "format", "num_threads", "caching"):
        assert form.widget_for(key) is None, key
    for key in ("lang_code", "voice", "speed"):
        assert form.widget_for(key) is not None, key
    qt_app.selection.clear()
    assert qt_app.settings_dock.scope_group.isHidden()


# -- General ------------------------------------------------------------------

def test_theme_follows_the_menu(qt_app, window):
    combo = window.widgets["theme"]
    other = "light" if combo.currentData() == "dark" else "dark"
    combo.setCurrentIndex(combo.findData(other))
    window.apply()
    assert qt_app.settings["theme"] == other
    assert qt_app.theme_actions[other].isChecked()


# -- Generation and Performance ---------------------------------------------------

def test_segmentation_lands_in_app_settings_and_get_state(qt_app, window):
    window.generation_form.widget_for("segment_target_words").setValue(12)
    window.generation_form.widget_for("segment_at_pauses").setChecked(False)
    window.apply()
    assert qt_app.settings["segment_target_words"] == 12
    state = qt_app.settings_dock.get_state()
    assert state["segment_target_words"] == 12 and state["segment_at_pauses"] is False
    assert qt_app._assemble_config()["segment_target_words"] == 12


def test_threads_are_per_engine(qt_app, window):
    import kokoro_gui.qt.app as qt_app_module

    window.engine_forms["kokoro"].widget_for("num_threads").setValue(4)
    window.apply()
    qt_app.save_settings()

    with open(qt_app_module.CONFIG_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["engines"]["kokoro"]["num_threads"] == 4
    assert qt_app.engine_settings("audio8")["num_threads"] == 1


def test_the_segment_cache_is_shared(qt_app, window):
    window.shared_performance_form.widget_for("caching").setChecked(False)
    window.apply()
    assert qt_app.settings["caching"] is False
    assert qt_app.settings_dock.get_state()["caching"] is False


# -- Project ------------------------------------------------------------------------

def test_pacing_fields_write_document_settings_undoably(qt_app, window):
    window.widgets["gap_s"].setValue(0.5)
    window.widgets["auto_crossfade"].setChecked(True)
    window.apply()
    assert qt_app.document.settings["gap_s"] == 0.5
    assert qt_app.document.settings["auto_crossfade"] is True

    qt_app.document.undo_stack.undo()
    qt_app.document.undo_stack.undo()
    assert "gap_s" not in qt_app.document.settings
    assert "auto_crossfade" not in qt_app.document.settings


def test_ripple_defaults_on_and_writes_the_setting(qt_app, window):
    ripple = window.widgets["ripple"]
    assert ripple.isChecked()
    ripple.setChecked(False)
    window.apply()
    assert qt_app.document.settings["ripple"] is False
    qt_app.document.undo_stack.undo()
    assert "ripple" not in qt_app.document.settings


def test_align_onset_shows_the_derived_default_and_an_untouched_box_writes_nothing(qt_app):
    clip = _make_clip(qt_app)
    window = qt_app.open_settings_window()
    assert not window.widgets["align_onset"].isChecked()  # no locked clip: off
    window.reject()

    clip.pinned = True
    window = qt_app.open_settings_window()
    align = window.widgets["align_onset"]
    assert align.isChecked()
    window.widgets["gap_s"].setValue(0.4)
    window.apply()
    assert "align_onset" not in qt_app.document.settings

    window.widgets["align_onset"].setChecked(False)
    window.apply()
    assert qt_app.document.settings["align_onset"] is False
    window.reject()


# -- Timeline -----------------------------------------------------------------------

def test_track_layout_switches_both_ways(qt_app):
    """Grill PR4: Unified puts clips on "Lane N" tracks by the lane rule;
    One per character puts them back on character tracks. Each switch is
    one undo step with its relane."""
    from kokoro_gui.daw.models import Character

    doc = qt_app.document
    alice = doc.characters[0]
    bob = Character.from_preset_dict("Bob", {})
    doc.characters.append(bob)
    doc.text = "one two three"
    first = doc.assign_character_to_range(0, 3, alice.id)
    second = doc.assign_character_to_range(4, 7, bob.id)
    third = doc.assign_character_to_range(8, 13, alice.id)
    character_track_ids = [c.track_id for c in (first, second, third)]
    window = qt_app.open_settings_window("Timeline")
    fields = window.widgets
    assert fields["track_layout"].currentData() == "character"
    assert not fields["track_lanes"].isEnabled()

    fields["track_layout"].setCurrentIndex(fields["track_layout"].findData("unified"))
    assert fields["track_lanes"].isEnabled()
    fields["track_lanes"].setValue(2)
    window.apply()

    assert doc.settings["track_layout"] == {"mode": "unified", "lanes": 2}
    assert [doc.get_track(c.track_id).name for c in (first, second, third)] == ["Lane 1", "Lane 2", "Lane 1"]
    assert [t.name for t in doc.used_tracks()] == ["Lane 1", "Lane 2"]

    fields = window.widgets
    fields["track_layout"].setCurrentIndex(fields["track_layout"].findData("character"))
    window.apply()
    assert [c.track_id for c in (first, second, third)] == character_track_ids

    doc.undo_stack.undo()
    assert [doc.get_track(c.track_id).lane for c in (first, second, third)] == [1, 2, 1]
    doc.undo_stack.undo()
    assert "track_layout" not in doc.settings
    assert [c.track_id for c in (first, second, third)] == character_track_ids
    window.reject()


def test_ducking_is_a_project_field(qt_app, window):
    spin = window.widgets["duck_db"]
    assert spin.value() == -12.0
    spin.setValue(-18.0)
    window.apply()
    assert qt_app.document.settings["duck_db"] == -18.0
    qt_app.document.undo_stack.undo()
    assert "duck_db" not in qt_app.document.settings


def test_timecode_fields_store_one_dict(qt_app, window):
    fields = window.widgets
    fields["tc_fps"].setCurrentIndex(fields["tc_fps"].findData(29.97))
    fields["tc_drop"].setChecked(True)
    fields["tc_start"].setText("01:00:00;00")
    fields["tc_enabled"].setChecked(True)
    window.apply()

    tc = qt_app.document.settings["timecode"]
    assert tc == {"enabled": True, "frame_rate": 29.97, "start": "01:00:00;00", "drop_frame": True}
    assert qt_app.transport_dock.time_label.text().startswith("01:00:00;00")

    window.widgets["tc_start"].setText("garbage")
    window.apply()
    assert qt_app.document.settings["timecode"]["start"] == "01:00:00;00"
