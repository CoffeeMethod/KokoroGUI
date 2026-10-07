"""Lexicon > Find words to check (plan 22): the rows the dialog lists, Hear
speaking a row's context with its answer applied, and Add writing whole word
rules that stale the clips they rewrite."""
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from kokoro_gui.daw.dirty import build_segments_from_results, compute_expected_cache_hash
from kokoro_gui.daw.models import Character
from kokoro_gui.engine.lexicon import apply_lexicon
from kokoro_gui.qt.harvest_dialog import ADD_COL, CONTEXT_COL, COUNT_COL, KIND_COL, SAY_COL, WORD_COL, HarvestDialog

TEXT = "He met Marcus at the harbour. Marcus paid $4.50 in 1999 and called NASA about Marcus."


def _open(qt_app, text=TEXT, dictionary=None):
    """The dialog over `text`, with the spelling dictionary stubbed to
    `dictionary` (None: not installed) so no test depends on a word list."""
    qt_app.document.text = text
    qt_app.spell_dictionary_for = lambda character=None: dictionary
    return qt_app.lexicon_dock.open_harvest()


def _clean_clip(qt_app, tmp_path, start, end, name):
    character = qt_app.document.characters[0]
    clip = qt_app.document.assign_character_to_range(start, end, character.id)
    path = str(tmp_path / f"{name}.wav")
    sf.write(path, np.full(2400, 0.2, dtype=np.float32), 24000)
    text = qt_app.document.clip_text(clip)
    expected = compute_expected_cache_hash(text, qt_app.document.effective_config_for_clip(clip))
    clip.segments = build_segments_from_results(expected, [{"text": text, "path": path, "duration": 0.1}])
    return clip


def _two_clips(qt_app, tmp_path):
    split = TEXT.index("Marcus paid")
    first = _clean_clip(qt_app, tmp_path, 0, split - 1, "a")
    second = _clean_clip(qt_app, tmp_path, split, len(TEXT), "b")
    assert qt_app.document.dirty_clips() == []
    return first, second


def test_the_lexicon_dock_button_opens_the_dialog_over_the_transcript(qt_app):
    qt_app.document.text = TEXT
    qt_app.spell_dictionary_for = lambda character=None: None
    qt_app.lexicon_dock.harvest_button.click()
    dialog = qt_app.lexicon_dock.harvest_dialog
    assert isinstance(dialog, HarvestDialog) and dialog.isVisible()
    assert dialog.words() == ["Marcus", "$4.50", "1999", "NASA"]
    dialog.close()


def test_rows_show_word_count_kind_and_context(qt_app):
    dialog = _open(qt_app)
    table = dialog.table
    assert table.item(0, WORD_COL).text() == "Marcus"
    assert table.item(0, COUNT_COL).text() == "3"
    assert table.item(0, KIND_COL).text() == "Name"
    assert "Marcus" in table.item(0, CONTEXT_COL).text()
    kinds = {table.item(r, WORD_COL).text(): table.item(r, KIND_COL).text() for r in range(table.rowCount())}
    assert kinds == {"Marcus": "Name", "$4.50": "Number", "1999": "Number", "NASA": "Acronym"}
    dialog.close()


def test_a_word_the_lexicon_already_covers_and_a_character_name_are_not_listed(qt_app):
    qt_app.settings["lexicon"] = [{"find": "NASA", "replace": "nassa", "mode": "word", "case": True}]
    qt_app.document.characters.append(Character(name="Marcus"))
    dialog = _open(qt_app)
    assert dialog.words() == ["$4.50", "1999"]
    dialog.close()


def test_unknown_words_need_the_dictionary_and_the_filters_hide_rows(qt_app):
    text = "The zephyrine wind met Marcus at 1999."
    dialog = _open(qt_app, text)
    assert not dialog.kind_checks["unknown"].isEnabled()
    assert dialog.words() == ["Marcus", "1999"]
    dialog.close()

    dialog = _open(qt_app, text, dictionary={"the", "wind", "met", "at", "marcus"})
    assert dialog.kind_checks["unknown"].isEnabled()
    assert set(dialog.words()) == {"zephyrine", "Marcus", "1999"}
    dialog.kind_checks["digits"].setChecked(False)
    assert set(dialog.words()) == {"zephyrine", "Marcus"}
    dialog.kind_checks["name"].setChecked(False)
    assert dialog.words() == ["zephyrine"]
    dialog.close()


def test_only_the_most_frequent_rows_are_shown(qt_app, monkeypatch):
    import kokoro_gui.qt.harvest_dialog as module

    monkeypatch.setattr(module, "MAX_ROWS", 2)
    dialog = _open(qt_app)
    assert dialog.row_count() == 2
    assert dialog.summary.text() == "Showing the 2 most frequent of 4."
    dialog.close()


def test_every_open_project_is_scanned(qt_app):
    child = SimpleNamespace(document=SimpleNamespace(text="Then Iris sailed to Lisbon.",
                                                      characters=[SimpleNamespace(name="Kid")]))
    qt_app.document.text = "He met Marcus."
    qt_app.open_projects = lambda: [SimpleNamespace(document=qt_app.document), child]
    try:
        dialog = _open(qt_app, "He met Marcus.")
        assert set(dialog.words()) == {"Marcus", "Iris", "Lisbon"}
        dialog.close()
    finally:
        del qt_app.open_projects  # the window closes through the real one


def test_add_writes_a_whole_word_rule_and_stales_the_clips_it_rewrites(qt_app, tmp_path):
    dialog = _open(qt_app)
    first, second = _two_clips(qt_app, tmp_path)

    dialog.set_say_as(0, "Mar-kus")
    assert dialog.add(0) is True

    assert qt_app.settings["lexicon"] == [{"find": "Marcus", "replace": "Mar-kus", "mode": "word", "case": True}]
    assert {c.id for c in qt_app.document.dirty_clips()} == {first.id, second.id}  # both clips say Marcus
    assert dialog.is_added(0)
    assert not dialog.table.cellWidget(0, ADD_COL).isEnabled()
    assert dialog.table.cellWidget(0, ADD_COL).text() == "Added"
    assert dialog.add(0) is False and len(qt_app.settings["lexicon"]) == 1
    dialog.close()


def test_a_rule_for_a_word_in_one_clip_leaves_the_other_clean(qt_app, tmp_path):
    dialog = _open(qt_app)
    first, second = _two_clips(qt_app, tmp_path)
    row = dialog.words().index("$4.50")
    dialog.set_say_as(row, "four fifty")
    dialog.add(row)
    assert qt_app.document.dirty_clips() == [second]
    assert first not in qt_app.document.dirty_clips()
    dialog.close()


def test_add_needs_an_answer(qt_app):
    dialog = _open(qt_app)
    assert dialog.add(0) is False
    dialog.set_say_as(0, "   ")
    assert dialog.add(0) is False
    assert qt_app.settings["lexicon"] == []
    dialog.close()


def test_add_all_filled_adds_each_answer_in_one_change(qt_app):
    dialog = _open(qt_app)
    changes = []
    qt_app.refresh_timeline = lambda *a, **k: changes.append(len(qt_app.settings["lexicon"]))
    dialog.set_say_as(dialog.words().index("Marcus"), "Mar-kus")
    dialog.set_say_as(dialog.words().index("1999"), "nineteen ninety-nine")
    dialog.set_say_as(dialog.words().index("NASA"), "")
    assert dialog.add_all_filled() == 2
    assert changes == [2]
    assert [(r["find"], r["replace"]) for r in qt_app.settings["lexicon"]] == [
        ("Marcus", "Mar-kus"), ("1999", "nineteen ninety-nine")]
    assert dialog.add_all_filled() == 0
    assert not dialog.add_all_button.isEnabled()
    dialog.close()


def test_what_was_typed_survives_a_filter_change(qt_app):
    dialog = _open(qt_app)
    dialog.set_say_as(0, "Mar-kus")
    dialog.kind_checks["name"].setChecked(False)
    dialog.kind_checks["name"].setChecked(True)
    assert dialog.table.item(0, SAY_COL).text() == "Mar-kus"
    dialog.close()


def test_hear_speaks_the_context_with_the_answer_and_the_saved_rules_applied_once(qt_app):
    qt_app.settings["lexicon"] = [{"find": "harbour", "replace": "harbor", "mode": "word", "case": False}]
    dialog = _open(qt_app)
    dialog.set_say_as(0, "Mar-kus")
    dialog.hear(0)

    (text, _voice, _speed, _path, extra), _kwargs = qt_app.engine.generate_preview.call_args
    expected = apply_lexicon(dialog._shown[0].context, qt_app.settings["lexicon"] + [dialog._rule(0)])
    assert text == expected
    assert "Mar-kus" in text and "Marcus" not in text
    # The rules are already in `text`; the engine must not apply them again.
    assert extra["lexicon"] == []
    # Listening adds nothing.
    assert len(qt_app.settings["lexicon"]) == 1
    dialog.close()


def test_hear_without_an_answer_speaks_the_context_under_the_saved_rules(qt_app):
    dialog = _open(qt_app)
    dialog.hear(0)
    (text, _voice, _speed, _path, extra), _kwargs = qt_app.engine.generate_preview.call_args
    assert text == dialog._shown[0].context
    assert extra["lexicon"] == qt_app.settings["lexicon"]
    dialog.close()
