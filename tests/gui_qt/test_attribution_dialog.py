"""Edit > Attribute Dialogue (plan 26): the review dialog over
`daw/attribution.plan`, its transcript highlight, and Apply as one undo
step."""
import copy

from kokoro_gui.daw.models import Character

TEXT = ('Marta came in.\n\n'
        '"Morning," said Marta.\n\n'
        '"Is it?" Jon asked.\n\n'
        '"It was."\n\n'
        'She sat down.\n\n'
        '"Hm," she said.')


def _setup(qt_app):
    marta = Character.from_preset_dict("Marta", {"voice": "af_heart"}, highlight_color="#ff0000")
    jon = Character.from_preset_dict("Jon", {"voice": "am_michael"}, highlight_color="#0000ff")
    qt_app.document.characters.extend([marta, jon])
    qt_app.document.text = TEXT
    qt_app.editor.load_text(TEXT)
    return marta, jon


def _owners(document) -> dict:
    """`{clip text: character id}`."""
    out = {}
    for clip in document.clips:
        start, end = document.clip_extent(clip.id)
        out[document.text[start:end]] = clip.character_id
    return out


def _open(qt_app):
    dialog = qt_app.attribute_dialogue()
    assert dialog is not None
    return dialog


def test_rows_appear_with_confidences(qt_app):
    marta, jon = _setup(qt_app)
    dialog = _open(qt_app)

    rows = dialog.rows()
    assert [TEXT[a.start:a.end] for a in rows] == ['"Morning,"', '"Is it?"', '"It was."', '"Hm,"']
    assert [a.character_id for a in rows] == [marta.id, jon.id, marta.id, None]
    assert [dialog.table.item(r, 2).text() for r in range(4)] == ["High", "High", "Medium", "Low"]
    assert dialog.table.item(3, 3).text() == "Pronoun tag: she said"
    dialog.reject()


def test_the_transcript_highlights_planned_quotes_and_low_ones_in_the_warning_color(qt_app):
    _setup(qt_app)
    dialog = _open(qt_app)

    marks = qt_app.editor.attribution_marks()
    assert [(TEXT[s:e], color) for s, e, color in marks] == [
        ('"Morning,"', "#ff0000"), ('"Is it?"', "#0000ff"), ('"It was."', "#ff0000"), ('"Hm,"', None)]
    dialog.reject()


def test_only_uncertain_hides_the_sure_rows(qt_app):
    _setup(qt_app)
    dialog = _open(qt_app)

    dialog.only_uncertain.setChecked(True)

    assert [dialog.table.isRowHidden(r) for r in range(4)] == [True, True, True, False]
    dialog.reject()


def test_clicking_a_row_selects_its_quote(qt_app):
    _setup(qt_app)
    dialog = _open(qt_app)

    dialog.table.setCurrentCell(1, 0)

    assert qt_app.editor.textCursor().selectedText() == '"Is it?"'
    dialog.reject()


def test_an_edited_row_is_used_on_apply(qt_app):
    marta, jon = _setup(qt_app)
    dialog = _open(qt_app)

    combo = dialog.table.cellWidget(3, 1)
    combo.setCurrentIndex(combo.findData(jon.id))
    assert dialog.chosen(3) == jon.id
    assert dialog.apply()

    assert _owners(qt_app.document) == {'"Morning,"': marta.id, '"Is it?"': jon.id, '"It was."': marta.id,
                                        '"Hm,"': jon.id}


def test_apply_is_one_undo_step(qt_app):
    marta, _jon = _setup(qt_app)
    document = qt_app.document
    before = _owners(document)
    depth = document.undo_stack.undo_depth()
    dialog = _open(qt_app)
    dialog.narrator_combo.setCurrentIndex(dialog.narrator_combo.findData(marta.id))
    commands = dialog.planned_commands()
    # The narration goes to the narrator too: "Marta came in.", "said
    # Marta.", "Jon asked.", "She sat down." and "she said.".
    assert len(commands) == 3 + 5

    assert dialog.apply()
    assert document.undo_stack.undo_depth() == depth + 1
    assert _owners(document)["Marta came in."] == marta.id
    assert qt_app.editor.attribution_marks() == []

    document.undo_stack.undo()

    assert _owners(document) == before


def test_cancel_changes_nothing_and_clears_the_highlight(qt_app):
    _setup(qt_app)
    document = qt_app.document
    before = copy.deepcopy(document.clips)
    depth = document.undo_stack.undo_depth()
    dialog = _open(qt_app)
    assert qt_app.editor.attribution_marks()

    dialog.reject()

    assert document.clips == before
    assert qt_app.editor.attribution_marks() == []
    assert document.undo_stack.undo_depth() == depth


def test_apply_plans_again_when_the_transcript_changed(qt_app):
    _setup(qt_app)
    dialog = _open(qt_app)
    qt_app.document.text = '"New," said Jon.'

    assert not dialog.apply()
    assert [qt_app.document.text[a.start:a.end] for a in dialog.rows()] == ['"New,"']
    dialog.reject()


def test_selection_scope_plans_only_the_selection(qt_app):
    _setup(qt_app)
    start = TEXT.index('"Is it?"')
    qt_app.editor.select_span(start, start + len('"Is it?" Jon asked.'))
    dialog = _open(qt_app)

    assert dialog.scope_combo.currentData() == "selection"
    assert [TEXT[a.start:a.end] for a in dialog.rows()] == ['"Is it?"']
    dialog.reject()


def test_the_menu_opens_one_dialog(qt_app):
    _setup(qt_app)
    first = _open(qt_app)
    assert qt_app.attribute_dialogue() is first
    assert qt_app.attribute_dialogue_action in qt_app.edit_menu.actions()
    first.reject()
