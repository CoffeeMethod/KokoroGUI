"""Tests for Options > Transcript details (grill TE7-TE10): segment shading,
bars at pause and forced cuts, lexicon overlines, gap labels, the gutter's
status dot and length, and the caret strip."""
import json

from PySide6.QtGui import QPaintEvent, QTextCharFormat, QTextCursor

from kokoro_gui.daw.dirty import build_segments_from_results, compute_expected_cache_hash
from kokoro_gui.engine.segmenting import PAUSE, WORD
from kokoro_gui.qt.transcript_editor import HIGHLIGHT_ALPHA, SEGMENT_ALT_ALPHA

SENTENCES = " ".join(f"w{i} w{i + 1} w{i + 2} w{i + 3} w{i + 4}." for i in range(1, 30, 5))  # 6 x 5 words


def _type(editor, text):
    cursor = editor.textCursor()
    cursor.select(QTextCursor.SelectionType.Document)
    cursor.insertText(text)


def _place_caret(editor, position):
    cursor = editor.textCursor()
    cursor.setPosition(position)
    editor.setTextCursor(cursor)


def _format_at(editor, position):
    block = editor.document().findBlock(position)
    offset = position - block.position()
    for fmt_range in block.layout().formats():
        if fmt_range.start <= offset < fmt_range.start + fmt_range.length:
            return QTextCharFormat(fmt_range.format)  # a copy outlives the range
    return None


def _alpha_at(editor, position):
    fmt = _format_at(editor, position)
    return fmt.background().color().alpha() if fmt is not None else None


def _target_words(qt_app, value):
    window = qt_app.open_settings_window("Generation")
    window.generation_form.widget_for("segment_target_words").setValue(value)
    window.apply()
    window.reject()


def _one_clip(qt_app, text=SENTENCES):
    _type(qt_app.editor, text)
    alice = qt_app.document.characters[0]
    clip = qt_app.document.assign_character_to_range(0, len(text), alice.id)
    qt_app.editor.rehighlight()
    return clip


def _details(qt_app, on=True):
    qt_app.details_action.setChecked(on)


# -- the setting ---------------------------------------------------------------------------


def test_details_are_off_by_default(qt_app):
    assert qt_app.settings["transcript_details"] is False
    assert not qt_app.details_action.isChecked()
    assert all(not on for on in qt_app.details_flags().values())
    assert all(not action.isEnabled() for action in qt_app.details_layer_actions.values())


def test_show_details_turns_every_layer_on_and_enables_the_toggles(qt_app):
    _details(qt_app)
    assert qt_app.settings["transcript_details"] is True
    assert all(qt_app.details_flags().values())
    assert all(action.isEnabled() for action in qt_app.details_layer_actions.values())
    qt_app.details_layer_actions["details_lexicon"].setChecked(False)
    assert qt_app.settings["details_lexicon"] is False
    assert qt_app.details_flags()["details_lexicon"] is False
    assert qt_app.details_flags()["details_segments"] is True


def test_details_settings_persist(qt_app):
    import kokoro_gui.qt.app as qt_app_module

    _details(qt_app)
    qt_app.details_layer_actions["details_gaps"].setChecked(False)
    qt_app.save_settings()
    with open(qt_app_module.CONFIG_FILE, encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["transcript_details"] is True
    assert saved["details_gaps"] is False


# -- segment shading and bars ----------------------------------------------------------------


def test_details_off_keeps_one_tint_per_clip(qt_app):
    _target_words(qt_app, 10)
    _one_clip(qt_app)
    assert {_alpha_at(qt_app.editor, i) for i in range(0, len(SENTENCES), 7)} == {HIGHLIGHT_ALPHA}


def test_every_other_segment_gets_the_lighter_tint(qt_app):
    _target_words(qt_app, 10)
    _one_clip(qt_app)
    _details(qt_app)
    second = SENTENCES.index("w11")
    third = SENTENCES.index("w21")
    assert _alpha_at(qt_app.editor, 0) == HIGHLIGHT_ALPHA
    assert _alpha_at(qt_app.editor, second) == SEGMENT_ALT_ALPHA
    assert _alpha_at(qt_app.editor, third) == HIGHLIGHT_ALPHA


def test_shading_follows_the_segmentation_setting(qt_app):
    _target_words(qt_app, 10)
    _one_clip(qt_app)
    _details(qt_app)
    _target_words(qt_app, 5)
    assert _alpha_at(qt_app.editor, SENTENCES.index("w6")) == SEGMENT_ALT_ALPHA


def test_shading_off_with_its_toggle(qt_app):
    _target_words(qt_app, 10)
    _one_clip(qt_app)
    _details(qt_app)
    qt_app.details_layer_actions["details_segments"].setChecked(False)
    assert _alpha_at(qt_app.editor, SENTENCES.index("w11")) == HIGHLIGHT_ALPHA
    assert qt_app.editor.segment_marks() == []


def test_bars_mark_pause_and_forced_cuts_only(qt_app):
    _target_words(qt_app, 10)
    words = " ".join(f"w{i}" for i in range(1, 36))  # no punctuation: forced cuts
    _one_clip(qt_app, words + ". a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12, b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11.")
    _details(qt_app)
    levels = [level for _offset, level in qt_app.editor.segment_marks()]
    assert WORD in levels and PAUSE in levels
    text = qt_app.document.text
    assert (text.index("w11"), WORD) in qt_app.editor.segment_marks()
    assert (text.index("b1 "), PAUSE) in qt_app.editor.segment_marks()


def test_sentence_cuts_get_no_bar(qt_app):
    _target_words(qt_app, 10)
    _one_clip(qt_app)
    _details(qt_app)
    assert qt_app.editor.segment_marks() == []


def test_painting_with_details_on_does_not_raise(qt_app):
    _target_words(qt_app, 10)
    _one_clip(qt_app, " ".join(f"w{i}" for i in range(1, 36)))
    _details(qt_app)
    qt_app.editor.resize(400, 300)
    qt_app.editor.paintEvent(QPaintEvent(qt_app.editor.viewport().rect()))


# -- lexicon rewrites ------------------------------------------------------------------------


def test_lexicon_rewrites_are_marked_and_explained(qt_app):
    qt_app.settings["lexicon"] = {"Dr.": "Doctor"}
    text = "Ask Dr. Smith now."
    _one_clip(qt_app, text)
    assert qt_app.editor.rewrite_spans() == []
    _details(qt_app)
    start = text.index("Dr.")
    assert qt_app.editor.rewrite_spans() == [(start, start + 3)]
    assert qt_app.editor.details_tooltip_at(start).startswith("Spoken as: Doctor")
    assert qt_app.editor.details_tooltip_at(text.index("Smith")).startswith("Segment 1 of 1")
    qt_app.details_layer_actions["details_lexicon"].setChecked(False)
    assert qt_app.editor.rewrite_spans() == []


# -- tooltips ----------------------------------------------------------------------------------


def test_segment_tooltip_on_a_stale_clip_gives_an_estimate(qt_app):
    _target_words(qt_app, 10)
    _one_clip(qt_app)
    _details(qt_app)
    tip = qt_app.editor.details_tooltip_at(SENTENCES.index("w12"))
    assert tip.startswith("Segment 2 of 3 · 10 words · ends at a sentence")
    assert "(estimate) · stale" in tip


def _generate(qt_app, clip, durations):
    """Stores segments that match the clip's current pieces, so it's clean."""
    from kokoro_gui.daw.dirty import predict_segment_texts, spoken_text

    config = qt_app._assemble_generation_config(clip)
    text = qt_app.document.clip_text(clip)
    key = compute_expected_cache_hash(text, config, key_fn=qt_app.document.segment_key_fn, clip=clip)
    results = []
    for i, (piece, seconds) in enumerate(zip(predict_segment_texts(spoken_text(text, config), config), durations)):
        path = f"seg{i}.wav"
        with open(path, "wb") as f:
            f.write(b"RIFF")
        results.append({"text": piece, "path": path, "duration": seconds, "cache_key": key})
    clip.segments = build_segments_from_results(key, results)
    clip.status = "generated"
    qt_app.editor.rehighlight()


def test_segment_tooltip_on_a_generated_clip_gives_its_length_and_take(qt_app):
    _target_words(qt_app, 10)
    clip = _one_clip(qt_app)
    _generate(qt_app, clip, [3.2, 4.5, 2.0])
    assert clip not in qt_app.document.dirty_clips()
    _details(qt_app)
    tip = qt_app.editor.details_tooltip_at(SENTENCES.index("w12"))
    assert tip == "Segment 2 of 3 · 10 words · ends at a sentence\n4.5 s · take 1"
    last = qt_app.editor.details_tooltip_at(SENTENCES.index("w25"))
    assert "ends at the end of the clip" in last


def test_no_details_tooltip_outside_a_clip(qt_app):
    _type(qt_app.editor, "untagged text")
    _details(qt_app)
    assert qt_app.editor.details_tooltip_at(3) is None


# -- gap labels ------------------------------------------------------------------------------


def test_gap_labels_sit_at_each_clip_after_the_first(qt_app):
    text = "One.\nTwo.\n\nThree."
    _type(qt_app.editor, text)
    alice = qt_app.document.characters[0]
    for start, end in ((0, 4), (5, 9), (11, 17)):
        qt_app.document.assign_character_to_range(start, end, alice.id)
    qt_app.document.settings["gap_s"] = 0.35
    qt_app.document.settings["paragraph_gap_s"] = 0.9
    _details(qt_app)
    assert qt_app.editor.gap_labels() == [(5, "gap 0.35 s"), (11, "gap 0.90 s ¶")]
    qt_app.details_layer_actions["details_gaps"].setChecked(False)
    assert qt_app.editor.gap_labels() == []


# -- gutter ------------------------------------------------------------------------------------


def _repaint_gutter(qt_app):
    gutter = qt_app.editor._gutter
    gutter.resize(gutter.sizeHint().width(), 400)
    gutter.paintEvent(QPaintEvent(gutter.rect()))
    return gutter


def test_gutter_shows_clip_info_only_with_details_on(qt_app):
    _one_clip(qt_app, "One line of text.")
    assert _repaint_gutter(qt_app).info_rects() == []
    _details(qt_app)
    rects = _repaint_gutter(qt_app).info_rects()
    clip_id = qt_app.document.clips[0].id
    assert [cid for _rect, cid in rects] == [clip_id, clip_id]  # the dot and the length


def test_gutter_info_is_one_per_clip(qt_app):
    text = "One.\nTwo.\nThree."
    _type(qt_app.editor, text)
    alice = qt_app.document.characters[0]
    qt_app.document.assign_character_to_range(0, 9, alice.id)  # spans two lines
    qt_app.document.assign_character_to_range(10, 16, alice.id)
    qt_app.editor.rehighlight()
    _details(qt_app)
    ids = {cid for _rect, cid in _repaint_gutter(qt_app).info_rects()}
    assert ids == {c.id for c in qt_app.document.clips}
    assert len(_repaint_gutter(qt_app).info_rects()) == 4


def test_clip_length_text_estimates_a_stale_clip(qt_app):
    clip = _one_clip(qt_app)
    assert qt_app.clip_length_text(clip, stale=True).startswith("~")


# -- caret strip -----------------------------------------------------------------------------


def test_caret_strip_is_hidden_with_details_off(qt_app):
    _one_clip(qt_app)
    assert qt_app.transcript_dock.info_strip.isHidden()


def test_caret_strip_follows_the_caret(qt_app):
    _target_words(qt_app, 10)
    clip = _one_clip(qt_app)
    _details(qt_app)
    strip = qt_app.transcript_dock.info_strip
    assert not strip.isHidden()
    _place_caret(qt_app.editor, SENTENCES.index("w12"))
    name = qt_app.document.get_character(clip.character_id).name
    assert strip.text().startswith(f"{name} · ")
    assert "segment 2/3" in strip.text()
    assert "stale" in strip.text() and "to do" in strip.text()
    _place_caret(qt_app.editor, SENTENCES.index("w1 "))
    assert "segment 1/3" in strip.text()


def test_caret_strip_updates_after_a_generate(qt_app):
    _target_words(qt_app, 10)
    clip = _one_clip(qt_app)
    _details(qt_app)
    _place_caret(qt_app.editor, 1)
    _generate(qt_app, clip, [3.2, 4.5, 2.0])
    text = qt_app.transcript_dock.info_strip.text()
    assert "take 1" in text and "generated" in text and "stale" not in text


def test_caret_strip_hides_with_clip_info_off(qt_app):
    _one_clip(qt_app)
    _details(qt_app)
    qt_app.details_layer_actions["details_clip_info"].setChecked(False)
    assert qt_app.transcript_dock.info_strip.isHidden()
