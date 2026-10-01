"""Tests for kokoro_gui/daw/segment_view.py: where a clip's segments and
lexicon rewrites sit in the transcript (Options > Transcript details)."""
from kokoro_gui.daw import segment_view
from kokoro_gui.daw.dirty import predict_segment_texts, spoken_text
from kokoro_gui.daw.models import Character, Document, Track
from kokoro_gui.engine.segmenting import PARAGRAPH, SENTENCE, WORD


def _doc(text):
    alice = Character.from_preset_dict("Alice", {"voice": "af_bella"})
    doc = Document.from_plain_text(text, characters=[alice], tracks=[Track(name="Alice", character_id=alice.id)])
    return doc, alice


def _sentences(n, words=5, start=1):
    out, i = [], start
    for _ in range(n):
        out.append(" ".join(f"w{j}" for j in range(i, i + words)) + ".")
        i += words
    return " ".join(out)


def test_pieces_are_the_segments_generate_would_make():
    body = _sentences(6)
    doc, alice = _doc("Intro line.\n" + body)
    clip = doc.assign_character_to_range(12, 12 + len(body), alice.id)
    config = {"segment_target_words": 10}
    pieces = segment_view.clip_pieces(doc, clip, config)
    assert [" ".join(doc.text[p.start:p.end].split()) for p in pieces] == predict_segment_texts(body, config)
    assert [p.index for p in pieces] == [0, 1, 2]
    assert [p.level for p in pieces] == [SENTENCE, SENTENCE, PARAGRAPH]
    assert [p.words for p in pieces] == [10, 10, 10]
    assert pieces[0].start == 12


def test_a_forced_cut_reports_a_word_break():
    body = " ".join(f"w{i}" for i in range(1, 51))
    doc, alice = _doc(body)
    clip = doc.assign_character_to_range(0, len(body), alice.id)
    pieces = segment_view.clip_pieces(doc, clip, {"segment_target_words": 10})
    assert [p.level for p in pieces] == [WORD, WORD, WORD, PARAGRAPH]


def test_offsets_map_back_through_a_length_changing_lexicon_rule():
    body = "Dr. Smith w2 w3 w4 w5 w6 w7 w8 w9. Then Dr. Jones w12 w13 w14 w15 w16 w17 w18 w19."
    doc, alice = _doc(body)
    clip = doc.assign_character_to_range(0, len(body), alice.id)
    config = {"segment_target_words": 10, "lexicon": {"Dr.": "Doctor"}}
    pieces = segment_view.clip_pieces(doc, clip, config)
    assert len(pieces) == len(predict_segment_texts(spoken_text(body, config), config)) == 2
    assert doc.text[pieces[0].start:pieces[0].end] == "Dr. Smith w2 w3 w4 w5 w6 w7 w8 w9."
    assert doc.text[pieces[1].start:pieces[1].end] == "Then Dr. Jones w12 w13 w14 w15 w16 w17 w18 w19."


def test_lexicon_rewrites_name_what_the_text_becomes():
    body = "See Dr. Smith and the NHS."
    doc, alice = _doc("x " + body)
    clip = doc.assign_character_to_range(2, 2 + len(body), alice.id)
    rewrites = segment_view.lexicon_rewrites(doc, clip, {"lexicon": {"Dr.": "Doctor", "NHS": "NHs"}})
    assert [(doc.text[s:e], spoken) for s, e, spoken in rewrites] == [("Dr.", "Doctor"), ("NHS", "NHs")]


def test_no_lexicon_means_no_rewrites():
    doc, alice = _doc("Plain text.")
    clip = doc.assign_character_to_range(0, 11, alice.id)
    assert segment_view.lexicon_rewrites(doc, clip, {}) == []


def test_default_config_is_the_documents_per_clip_config():
    body = _sentences(4)
    doc, alice = _doc(body)
    clip = doc.assign_character_to_range(0, len(body), alice.id)
    doc.generation_config_fn = lambda c: {"segment_target_words": 5}
    assert len(segment_view.clip_pieces(doc, clip)) == 4


def test_clips_not_generated_from_text_have_no_pieces():
    doc, alice = _doc("Some words here.")
    clip = doc.assign_character_to_range(0, 16, alice.id)
    clip.source = "imported"
    assert segment_view.clip_pieces(doc, clip, {}) == []
    assert segment_view.lexicon_rewrites(doc, clip, {"lexicon": {"words": "werds"}}) == []
    assert segment_view.clip_pieces(doc, None, {}) == []


def test_memo_reuses_and_recomputes_on_a_segmentation_change():
    text = _sentences(4)
    first = segment_view._analyse(text, {"segment_target_words": 10})
    assert segment_view._analyse(text, {"segment_target_words": 10}) is first
    other = segment_view._analyse(text, {"segment_target_words": 5})
    assert other is not first and len(other[0]) == 4


def test_piece_at():
    pieces = [segment_view.PieceSpan(0, 0, 5, SENTENCE, 1), segment_view.PieceSpan(1, 6, 9, PARAGRAPH, 1)]
    assert segment_view.piece_at(pieces, 0).index == 0
    assert segment_view.piece_at(pieces, 5) is None
    assert segment_view.piece_at(pieces, 8).index == 1


def test_gap_before_names_where_the_silence_comes_from():
    doc, alice = _doc("One. Two.\n\nThree. Four.")
    first = doc.assign_character_to_range(0, 4, alice.id)
    second = doc.assign_character_to_range(5, 9, alice.id)
    third = doc.assign_character_to_range(11, 17, alice.id)
    fourth = doc.assign_character_to_range(18, 23, alice.id)
    doc.settings["gap_s"] = 0.35
    doc.settings["paragraph_gap_s"] = 0.9
    assert segment_view.gap_before(doc, first) == ("first", 0.0)
    assert segment_view.gap_before(doc, second) == ("clip", 0.35)
    assert segment_view.gap_before(doc, third) == ("paragraph", 0.9)
    fourth.gap_before_s = 1.5
    assert segment_view.gap_before(doc, fourth) == ("override", 1.5)
    fourth.timeline_timestamp = 62.3
    assert segment_view.gap_before(doc, fourth) == ("time", 62.3)


def test_estimated_length_uses_the_learned_rate():
    doc, alice = _doc("x" * 30)
    clip = doc.assign_character_to_range(0, 30, alice.id)
    assert segment_view.estimated_length_s(doc, clip, {alice.id: 10.0}) == 3.0
    assert segment_view.estimated_length_s(doc, clip, {None: 15.0}) == 2.0


def test_format_length():
    assert segment_view.format_length(11.44) == "11.4 s"
    assert segment_view.format_length(11.44, estimate=True) == "~11 s"
    assert segment_view.format_length(62.3) == "1:02"
    assert segment_view.format_length(3723, estimate=True) == "~1:02:03"
    assert segment_view.format_length(None) == "0.0 s"


def test_pieces_start_after_a_tag_inside_the_clip():
    text = "[Alice:Radio]: " + _sentences(2)
    doc, alice = _doc(text)
    clip = doc.assign_character_to_range(0, len(text), alice.id)
    config = {"segment_target_words": 5}
    pieces = segment_view.clip_pieces(doc, clip, config)
    assert [" ".join(doc.text[p.start:p.end].split()) for p in pieces] == ["w1 w2 w3 w4 w5.", "w6 w7 w8 w9 w10."]
    assert pieces[0].start == len("[Alice:Radio]: ")
    assert segment_view.lexicon_rewrites(doc, clip, config) == []
