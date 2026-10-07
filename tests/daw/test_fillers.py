"""Tests for kokoro_gui/daw/fillers.py: which words of an imported recording
count as fillers, the range removing each one takes, and that generated clips
and clip boundaries are never touched (plan 32). No Qt."""
from kokoro_gui.daw import fillers
from kokoro_gui.daw.fillers import CONTEXT, SAFE, find_fillers, timed_span
from kokoro_gui.daw.imported import run_from_asr_words
from kokoro_gui.daw.models import Clip, Document, Run

A = "aaaaaaaaaaaaaaaa"
SOURCES = {A: {"path": "/p/audio/imported/a.wav", "sample_rate": 24000, "duration_s": 60.0}}


def _recording(sentence, start_s=0.0):
    """A recording clip for `sentence`, one word per token, 0.3 s each."""
    tokens = sentence.split()
    rows = [(token, start_s + 0.3 * i, start_s + 0.3 * i + 0.3) for i, token in enumerate(tokens)]
    text, words = run_from_asr_words(rows, A)
    clip = Clip(source="imported")
    return clip, Run(text=text, clip_id=clip.id, kind="imported", words=words)


def _doc(*sentences):
    clips, runs = [], []
    for i, sentence in enumerate(sentences):
        clip, run = _recording(sentence, start_s=i * 30.0)
        if runs:
            runs.append(Run(text="\n\n"))
        clips.append(clip)
        runs.append(run)
    return Document(runs=runs, clips=clips, settings={"sources": dict(SOURCES)})


def _cut(document, hits):
    """The text after removing `hits`, last first."""
    text = document.text
    for hit in sorted(hits, key=lambda h: h.start, reverse=True):
        text = text[:hit.start] + text[hit.end:]
    return text


def test_um_is_found_with_the_space_after_it():
    doc = _doc("Well um we went home")
    hit, = find_fillers(doc)
    assert (hit.kind, hit.text) == (SAFE, "um")
    assert doc.text[hit.start:hit.end] == "um "
    assert _cut(doc, [hit]) == "Well we went home"


def test_uh_comma_takes_the_comma_and_the_space():
    doc = _doc("Well, uh, we went home")
    hit, = find_fillers(doc)
    assert doc.text[hit.start:hit.end] == "uh, "
    assert _cut(doc, [hit]) == "Well, we went home"


def test_a_filler_at_the_start_of_a_clip_and_at_its_end():
    doc = _doc("Um we went home uh")
    first, last = find_fillers(doc)
    assert doc.text[first.start:first.end] == "Um "
    assert doc.text[last.start:last.end] == " uh"
    assert _cut(doc, [first, last]) == "we went home"


def test_a_filler_before_a_full_stop_takes_the_comma_before_it():
    doc = _doc("We went to the store, um.")
    hit, = find_fillers(doc)
    assert _cut(doc, [hit]) == "We went to the store."


def test_stretched_spellings_are_found():
    doc = _doc("umm uhh erm hmm ahh er")
    assert [h.text for h in find_fillers(doc)] == ["umm", "uhh", "erm", "hmm", "ahh", "er"]


def test_real_words_and_compounds_are_left_alone():
    doc = _doc("To err is human, the umbrella, uh-huh, ah-ha, a humming hum")
    assert find_fillers(doc) == []


def test_two_in_a_row_do_not_overlap():
    doc = _doc("so um um we went")
    hits = find_fillers(doc)
    assert [h.text for h in hits] == ["um", "um"]
    assert hits[0].end <= hits[1].start
    assert _cut(doc, hits) == "so we went"


def test_i_like_it_is_not_a_filler():
    doc = _doc("I like it and you know what I mean and it is kind of nice")
    assert find_fillers(doc) == []


def test_like_set_off_by_commas_is_a_context_filler():
    doc = _doc("So, like, we went home")
    hit, = find_fillers(doc)
    assert (hit.kind, hit.text) == (CONTEXT, "like")
    assert doc.text[hit.start:hit.end] == "like, "
    assert _cut(doc, [hit]) == "So, we went home"


def test_a_phrase_at_a_sentence_start_followed_by_a_comma_matches():
    doc = _doc("We left. You know, it rained. I mean, a lot. Like, a lot")
    hits = find_fillers(doc)
    assert [h.text for h in hits] == ["You know", "I mean", "Like"]
    assert {h.kind for h in hits} == {CONTEXT}
    assert _cut(doc, hits) == "We left. it rained. a lot. a lot"


def test_a_phrase_with_only_a_comma_after_it_mid_sentence_does_not_match():
    # "went like, home" has no comma before it and is no sentence start.
    doc = _doc("We went like, home")
    assert find_fillers(doc) == []


def test_sort_of_and_kind_of_between_commas():
    doc = _doc("It was, sort of, nice, kind of, odd")
    assert [h.text for h in find_fillers(doc)] == ["sort of", "kind of"]


def test_generated_clips_are_ignored():
    clip = Clip()
    doc = Document(runs=[Run(text="Well um we went", clip_id=clip.id)], clips=[clip])
    assert find_fillers(doc) == []


def test_text_typed_into_a_recording_without_timing_is_ignored():
    clip, run = _recording("Well we went home")
    run.text = run.text + " um"  # no word covers it
    doc = Document(runs=[run], clips=[clip], settings={"sources": dict(SOURCES)})
    assert find_fillers(doc) == []


def test_a_hit_never_spans_two_clips():
    doc = _doc("we went um", "um we came back")
    first, second = find_fillers(doc)
    extent_a, extent_b = (doc.clip_extent(c.id) for c in doc.clips)
    assert first.end <= extent_a[1] and second.start >= extent_b[0]
    assert doc.text[first.start:first.end] == " um"
    assert doc.text[second.start:second.end] == "um "
    assert first.clip_id == doc.clips[0].id and second.clip_id == doc.clips[1].id


def test_hits_come_back_in_text_order():
    doc = _doc("one um two", "three uh four")
    hits = find_fillers(doc)
    assert [h.start for h in hits] == sorted(h.start for h in hits)
    assert fillers.find_fillers(doc) == hits


def test_timed_span_is_the_word_in_the_recording_file():
    doc = _doc("Well um we went")
    hit, = find_fillers(doc)
    source, start_s, end_s = timed_span(doc, hit)
    assert (source, round(start_s, 2), round(end_s, 2)) == (A, 0.3, 0.6)
