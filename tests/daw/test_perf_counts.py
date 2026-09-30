"""Operation counts on the document's derived state (Claude/PLAN_performance.md).

Counts, not timings, so they hold on any machine: an edit inside one clip
reruns one dirty check, and lookups read the index instead of walking the
runs. `tests/gui_qt/test_perf_counts.py` holds the widget half.
"""
import pytest

from kokoro_gui.daw import derived, dirty
from kokoro_gui.daw.models import Character, Document

LINE = "A line of narration for one clip."
CLIPS = 20


@pytest.fixture
def document():
    alice = Character.from_preset_dict("Alice", {"voice": "af_heart"})
    doc = Document.from_plain_text("\n".join(LINE for _ in range(CLIPS)), characters=[alice])
    for i in range(CLIPS):
        start = i * (len(LINE) + 1)
        doc.assign_character_to_range(start, start + len(LINE), alice.id)
    return doc


@pytest.fixture
def counted_dirty_checks(monkeypatch):
    calls = []
    real = dirty.is_clip_dirty

    def counting(clip, *args, **kwargs):
        calls.append(clip.id)
        return real(clip, *args, **kwargs)

    monkeypatch.setattr(dirty, "is_clip_dirty", counting)
    return calls


def test_an_edit_inside_one_clip_reruns_one_dirty_check(document, counted_dirty_checks):
    assert len(document.dirty_ids()) == CLIPS
    assert len(counted_dirty_checks) == CLIPS
    counted_dirty_checks.clear()

    position = 5 * (len(LINE) + 1) + 3
    text = document.text
    document.replace_text(position, 0, 1, text[:position] + "x" + text[position:])
    document.dirty_ids()

    assert counted_dirty_checks == [document.clips[5].id]


def test_asking_again_without_a_change_reruns_nothing(document, counted_dirty_checks, monkeypatch):
    from kokoro_gui.daw import revision

    monkeypatch.setattr(revision, "VERIFY", False)  # verify mode reruns every check on purpose
    document.dirty_ids()
    counted_dirty_checks.clear()
    for _ in range(5):
        document.dirty_ids()
    assert counted_dirty_checks == []


def test_lookups_build_the_index_once(document, monkeypatch):
    from kokoro_gui.daw import revision

    monkeypatch.setattr(revision, "VERIFY", False)
    builds = []
    real_init = derived.DocumentIndex.__init__

    def counting_init(self, *args, **kwargs):
        builds.append(1)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(derived.DocumentIndex, "__init__", counting_init)
    document.touch()
    for position in range(len(document.text)):
        document.clip_covering(position)
    for clip in document.clips:
        document.clip_extent(clip.id)
        document.clip_text(clip)
    assert builds == [1]
