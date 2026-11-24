"""kokoro_gui/core/generation.py: which clips a batch generates, grouped by
engine, and landing the outcomes."""
from kokoro_gui.core import generation as core_generation
from kokoro_gui.daw.models import Clip, Document
from kokoro_gui.engines.missing import MissingBackend


class FakeBackend:
    def __init__(self, engine_id):
        self.id = engine_id


KOKORO = FakeBackend("kokoro")
AUDIO8 = FakeBackend("audio8")


def _document(*clips):
    return Document(runs=[], clips=list(clips), characters=[])


def test_groups_by_engine_in_order():
    a, b, c = Clip(), Clip(), Clip()
    document = _document(a, b, c)
    engine_of = {a.id: KOKORO, b.id: AUDIO8, c.id: KOKORO}
    batch = core_generation.plan_batch(document, [a, b, c], lambda clip: engine_of[clip.id],
                                       lambda clip: {"for": clip.id})
    assert batch.clips == [a, b, c]
    assert list(batch.groups) == ["kokoro", "audio8"]
    backend, items = batch.groups["kokoro"]
    assert backend is KOKORO
    assert items == [(a.id, document.clip_text(a), {"for": a.id}), (c.id, document.clip_text(c), {"for": c.id})]
    assert [item[0] for item in batch.groups["audio8"][1]] == [b.id]
    assert batch.blocked == {}


def test_blocked_clips_are_counted_by_reason():
    missing = MissingBackend("audio8")
    a, b, c = Clip(), Clip(), Clip()
    engine_of = {a.id: missing, b.id: KOKORO, c.id: missing}
    configs = []
    batch = core_generation.plan_batch(_document(a, b, c), [a, b, c], lambda clip: engine_of[clip.id],
                                       lambda clip: configs.append(clip.id) or {})
    assert batch.clips == [b]
    assert batch.blocked == {missing.message: 2}
    assert configs == [b.id]


def test_nested_and_imported_clips_are_dropped_uncounted():
    nested, imported, plain = Clip(source="nested"), Clip(source="imported"), Clip()
    asked = []

    def backend_for(clip):
        asked.append(clip.id)
        return KOKORO

    kept, blocked = core_generation.generatable([nested, imported, plain], backend_for)
    assert kept == [plain]
    assert blocked == {}
    assert asked == [plain.id]


def test_apply_outcomes_lands_successes_and_lists_failures():
    a, b = Clip(), Clip()
    document = _document(a, b)
    applied = []
    outcomes = [
        {"clip_id": a.id, "success": True, "results": ["ra"]},
        {"clip_id": b.id, "success": False, "results": [], "error": "boom"},
        {"clip_id": "deleted", "success": True, "results": ["rx"]},
    ]
    succeeded, failed = core_generation.apply_outcomes(document, outcomes,
                                                       lambda clip, results: applied.append((clip, results)))
    assert (succeeded, failed) == ([a.id], [b.id])
    assert applied == [(a, ["ra"])]
