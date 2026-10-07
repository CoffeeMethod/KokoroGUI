"""Tests for kokoro_gui/daw/genqueue.py (plan 28): batching, ordering, the time
left, and reading a saved queue back."""
import json

import pytest

from kokoro_gui.daw import genqueue
from kokoro_gui.daw.genqueue import GenerationQueue, QueueItem, plan_items


def _item(*clip_ids, kind=genqueue.CLIPS, project="p1", state=genqueue.QUEUED, chars=0, engines=None, title="t"):
    return QueueItem(kind=kind, project_id=project, clip_ids=list(clip_ids), title=title, state=state, chars=chars,
                     engine_chars=dict(engines or {}))


def _queue(*names):
    queue = GenerationQueue()
    for name in names:
        queue.add(_item(name, title=name))
    return queue


def _titles(queue):
    return [i.title for i in queue.items]


# -- planning ---------------------------------------------------------------------


def test_twenty_clips_split_into_batches_of_eight_in_order():
    entries = [(f"c{i}", 10, "kokoro") for i in range(20)]

    items = plan_items("p1", entries, "Book")

    assert [i.clip_count for i in items] == [8, 8, 4]
    assert [c for i in items for c in i.clip_ids] == [f"c{i}" for i in range(20)]
    assert [i.title for i in items] == ["Book, clips 1-8 of 20", "Book, clips 9-16 of 20", "Book, clips 17-20 of 20"]
    assert all(i.kind == genqueue.CLIPS and i.state == genqueue.QUEUED and i.project_id == "p1" for i in items)


def test_a_short_list_keeps_the_plain_title_and_splits_characters_by_engine():
    items = plan_items("p1", [("a", 5, "kokoro"), ("b", 7, "audio8"), ("c", 3, "kokoro")], "Intro")

    assert len(items) == 1
    assert items[0].title == "Intro"
    assert items[0].chars == 15
    assert items[0].engine_chars == {"kokoro": 8, "audio8": 7}


def test_exactly_one_batch_is_not_split_and_nothing_gives_no_items():
    assert len(plan_items("p", [(f"c{i}", 1, "k") for i in range(genqueue.BATCH_CLIPS)], "t")) == 1
    assert plan_items("p", [], "t") == []
    assert [i.clip_count for i in plan_items("p", [(f"c{i}", 1, "k") for i in range(5)], "t", batch=2)] == [2, 2, 1]
    assert [i.clip_count for i in plan_items("p", [("a", 1, "k"), ("b", 1, "k")], "t", batch=0)] == [1, 1]


# -- add, move, next ----------------------------------------------------------------


def test_next_queued_skips_running_and_finished_items():
    queue = _queue("a", "b", "c")
    queue.mark(queue.items[0], genqueue.DONE)
    queue.mark(queue.items[1], genqueue.RUNNING)

    assert queue.next_queued().title == "c"
    assert queue.running().title == "b"
    queue.mark(queue.items[2], genqueue.RUNNING)
    assert queue.next_queued() is None


def test_add_at_an_index_and_find():
    queue = _queue("a", "b")
    added = queue.add(_item("z", title="z"), index=0)

    assert _titles(queue) == ["z", "a", "b"]
    assert queue.find(added.id) is added
    assert queue.find("nope") is None
    assert queue.index_of(added) == 0


def test_move_reorders_queued_items():
    queue = _queue("a", "b", "c", "d")

    assert queue.move(3, 1) is True
    assert _titles(queue) == ["a", "d", "b", "c"]
    assert queue.move(0, 2) is True
    assert _titles(queue) == ["d", "b", "a", "c"]
    assert queue.move(1, 1) is False


def test_move_clamps_and_rejects_a_bad_index():
    queue = _queue("a", "b", "c")

    assert queue.move(0, 99) is True
    assert _titles(queue) == ["b", "c", "a"]
    assert queue.move(-1, 0) is False
    assert queue.move(7, 0) is False


def test_a_queued_item_never_moves_above_a_running_or_finished_one():
    queue = _queue("a", "b", "c", "d")
    queue.mark(queue.items[0], genqueue.DONE)
    queue.mark(queue.items[1], genqueue.RUNNING)

    assert queue.move(3, 0) is True
    assert _titles(queue) == ["a", "b", "d", "c"]
    # A running or finished item can't be dragged at all.
    assert queue.move(1, 3) is False
    assert queue.move(0, 3) is False


def test_move_to_top_goes_ahead_of_the_other_queued_items_only():
    queue = _queue("a", "b", "c")
    queue.mark(queue.items[0], genqueue.RUNNING)

    assert queue.move_to_top(queue.items[2]) is True
    assert _titles(queue) == ["a", "c", "b"]
    assert queue.move_to_top(queue.items[0]) is False  # running


def test_remove_only_takes_out_a_queued_item():
    queue = _queue("a", "b")
    queue.mark(queue.items[0], genqueue.RUNNING)

    assert queue.remove(queue.items[0]) is False
    assert queue.remove(queue.items[1]) is True
    assert _titles(queue) == ["a"]
    assert queue.remove(_item("x")) is False


def test_cancel_queued_leaves_the_running_item_and_drop_finished_clears_history():
    queue = _queue("a", "b", "c")
    queue.mark(queue.items[0], genqueue.RUNNING)

    assert queue.cancel_queued() == 2
    assert [i.state for i in queue.items] == [genqueue.RUNNING, genqueue.CANCELLED, genqueue.CANCELLED]
    queue.mark(queue.items[0], genqueue.DONE)
    queue.drop_finished()
    assert queue.items == []


def test_mark_refuses_an_unknown_state():
    queue = _queue("a")
    with pytest.raises(ValueError):
        queue.mark(queue.items[0], "paused")


def test_counts_and_queued_clip_ids():
    queue = GenerationQueue([_item("a", "b", chars=10), _item("c", chars=5, project="p2"),
                             _item("n1", kind=genqueue.SUBPROJECT), _item("d", state=genqueue.DONE),
                             _item("e", state=genqueue.CANCELLED)])
    queue.mark(queue.items[0], genqueue.RUNNING)

    assert queue.clip_total() == 4  # a, b, c, d: the cancelled item no longer counts
    assert queue.clips_finished() == 1
    assert queue.pending_clip_count() == 3
    assert queue.remaining_chars() == 15
    assert queue.queued_clip_ids() == {"c", "n1"}
    assert queue.queued_clip_ids("p2") == {"c"}


# -- time left -------------------------------------------------------------------------


RATES = {"kokoro": 10.0, "audio8": 2.0}


def test_eta_sums_items_and_takes_the_slowest_engine_inside_one():
    queue = GenerationQueue([
        _item("a", chars=100, engines={"kokoro": 100}),
        _item("b", chars=60, engines={"kokoro": 50, "audio8": 10}),
    ])

    seconds, complete = queue.eta_s(RATES.get)

    assert complete is True
    assert seconds == pytest.approx(10.0 + 5.0)  # item b: max(50/10, 10/2)


def test_eta_counts_only_what_is_left_of_the_running_item():
    queue = GenerationQueue([_item("a", chars=100, engines={"kokoro": 100}),
                             _item("b", chars=100, engines={"kokoro": 100})])
    queue.mark(queue.items[0], genqueue.RUNNING)

    seconds, _complete = queue.eta_s(RATES.get, running_fraction=0.75)

    assert seconds == pytest.approx(2.5 + 10.0)


def test_eta_skips_finished_items_and_is_a_floor_when_a_rate_or_a_size_is_unknown():
    queue = GenerationQueue([_item("a", chars=100, engines={"kokoro": 100}, state=genqueue.DONE),
                             _item("b", chars=100, engines={"kokoro": 100}),
                             _item("c", chars=50, engines={"mystery": 50}),
                             _item("n", kind=genqueue.SUBPROJECT)])

    seconds, complete = queue.eta_s(RATES.get)

    assert seconds == pytest.approx(10.0)
    assert complete is False


def test_eta_of_an_empty_queue_is_zero_and_complete():
    assert GenerationQueue().eta_s(RATES.get) == (0.0, True)


@pytest.mark.parametrize("seconds, text", [
    (0, "under a minute"), (30, "1 min"), (60, "1 min"), (61, "2 min"), (59 * 60, "59 min"),
    (60 * 60, "1 h"), (2 * 3600 + 10 * 60, "2 h 10 min"), (-5, "under a minute"),
])
def test_format_eta(seconds, text):
    assert genqueue.format_eta(seconds) == text


# -- saving ------------------------------------------------------------------------------


def test_a_queue_round_trips_through_json():
    queue = GenerationQueue([_item("a", "b", chars=12, engines={"kokoro": 12}, title="Chapter 1"),
                             _item("n1", kind=genqueue.SUBPROJECT, title="Chapter 2", state=genqueue.DONE)])

    again = GenerationQueue.from_dict(json.loads(json.dumps(queue.to_dict())))

    assert [i.to_dict() for i in again.items] == [i.to_dict() for i in queue.items]


def test_a_running_item_comes_back_queued():
    queue = GenerationQueue([_item("a")])
    queue.mark(queue.items[0], genqueue.RUNNING)

    assert GenerationQueue.from_dict(queue.to_dict()).items[0].state == genqueue.QUEUED


def test_an_unknown_state_becomes_queued():
    raw = _item("a").to_dict()
    raw["state"] = "paused"

    assert genqueue.item_from_dict(raw).state == genqueue.QUEUED


@pytest.mark.parametrize("junk", [None, 5, "x", [], {}, {"items": "no"}, {"items": [None, 3, "a", [], {}]}])
def test_junk_that_is_not_a_queue_gives_an_empty_one(junk):
    assert GenerationQueue.from_dict(junk).items == []


def test_items_that_cannot_be_items_are_dropped_and_the_rest_kept():
    good = _item("a").to_dict()
    bad_kind = dict(good, kind="render")
    no_project = dict(good, project_id="")
    no_clips = dict(good, clip_ids=[])
    clips_not_strings = dict(good, clip_ids=[1, None, ["x"]])
    two_nested = dict(good, kind=genqueue.SUBPROJECT, clip_ids=["a", "b"])
    same_id = dict(good, clip_ids=["z"])

    queue = GenerationQueue.from_dict({"items": [bad_kind, no_project, no_clips, clips_not_strings, two_nested,
                                                 good, same_id]})

    assert [i.clip_ids for i in queue.items] == [["a"]]


def test_fields_of_the_wrong_type_are_cleaned_not_trusted():
    raw = _item("a", "b").to_dict()
    raw.update({"title": 7, "chars": "many", "engine_chars": {"kokoro": float("inf"), 3: 5, "audio8": -9},
                "id": ["x"], "clip_ids": ["a", 4, "", "b" * 500, "c"]})

    item = genqueue.item_from_dict(raw)

    assert item.title == ""
    assert item.chars == 0
    assert item.engine_chars == {"kokoro": 0, "audio8": 0}
    assert item.clip_ids == ["a", "c"]
    assert isinstance(item.id, str) and item.id


def test_a_huge_queue_is_cut_to_the_limits():
    items = [_item(f"c{i}").to_dict() for i in range(genqueue.MAX_ITEMS + 50)]
    for n, raw in enumerate(items):
        raw["id"] = f"id{n}"
    big = {"kind": genqueue.CLIPS, "project_id": "p", "clip_ids": [f"c{i}" for i in range(genqueue.MAX_ITEM_CLIPS + 10)]}

    assert len(GenerationQueue.from_dict({"items": items}).items) == genqueue.MAX_ITEMS
    assert len(genqueue.item_from_dict(big).clip_ids) == genqueue.MAX_ITEM_CLIPS
