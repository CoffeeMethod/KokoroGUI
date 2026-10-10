"""The generation queue (plan 28): what Generate hands the app to run.

Generate no longer starts one batch for every stale clip. It plans the stale
clips into items of at most `BATCH_CLIPS` clips each (`plan_items`), plus one
item per stale subproject, and the app runs them one after another. Each item
lands on the timeline when it finishes, so a book fills in clip by clip, and
a queue can pause, reorder and resume.

An item names clips by id and holds no clip object, so a queue written to
`session.json["queue"]` outlives the window. `from_dict` type-checks every
field, since that file is untrusted input like the rest of `session.json`.
Queue state is runtime state: it is never in `document.json` or a bundle.

Qt-free and model-free. `qt/gen_queue.py` runs the queue.
"""
from __future__ import annotations

import math
import secrets
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

CLIPS = "clips"
SUBPROJECT = "subproject"
KINDS = (CLIPS, SUBPROJECT)

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"
STATES = (QUEUED, RUNNING, DONE, FAILED, CANCELLED)
FINISHED = (DONE, FAILED, CANCELLED)

# A "clips" item holds at most this many clips, so results land every few clips.
BATCH_CLIPS = 8

# Limits for a queue read back from `session.json`.
MAX_ITEMS = 5_000
MAX_ITEM_CLIPS = 1_000
MAX_ID_CHARS = 200
MAX_TITLE_CHARS = 200
MAX_ENGINES = 20
MAX_CHARS = 100_000_000


def new_item_id() -> str:
    return secrets.token_hex(6)


@dataclass
class QueueItem:
    """One unit of queued work. `clip_ids` are the clips of `project_id`'s
    document a "clips" item generates, or the one nested clip a "subproject"
    item opens, generates and renders. `chars` is the text length to speak
    and `engine_chars` the same split by engine id, for the time left; a
    subproject item starts at 0 and learns them when its child opens."""

    kind: str
    project_id: str
    clip_ids: list
    title: str = ""
    state: str = QUEUED
    chars: int = 0
    engine_chars: dict = field(default_factory=dict)
    id: str = field(default_factory=new_item_id)

    @property
    def clip_count(self) -> int:
        return len(self.clip_ids)

    @property
    def finished(self) -> bool:
        return self.state in FINISHED

    def to_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "project_id": self.project_id, "clip_ids": list(self.clip_ids),
                "title": self.title, "state": self.state, "chars": self.chars,
                "engine_chars": dict(self.engine_chars)}


def _clean_text(value, limit: int) -> Optional[str]:
    if not isinstance(value, str):
        return None
    return value[:limit]


def _clean_count(value) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    if isinstance(value, float) and not math.isfinite(value):
        return 0
    return max(0, min(MAX_CHARS, int(value)))


def item_from_dict(raw) -> Optional[QueueItem]:
    """The item `raw` describes, or None when it can't be one: not a dict, an
    unknown kind, or a missing project id or clip list. A state that isn't
    one of `STATES` becomes "queued", and so does "running", which a saved
    queue only holds after a crash."""
    if not isinstance(raw, dict):
        return None
    kind = raw.get("kind")
    project_id = _clean_text(raw.get("project_id"), MAX_ID_CHARS)
    clip_ids = raw.get("clip_ids")
    if kind not in KINDS or not project_id or not isinstance(clip_ids, list):
        return None
    ids = [c for c in clip_ids if isinstance(c, str) and 0 < len(c) <= MAX_ID_CHARS][:MAX_ITEM_CLIPS]
    if not ids or (kind == SUBPROJECT and len(ids) != 1):
        return None
    state = raw.get("state")
    if state not in STATES or state == RUNNING:
        state = QUEUED
    engines = raw.get("engine_chars")
    engine_chars = {}
    if isinstance(engines, dict):
        for engine_id, chars in list(engines.items())[:MAX_ENGINES]:
            if isinstance(engine_id, str) and 0 < len(engine_id) <= MAX_ID_CHARS:
                engine_chars[engine_id] = _clean_count(chars)
    item_id = raw.get("id")
    if not isinstance(item_id, str) or not 0 < len(item_id) <= MAX_ID_CHARS:
        item_id = new_item_id()
    return QueueItem(kind=kind, project_id=project_id, clip_ids=ids, title=_clean_text(raw.get("title"), MAX_TITLE_CHARS) or "",
                     state=state, chars=_clean_count(raw.get("chars")), engine_chars=engine_chars, id=item_id)


def format_eta(seconds: float) -> str:
    """"under a minute", "12 min", "2 h 10 min" or "3 h"."""
    minutes = int(math.ceil(max(0.0, seconds) / 60.0))
    if minutes < 1:
        return "under a minute"
    hours, minutes = divmod(minutes, 60)
    if not hours:
        return f"{minutes} min"
    return f"{hours} h {minutes} min" if minutes else f"{hours} h"


class GenerationQueue:
    """An ordered list of `QueueItem`s. The app runs the first "queued" one
    when nothing is running."""

    def __init__(self, items: Iterable = ()):
        self.items: list = list(items)

    # -- editing -----------------------------------------------------------------

    def add(self, item: QueueItem, index: Optional[int] = None) -> QueueItem:
        """Appends `item`, or puts it at `index`."""
        if index is None:
            self.items.append(item)
        else:
            self.items.insert(max(0, min(len(self.items), index)), item)
        return item

    def index_of(self, item: QueueItem) -> int:
        for i, candidate in enumerate(self.items):
            if candidate is item:
                return i
        return -1

    def find(self, item_id: str) -> Optional[QueueItem]:
        return next((i for i in self.items if i.id == item_id), None)

    def move(self, index: int, new_index: int) -> bool:
        """Moves the queued item at `index` to `new_index`. Only a queued item
        moves, and never above a running or finished one: the order of what
        already ran is history. Returns whether anything moved."""
        if not 0 <= index < len(self.items) or self.items[index].state != QUEUED:
            return False
        floor = self.first_queued_index()
        new_index = max(floor, min(len(self.items) - 1, new_index))
        if new_index == index:
            return False
        self.items.insert(new_index, self.items.pop(index))
        return True

    def move_to_top(self, item: QueueItem) -> bool:
        """Puts a queued `item` ahead of every other queued one."""
        index = self.index_of(item)
        return index >= 0 and self.move(index, self.first_queued_index())

    def remove(self, item: QueueItem) -> bool:
        """Takes a queued `item` out. A running or finished one stays."""
        index = self.index_of(item)
        if index < 0 or item.state != QUEUED:
            return False
        del self.items[index]
        return True

    def mark(self, item: QueueItem, state: str) -> None:
        if state not in STATES:
            raise ValueError(f"unknown state {state!r}")
        item.state = state

    def cancel_queued(self) -> int:
        """Every queued item becomes cancelled. Returns how many."""
        count = 0
        for item in self.items:
            if item.state == QUEUED:
                item.state = CANCELLED
                count += 1
        return count

    def drop_finished(self) -> None:
        self.items = [i for i in self.items if not i.finished]

    # -- reading -----------------------------------------------------------------

    def first_queued_index(self) -> int:
        """Where the queued items start: after anything running or finished."""
        for i, item in enumerate(self.items):
            if item.state == QUEUED:
                return i
        return len(self.items)

    def next_queued(self) -> Optional[QueueItem]:
        return next((i for i in self.items if i.state == QUEUED), None)

    def running(self) -> Optional[QueueItem]:
        return next((i for i in self.items if i.state == RUNNING), None)

    def pending(self) -> list:
        """Queued and running items."""
        return [i for i in self.items if i.state in (QUEUED, RUNNING)]

    def queued_clip_ids(self, project_id: Optional[str] = None) -> set:
        """The clip ids waiting in queued items (of `project_id`, when given)."""
        return {c for i in self.items if i.state == QUEUED and (project_id is None or i.project_id == project_id)
                for c in i.clip_ids}

    def pending_clip_ids(self, project_id: Optional[str] = None) -> set:
        """The clip ids in queued and running items (of `project_id`, when given)."""
        return {c for i in self.pending() if project_id is None or i.project_id == project_id for c in i.clip_ids}

    def clip_total(self) -> int:
        return sum(i.clip_count for i in self.items if i.kind == CLIPS and i.state != CANCELLED)

    def clips_finished(self) -> int:
        return sum(i.clip_count for i in self.items if i.kind == CLIPS and i.state in (DONE, FAILED))

    def pending_clip_count(self) -> int:
        return sum(i.clip_count for i in self.items if i.kind == CLIPS and i.state in (QUEUED, RUNNING))

    def remaining_chars(self) -> int:
        return sum(i.chars for i in self.pending())

    @staticmethod
    def item_eta_s(item: QueueItem, rate_for: Callable, fraction_done: float = 0.0) -> tuple:
        """`(seconds, complete)` for one item. Its engines run side by side, so
        the slowest one sets the time. `rate_for(engine_id)` is characters per
        second or None; an engine with no rate is left out and `complete` is
        False. An item with no characters known yet is not complete either."""
        if not item.engine_chars:
            return 0.0, item.kind == CLIPS
        left = max(0.0, 1.0 - max(0.0, min(1.0, fraction_done)))
        seconds, complete = 0.0, True
        for engine_id, chars in item.engine_chars.items():
            rate = rate_for(engine_id)
            if not rate or rate <= 0:
                complete = False
                continue
            seconds = max(seconds, chars * left / rate)
        return seconds, complete

    def eta_s(self, rate_for: Callable, running_fraction: float = 0.0) -> tuple:
        """`(seconds, complete)` for everything still to do: queued items in
        full, the running one by what is left of it. `complete` is False when
        some item's time can't be estimated, so the figure is a floor."""
        total, complete = 0.0, True
        for item in self.pending():
            seconds, known = self.item_eta_s(item, rate_for, running_fraction if item.state == RUNNING else 0.0)
            total += seconds
            complete = complete and known
        return total, complete

    # -- saving ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {"items": [i.to_dict() for i in self.items]}

    @classmethod
    def from_dict(cls, raw) -> "GenerationQueue":
        """The queue `raw` describes; anything that isn't a queue gives an empty
        one, and an item that can't be one is dropped."""
        queue = cls()
        items = raw.get("items") if isinstance(raw, dict) else None
        if not isinstance(items, list):
            return queue
        seen = set()
        for entry in items[:MAX_ITEMS]:
            item = item_from_dict(entry)
            if item is None or item.id in seen:
                continue
            seen.add(item.id)
            queue.items.append(item)
        return queue


def plan_items(project_id: str, entries, title: str, batch: int = BATCH_CLIPS) -> list:
    """Splits clips into "clips" items of at most `batch` clips, in the order
    given (the caller passes text order). `entries` is `[(clip_id, chars,
    engine_id), ...]`. One item keeps `title`; several add the clip range
    ("Chapter 3, clips 9-16 of 40")."""
    entries = list(entries)
    batch = max(1, int(batch))
    total = len(entries)
    items = []
    for start in range(0, total, batch):
        chunk = entries[start:start + batch]
        engine_chars: dict = {}
        for _clip_id, chars, engine_id in chunk:
            engine_chars[engine_id] = engine_chars.get(engine_id, 0) + int(chars)
        label = title if total <= batch else f"{title}, clips {start + 1}-{start + len(chunk)} of {total}"
        items.append(QueueItem(kind=CLIPS, project_id=project_id, clip_ids=[c for c, _n, _e in chunk], title=label,
                               chars=sum(engine_chars.values()), engine_chars=engine_chars))
    return items
