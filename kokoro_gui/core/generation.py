"""Generation bookkeeping without threads or widgets: which clips a batch
generates and with which engine, and landing the outcomes.

`TimelineDock.generate_dirty_clips_requested` and `start_batch` call
`generatable` and `plan_batch`; `_on_batch_generation_raw` calls
`apply_outcomes` with the dock's `_apply_results`. The threads, the busy
check, signals, progress counters, ripple, fit to slot and word alignment
stay in the dock and the app.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from kokoro_gui.engines.missing import MissingBackend


@dataclass
class Batch:
    """What one batch generates. `clips` is the generatable clips in the
    order they came in; `groups` maps an engine id to `(backend, [(clip_id,
    text, config), ...])`, one `generate_clips` call per engine (the engines
    run side by side on their own workers); `blocked` maps a reason
    (`MissingBackend.message`) to how many clips it kept out."""
    clips: list = field(default_factory=list)
    groups: dict = field(default_factory=dict)
    blocked: dict = field(default_factory=dict)


def generatable(clips, backend_for_clip: Callable) -> tuple:
    """`(kept, blocked)`: the clips a generate can run on, in order, and
    how many were kept out per reason. A nested clip isn't TTS (a stale
    subproject generates through its own document) and an imported clip's
    audio is its recording, so both are dropped without a count. A clip
    whose engine isn't installed stays stale (grill EN6) and counts under
    its `MissingBackend.message`."""
    kept = []
    blocked: dict = {}
    for clip in clips:
        if clip.is_nested or clip.source == "imported":
            continue
        backend = backend_for_clip(clip)
        if isinstance(backend, MissingBackend):
            blocked[backend.message] = blocked.get(backend.message, 0) + 1
            continue
        kept.append(clip)
    return kept, blocked


def plan_batch(document, clips, backend_for_clip: Callable, config_for_clip: Callable) -> Batch:
    """The `Batch` for `clips` of `document`: `generatable`'s filter, then
    each kept clip's text and `config_for_clip(clip)` (the app's
    `_assemble_clip_config`) grouped by its backend's id."""
    kept, blocked = generatable(clips, backend_for_clip)
    groups: dict = {}
    for clip in kept:
        text = document.clip_text(clip)
        config = config_for_clip(clip)
        backend = backend_for_clip(clip)
        groups.setdefault(backend.id, (backend, []))[1].append((clip.id, text, config))
    return Batch(clips=kept, groups=groups, blocked=blocked)


def apply_outcomes(document, outcomes, apply: Callable) -> tuple:
    """Lands a batch's outcomes (`generate_clips`' dicts with `clip_id`,
    `success` and `results`) on `document`: `apply(clip, results)` for each
    success. Returns `(succeeded_ids, failed_ids)`.

    A clip deleted, or moved into a subproject, while it generated is
    skipped: its file stays a cache entry, and nothing lands or fails. A
    clip edited meanwhile gets the results too; their key is the old
    text's, so it stays stale (plan 29). A failed or cancelled clip keeps
    its segments as they were."""
    succeeded_ids = []
    failed_ids = []
    for outcome in outcomes:
        clip_id = outcome["clip_id"]
        clip = document.get_clip(clip_id)
        if clip is None:
            continue
        if outcome["success"]:
            apply(clip, outcome["results"])
            succeeded_ids.append(clip_id)
        else:
            failed_ids.append(clip_id)
    return succeeded_ids, failed_ids
