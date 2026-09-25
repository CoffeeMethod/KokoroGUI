"""Track layouts (grill PR4): which track each clip sits on.

Two layouts, chosen by `Document.settings["track_layout"]` (read through
`Document.track_layout()`):

- One per character (the default). A character's track is made the first
  time the transcript uses it (`Document.track_for_character`), and a
  track with no clips isn't drawn.
- Unified, N lanes (`{"mode": "unified", "lanes": N}`). Clips go on N plain
  tracks named "Lane 1".."Lane N" (`Track.lane`, no character) by
  `assign_lanes`: in text order, the first clip takes lane 1, a clip with
  the same character as the previous one stays on its lane, and a
  character change moves to the next lane, wrapping after N. So a
  conversation alternates lanes and overlapping speakers never share one.
  Mute, solo, fader and pan act per lane; a character's own level stays in
  its preset's volume.

Nothing here moves a clip in time. `RelaneCommand` (kokoro_gui/daw/undo.py)
applies `plan_relane`; `relane_follow_up` is the undo stack's hook that
adds one to the same undo step as an edit that changed clip order or a
clip's character (or switched the layout). Typing in the transcript isn't on
that stack, so lanes catch up at the next such edit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from kokoro_gui.daw.models import Track

LANE_NAME = "Lane {}"


def text_ordered_clips(document) -> list:
    """Clips with text, in text order."""
    placed = []
    for clip in document.clips:
        extent = document.clip_extent(clip.id)
        if extent is not None:
            placed.append((extent[0], clip))
    placed.sort(key=lambda item: item[0])
    return [clip for _start, clip in placed]


def lane_numbers(document, lanes: int) -> dict:
    """`{clip_id: lane}` (1-based) by the lane rule over `lanes` lanes."""
    lanes = max(1, int(lanes))
    out = {}
    lane = 0
    previous_character = object()  # matches nothing, so the first clip starts lane 1
    for clip in text_ordered_clips(document):
        if not out:
            lane = 1
        elif clip.character_id != previous_character:
            lane = lane % lanes + 1
        out[clip.id] = lane
        previous_character = clip.character_id
    return out


def lane_tracks(document) -> dict:
    """`{lane number: Track}` for the unified layout's lane tracks."""
    out = {}
    for track in document.tracks:
        if track.lane is not None and track.lane not in out:
            out[track.lane] = track
    return out


def _next_order(document, new_tracks) -> int:
    return max((t.order_index for t in [*document.tracks, *new_tracks]), default=-1) + 1


def assign_lanes(document, lanes: int) -> dict:
    """`{clip_id: track_id}` for the unified layout over `lanes` lanes,
    every clip included. Read-only: a lane with no track yet maps to a
    track this call makes up and doesn't keep, so apply the rule with
    `RelaneCommand` when the tracks have to exist."""
    return plan_relane(document, {"mode": "unified", "lanes": lanes}, only_changes=False).assignments


@dataclass
class RelanePlan:
    """Clips whose `track_id` changes, and the tracks to add first."""

    assignments: dict = field(default_factory=dict)
    new_tracks: list = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.assignments or self.new_tracks)


def plan_relane(document, layout: dict | None = None, only_changes: bool = True) -> RelanePlan:
    """What `RelaneCommand` does under `layout` (default: the document's):
    the unified lane rule, or, in the per-character layout, every clip back
    on its own character's track. Tracks that don't exist yet are in
    `new_tracks`; only lanes and characters actually used get one. With
    `only_changes`, clips already on the right track are left out."""
    layout = layout or document.track_layout()
    plan = RelanePlan()
    wanted = {}
    if layout["mode"] == "unified":
        existing = lane_tracks(document)
        for clip_id, lane in lane_numbers(document, layout["lanes"]).items():
            track = existing.get(lane)
            if track is None:
                track = Track(name=LANE_NAME.format(lane), lane=lane,
                              order_index=_next_order(document, plan.new_tracks))
                existing[lane] = track
                plan.new_tracks.append(track)
            wanted[clip_id] = track.id
        plan.new_tracks.sort(key=lambda t: t.lane)
        base = min((t.order_index for t in plan.new_tracks), default=0)
        for i, track in enumerate(plan.new_tracks):
            track.order_index = base + i
    else:
        by_character = {t.character_id: t for t in document.tracks if t.character_id is not None}
        for clip in text_ordered_clips(document):
            if clip.character_id is None:
                continue
            track = by_character.get(clip.character_id)
            if track is None:
                character = document.get_character(clip.character_id)
                if character is None:
                    continue
                track = Track(name=character.name, character_id=character.id,
                              order_index=_next_order(document, plan.new_tracks))
                by_character[character.id] = track
                plan.new_tracks.append(track)
            wanted[clip.id] = track.id
    for clip_id, track_id in wanted.items():
        clip = document.get_clip(clip_id)
        if not only_changes or (clip is not None and clip.track_id != track_id):
            plan.assignments[clip_id] = track_id
    return plan


def relane_follow_up(document, command):
    """The undo stack's `follow_up` hook: a `RelaneCommand` after a layout
    switch, or, in the unified layout, after an edit that changed clip order
    or a clip's character; None when nothing would move."""
    from kokoro_gui.daw.undo import (
        AssignCharacterCommand, ImportCuesCommand, MoveClipBeforeCommand, ReassignTrackCommand, RelaneCommand,
        ReplaceWithNestedCommand, SetFieldCommand, TextEditCommand,
    )

    switched = (isinstance(command, SetFieldCommand) and command.target_kind == "document"
                and command.field == "settings" and command.key == "track_layout")
    reordered = isinstance(command, (AssignCharacterCommand, TextEditCommand, MoveClipBeforeCommand,
                                     ReassignTrackCommand, ReplaceWithNestedCommand, ImportCuesCommand))
    if not switched and not (reordered and document.track_layout()["mode"] == "unified"):
        return None
    if not plan_relane(document):
        return None
    return RelaneCommand()
