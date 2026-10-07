"""The keyboard map for the transport and the timeline: one table that
`QtTTSApp._build_shortcuts` turns into `QShortcut`s and the Keyboard
Shortcuts sheet lists.

Plain keys (a letter, the arrows, Home, End, Delete) only work while the
timeline has the focus, so they never type or move the caret in the
transcript (grill PG3). Each has a Ctrl-modified twin that works in any
panel. Space and Esc are the two plain keys that work everywhere: Space
because the editor and line edits claim it as text first, Esc because it is
armed only while a generate runs.

This module has no Qt import. The sequences are strings Qt parses
(`QKeySequence("Ctrl+Alt+Left")`), and `previous_time` / `next_time` are
the pure part of the clip and marker jumps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

# Where a binding listens.
WINDOW = "window"      # the main window; a widget that takes the key as text claims it first
TIMELINE = "timeline"  # only while the timeline view or a child of it has the focus
APP = "app"            # any panel (but not over a modal dialog)

SCOPES = (WINDOW, TIMELINE, APP)

# The sheet's groups.
TRANSPORT = "Transport"
TIMELINE_GROUP = "Timeline"
GENERATE = "Generate"

# How far a jump has to move the playhead, so a press at a clip's start goes
# to the one before instead of staying put.
JUMP_EPSILON_S = 0.001
# What J does while the transport can't play backwards.
JUMP_BACK_S = 5.0


@dataclass(frozen=True)
class KeyBinding:
    """`id` names the shortcut (`QtTTSApp.<id>_shortcut`), `slot_name` is the
    `QtTTSApp` method it calls, `label` is its row in the shortcut sheet.
    `while_generating` keeps the shortcut disabled until a generate runs, so
    an idle Esc still reaches the widget under it."""

    id: str
    sequence: str
    scope: str
    label: str
    slot_name: str
    group: str = TRANSPORT
    while_generating: bool = False


KEYS: tuple = (
    KeyBinding("space", "Space", WINDOW, "Play / pause", "toggle_playback"),
    KeyBinding("ctrl_space", "Ctrl+Space", APP, "Play / pause (works in any panel)", "toggle_playback"),
    KeyBinding("play", "L", TIMELINE, "Play (timeline focused)", "play_key"),
    KeyBinding("pause", "K", TIMELINE, "Pause (timeline focused)", "pause_key"),
    KeyBinding("jump_back", "J", TIMELINE, "Jump back 5 s (timeline focused)", "jump_back_key"),
    KeyBinding("go_start", "Home", TIMELINE, "Stop and return to start (timeline focused)", "go_to_start"),
    KeyBinding("go_start_any", "Ctrl+Alt+Home", APP, "Stop and return to start (works in any panel)", "go_to_start"),
    KeyBinding("go_end", "End", TIMELINE, "Go to end (timeline focused)", "go_to_end"),
    KeyBinding("go_end_any", "Ctrl+Alt+End", APP, "Go to end (works in any panel)", "go_to_end"),
    KeyBinding("clip_back", "Left", TIMELINE, "Previous clip start (timeline focused)",
               "go_to_previous_clip", TIMELINE_GROUP),
    KeyBinding("clip_back_any", "Ctrl+Alt+Left", APP, "Previous clip start (works in any panel)",
               "go_to_previous_clip", TIMELINE_GROUP),
    KeyBinding("clip_forward", "Right", TIMELINE, "Next clip start (timeline focused)",
               "go_to_next_clip", TIMELINE_GROUP),
    KeyBinding("clip_forward_any", "Ctrl+Alt+Right", APP, "Next clip start (works in any panel)",
               "go_to_next_clip", TIMELINE_GROUP),
    KeyBinding("marker_back", "Shift+Left", TIMELINE, "Previous marker (timeline focused)",
               "go_to_previous_marker", TIMELINE_GROUP),
    KeyBinding("marker_back_any", "Ctrl+Alt+Shift+Left", APP, "Previous marker (works in any panel)",
               "go_to_previous_marker", TIMELINE_GROUP),
    KeyBinding("marker_forward", "Shift+Right", TIMELINE, "Next marker (timeline focused)",
               "go_to_next_marker", TIMELINE_GROUP),
    KeyBinding("marker_forward_any", "Ctrl+Alt+Shift+Right", APP, "Next marker (works in any panel)",
               "go_to_next_marker", TIMELINE_GROUP),
    KeyBinding("split", "S", TIMELINE, "Split clip at playhead (timeline focused)",
               "split_clip_at_playhead", TIMELINE_GROUP),
    KeyBinding("zoom_fit", "F", TIMELINE, "Zoom to fit (timeline focused)", "zoom_timeline_to_fit", TIMELINE_GROUP),
    KeyBinding("snap_grid", "G", TIMELINE, "Snap to grid on or off (timeline focused)",
               "toggle_snap_to_grid", TIMELINE_GROUP),
    KeyBinding("delete_bed", "Delete", TIMELINE, "Delete the selected music bed (timeline focused)",
               "delete_selected_bed", TIMELINE_GROUP),
    KeyBinding("generate_stale", "Ctrl+G", APP, "Generate stale clips (works in any panel)",
               "generate_stale_key", GENERATE),
    KeyBinding("cancel_generate", "Esc", WINDOW, "Cancel the running generate",
               "cancel_generate_key", GENERATE, while_generating=True),
)


def bindings(scope: Optional[str] = None) -> list:
    """The table's rows, or only those listening in `scope`."""
    return [b for b in KEYS if scope is None or b.scope == scope]


def previous_time(times: Iterable[float], now: float) -> Optional[float]:
    """The latest of `times` before `now`, or None when none is."""
    earlier = [t for t in times if t < now - JUMP_EPSILON_S]
    return max(earlier) if earlier else None


def next_time(times: Iterable[float], now: float) -> Optional[float]:
    """The earliest of `times` after `now`, or None when none is."""
    later = [t for t in times if t > now + JUMP_EPSILON_S]
    return min(later) if later else None
