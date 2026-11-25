"""The keyboard map for the transport and the timeline: one table that
`QtTTSApp._build_shortcuts` turns into `QShortcut`s and the Keyboard
Shortcuts sheet lists.

Plain keys (a letter, a bracket, the arrows, Home, End, Delete) only work
while the timeline has the focus, so they never type or move the caret in
the transcript (grill PG3). The arrows, Home and End have a Ctrl+Alt twin
that works in any panel. The letters and brackets don't: on a layout where
AltGr is Ctrl+Alt, a twin would swallow a typed character (AltGr+N is a
letter in Polish). Space and Esc are the two plain keys that work everywhere: Space
because the editor and line edits claim it as text first, Esc because it is
armed only while a generate runs.

This module has no Qt import. The sequences are strings Qt parses
(`QKeySequence("Ctrl+Alt+Left")`), and `previous_time` / `next_time` are
the pure part of the clip and marker jumps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from kokoro_gui.i18n import N_



# Where a binding listens.
WINDOW = "window"      # the main window; a widget that takes the key as text claims it first
TIMELINE = "timeline"  # only while the timeline view or a child of it has the focus
APP = "app"            # any panel (but not over a modal dialog)

SCOPES = (WINDOW, TIMELINE, APP)

# The sheet's groups.
TRANSPORT = N_("Transport")
TIMELINE_GROUP = N_("Timeline")
GENERATE = N_("Generate")
FIND = N_("Find")

# How far a jump has to move the playhead, so a press at a clip's start goes
# to the one before instead of staying put.
JUMP_EPSILON_S = 0.001
# What J does: the transport has no backward playback.
JUMP_BACK_S = 5.0
# The speeds L steps through while it plays: 1x, 1.5x, 2x.
LISTEN_RATES = (1.0, 1.5, 2.0)


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
    KeyBinding("space", "Space", WINDOW, N_("Play / pause"), "toggle_playback"),
    KeyBinding("ctrl_space", "Ctrl+Space", APP, N_("Play / pause (works in any panel)"), "toggle_playback"),
    KeyBinding("play", "L", TIMELINE, N_("Play, then faster: 1.5x, 2x (timeline focused)"), "play_key"),
    KeyBinding("pause", "K", TIMELINE, N_("Pause (timeline focused)"), "pause_key"),
    KeyBinding("jump_back", "J", TIMELINE, N_("Jump back 5 s (timeline focused)"), "jump_back_key"),
    KeyBinding("rate_up", "]", TIMELINE, N_("Playback speed up a step (timeline focused)"), "rate_up_key"),
    KeyBinding("rate_down", "[", TIMELINE, N_("Playback speed down a step (timeline focused)"), "rate_down_key"),
    KeyBinding("go_start", "Home", TIMELINE, N_("Stop and return to start (timeline focused)"), "go_to_start"),
    KeyBinding("go_start_any", "Ctrl+Alt+Home", APP, N_("Stop and return to start (works in any panel)"),
               "go_to_start"),
    KeyBinding("go_end", "End", TIMELINE, N_("Go to end (timeline focused)"), "go_to_end"),
    KeyBinding("go_end_any", "Ctrl+Alt+End", APP, N_("Go to end (works in any panel)"), "go_to_end"),
    KeyBinding("clip_back", "Left", TIMELINE, N_("Previous clip start (timeline focused)"),
               "go_to_previous_clip", TIMELINE_GROUP),
    KeyBinding("clip_back_any", "Ctrl+Alt+Left", APP, N_("Previous clip start (works in any panel)"),
               "go_to_previous_clip", TIMELINE_GROUP),
    KeyBinding("clip_forward", "Right", TIMELINE, N_("Next clip start (timeline focused)"),
               "go_to_next_clip", TIMELINE_GROUP),
    KeyBinding("clip_forward_any", "Ctrl+Alt+Right", APP, N_("Next clip start (works in any panel)"),
               "go_to_next_clip", TIMELINE_GROUP),
    KeyBinding("marker_back", "Shift+Left", TIMELINE, N_("Previous marker (timeline focused)"),
               "go_to_previous_marker", TIMELINE_GROUP),
    KeyBinding("marker_back_any", "Ctrl+Alt+Shift+Left", APP, N_("Previous marker (works in any panel)"),
               "go_to_previous_marker", TIMELINE_GROUP),
    KeyBinding("marker_forward", "Shift+Right", TIMELINE, N_("Next marker (timeline focused)"),
               "go_to_next_marker", TIMELINE_GROUP),
    KeyBinding("marker_forward_any", "Ctrl+Alt+Shift+Right", APP, N_("Next marker (works in any panel)"),
               "go_to_next_marker", TIMELINE_GROUP),
    KeyBinding("flag", "M", TIMELINE, N_("Drop a flag at the playhead (timeline focused)"),
               "drop_flag", TIMELINE_GROUP),
    KeyBinding("flag_back", "Shift+N", TIMELINE, N_("Previous flag (timeline focused)"),
               "go_to_previous_flag", TIMELINE_GROUP),
    KeyBinding("flag_forward", "N", TIMELINE, N_("Next flag (timeline focused)"),
               "go_to_next_flag", TIMELINE_GROUP),
    KeyBinding("split", "S", TIMELINE, N_("Split clip at playhead (timeline focused)"),
               "split_clip_at_playhead", TIMELINE_GROUP),
    KeyBinding("record_take", "R", TIMELINE, N_("Record a guide take on the selected clip (timeline focused)"),
               "record_guide_take_key", TIMELINE_GROUP),
    KeyBinding("zoom_fit", "F", TIMELINE, N_("Zoom to fit (timeline focused)"), "zoom_timeline_to_fit", TIMELINE_GROUP),
    KeyBinding("snap_grid", "G", TIMELINE, N_("Snap to grid on or off (timeline focused)"),
               "toggle_snap_to_grid", TIMELINE_GROUP),
    KeyBinding("delete_bed", "Delete", TIMELINE, N_("Delete the selected music bed (timeline focused)"),
               "delete_selected_bed", TIMELINE_GROUP),
    KeyBinding("generate_stale", "Ctrl+G", APP, N_("Generate stale clips (works in any panel)"),
               "generate_stale_key", GENERATE),
    KeyBinding("find", "Ctrl+F", APP, N_("Find in the transcript (works in any panel)"), "find_in_transcript", FIND),
    KeyBinding("replace", "Ctrl+H", APP, N_("Find and replace in the transcript (works in any panel)"),
               "replace_in_transcript", FIND),
    KeyBinding("find_next", "F3", WINDOW, N_("Next match"), "find_next_match", FIND),
    KeyBinding("find_previous", "Shift+F3", WINDOW, N_("Previous match"), "find_previous_match", FIND),
    KeyBinding("cancel_generate", "Esc", WINDOW, N_("Cancel the running generate"),
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
