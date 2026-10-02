"""How the reference video follows the transport (phase 5, TB16). Pure
functions, no Qt, so the decisions are testable without a media player.

The transport is the clock: its 30 Hz `positionChanged` and its
`stateChanged` drive the video dock's player, and nothing the player does
moves the transport. The player runs on its own clock while playing and
is only corrected when it drifts more than `DRIFT_MS` from where the
transport says it should be; a smaller gap is left alone, because every
seek stalls the decoder for a frame or two.
"""
from __future__ import annotations

# The gap between the player's position and the transport's before the
# player is seeked, in milliseconds.
DRIFT_MS = 40


def target_ms(transport_s: float, offset_s: float) -> int:
    """Where the video should be for transport time `transport_s`: the
    transport time plus the project's video offset, in milliseconds, never
    before the start of the file."""
    return max(0, int((float(transport_s) + float(offset_s)) * 1000))


def seek_target_ms(transport_s: float, offset_s: float, player_ms: int, threshold_ms: int = DRIFT_MS) -> int | None:
    """The position to seek the player to, or None when it is within
    `threshold_ms` of `target_ms` and should be left to play on."""
    target = target_ms(transport_s, offset_s)
    if abs(target - int(player_ms)) > threshold_ms:
        return target
    return None


def player_command(transport_state: str) -> str:
    """"play" while the transport plays, else "pause". A stopped transport
    pauses the player rather than stopping it, so the frame at the playhead
    stays on screen; the playhead's jump back to 0 seeks it there."""
    return "play" if transport_state == "playing" else "pause"
