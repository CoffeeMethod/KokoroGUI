"""What `speak` and `render_script` hand back.

Paths and numbers only, never audio bytes: an MCP result and a `--json` line
stay small, and the caller reads the file.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class SpeakResult:
    """`path` is None when `save=False` wrote nothing."""

    path: str | None = None
    duration_s: float = 0.0
    sample_rate: int = 0
    engine: str = ""
    voice: str = ""
    words: list = field(default_factory=list)
    cached: bool = False
    played: bool = False
    reason: str | None = None  # why not played: "no_audio_device", "disabled"
    warnings: list = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class RenderResult:
    """`clips` is the total, `regenerated` the clips this call generated (an
    update with one changed paragraph of three gives 1)."""

    project_path: str | None = None
    mixdown_path: str | None = None
    out_dir: str | None = None
    duration_s: float = 0.0
    sample_rate: int = 0
    channels: int = 1
    engine: str = ""
    clips: int = 0
    regenerated: int = 0
    speakers: list = field(default_factory=list)
    words: list = field(default_factory=list)
    played: bool = False
    reason: str | None = None
    warnings: list = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)
