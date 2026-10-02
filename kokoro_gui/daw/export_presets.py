"""Export presets for ACX (Audible) and the podcast platforms. No Qt.

An `ExportPreset` is a set of values for the Export dialog's fields
(`values`, the same keys `export_dialog.export_defaults` returns) and the
`Check`s each written file has to pass. Picking a preset fills the fields;
editing a field afterwards makes the dialog go back to "Custom", which runs no
checks. `mixdown(checks=preset.checks)` measures the finished file and lists
the checks it failed in `ExportResult.files`.

The numbers come from ACX's audio submission requirements and Apple's audio
requirements (checked 2026-10-02): ACX wants 192 kbps CBR mp3 at 44.1 kHz,
RMS from -23 to -18 dBFS, peaks under -3 dBFS, a noise floor under -60 dBFS,
files up to 120 minutes and 1 to 5 seconds of room tone at both ends (a
recommendation, so `head_s` and `tail_s` are values and not checks). Apple
wants -16 LUFS with a 1 LU tolerance and -1 dBTP; the mono preset is 3 LU
lower because `audio.loudness.measure` counts dual-mono stereo 3 LU above the
same mono file. Spotify and YouTube normalize to -14 LUFS.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

CUSTOM_ID = "custom"
SPLIT_MODES = (None, "subprojects", "markers")

# Every preset sets all of these, so going from one preset to another never
# leaves the first one's head silence or split behind.
_BASE = {
    "normalize_loudness": True, "normalize_mode": "lufs", "target_lufs": -16.0, "ceiling_dbtp": -1.0,
    "target_rms_dbfs": -20.0, "limiter_dbfs": -3.5, "head_s": 0.0, "tail_s": 0.0, "split": None,
}

# `LoudnessReport` fields a check can read, and the unit each prints with.
_UNITS = {
    "integrated_lufs": "LUFS", "true_peak_dbtp": "dBTP", "sample_peak_dbfs": "dBFS", "rms_dbfs": "dBFS",
    "noise_floor_dbfs": "dBFS", "duration_min": "min",
}


@dataclass(frozen=True)
class Check:
    """One requirement on a finished file: `metric` (a `LoudnessReport` field,
    or "duration_min") must lie between `low` and `high` (either may be None).
    Silence reads as -inf, so it fails a lower bound and passes an upper one."""
    name: str
    metric: str
    low: Optional[float] = None
    high: Optional[float] = None

    def value(self, report) -> float:
        if self.metric == "duration_min":
            return report.duration_s / 60.0
        return float(getattr(report, self.metric))

    def passed(self, report) -> bool:
        value = self.value(report)
        if math.isnan(value):
            return False
        return (self.low is None or value >= self.low) and (self.high is None or value <= self.high)

    def needs(self) -> str:
        unit = _UNITS[self.metric]
        if self.low is not None and self.high is not None:
            return f"{self.low:g} to {self.high:g} {unit}"
        if self.high is not None:
            return f"at most {self.high:g} {unit}"
        return f"at least {self.low:g} {unit}"

    def describe(self, report) -> str:
        """"RMS -24.1 dBFS, needs -23 to -18 dBFS"."""
        value = self.value(report)
        shown = f"{value:.1f} {_UNITS[self.metric]}" if math.isfinite(value) else "silent"
        return f"{self.name} {shown}, needs {self.needs()}"


@dataclass(frozen=True)
class ExportPreset:
    id: str
    label: str
    values: dict
    checks: tuple = ()
    max_minutes: Optional[float] = None  # a longer chapter is cut into parts (`mixdown.plan_chapters`)

    def failed(self, report) -> list:
        """`describe` of every check `report` fails."""
        return [c.describe(report) for c in self.checks if not c.passed(report)]


def _podcast(id: str, label: str, fmt: str, channels: int, lufs: float, **values) -> ExportPreset:
    return ExportPreset(
        id, label,
        {**_BASE, "format": fmt, "sample_rate": 48000 if fmt == "wav" else 44100, "channels": channels,
         "target_lufs": lufs, "bitrate_kbps": 128, **values},
        (Check("Loudness", "integrated_lufs", lufs - 1.0, lufs + 1.0), Check("True peak", "true_peak_dbtp", high=-1.0)),
    )


PRESETS = (
    ExportPreset(
        "acx", "ACX (Audible)",
        {**_BASE, "format": "mp3", "bitrate_kbps": 192, "sample_rate": 44100, "channels": 1,
         "normalize_mode": "rms", "target_rms_dbfs": -20.0, "limiter_dbfs": -3.5,
         "head_s": 1.0, "tail_s": 2.0, "split": "subprojects"},
        (Check("RMS", "rms_dbfs", -23.0, -18.0), Check("Peak", "sample_peak_dbfs", high=-3.0),
         Check("Noise floor", "noise_floor_dbfs", high=-60.0), Check("Length", "duration_min", high=120.0)),
        max_minutes=120.0,
    ),
    _podcast("apple", "Apple Podcasts", "mp3", 2, -16.0),
    _podcast("apple_mono", "Apple Podcasts (mono)", "mp3", 1, -19.0),
    _podcast("spotify", "Spotify", "mp3", 2, -14.0),
    _podcast("youtube", "YouTube", "wav", 2, -14.0),
)


def get_preset(preset_id) -> Optional[ExportPreset]:
    """The preset called `preset_id`, or None for "custom" and anything unknown."""
    return next((p for p in PRESETS if p.id == preset_id), None)


def preset_ids() -> tuple:
    return (CUSTOM_ID,) + tuple(p.id for p in PRESETS)
