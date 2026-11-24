"""A clip's rendered length and samples, and the export mix, without widgets.

`QtTTSApp.clip_duration_s`, `rendered_clip_samples` and
`rendered_clip_samples_async` call the clip functions with a post config and
a rate they resolved. Nested clips, the "docks not built yet" fallback and
the duration memo stay in the app. The clip functions can write `post`'s
render memo, so the app calls them on the GUI thread only.

An export has two halves. `mix_inputs` and `export_options` read whatever
the caller's callbacks read (dock state, in the app), so `run_export` calls
them on the GUI thread. `export_mix` uses only what they returned, so it
runs on the export thread.
"""
from __future__ import annotations

import os
from typing import Callable, Optional

import numpy as np

from kokoro_gui.audio import loudness as loudness_mod, post
from kokoro_gui.daw.beds import playable_segments
from kokoro_gui.daw.mixdown import mixdown, mixdown_chapters
from kokoro_gui.daw.transcripts import clean_extras

DEFAULT_TARGET_LUFS = -16.0
DEFAULT_CEILING_DBTP = -1.0
DEFAULT_RMS_DBFS = -20.0
DEFAULT_LIMITER_DBFS = -3.5


# --- clips -------------------------------------------------------------------

def clip_duration_s(clip, post_config: Optional[dict], rate: int,
                    segments: Optional[list] = None) -> Optional[float]:
    """The rendered length of a clip's own audio (trim, pitch and stretch
    change it), or None with no audio at all. A segment with a stored onset
    and tail is measured from those, without reading its file
    (`post.duration_hint`), and a segment with a `range` as `end - start`. A
    file that can't be read counts its raw `Segment.duration`. `segments` is
    `playable_segments(clip)` when the caller already has it (a music bed's
    reads the file's header)."""
    if segments is None:
        segments = playable_segments(clip)
    if not segments:
        return None
    total = 0.0
    for segment in segments:
        try:
            total += post.rendered_duration_s(segment.audio_path, post_config, rate,
                                              hint=post.duration_hint(segment, post_config),
                                              range_s=post.segment_range(segment))
        except Exception:
            total += segment.duration or 0.0
    return total


def clip_render_requests(clip, post_config_for_clip: Callable) -> Optional[list]:
    """`[(path, post_config, range_s), ...]`: the renders whose
    concatenation is the clip's samples, or None for a clip with no audio.
    `post_config_for_clip(clip)` is only asked when there is some."""
    segments = playable_segments(clip)
    if not segments:
        return None
    post_config = post_config_for_clip(clip)
    return [(s.audio_path, post_config, post.segment_range(s)) for s in segments]


def concat_renders(parts: list, rate: int) -> Optional[tuple]:
    """`(samples, rate)` of the parts in order, or None with no parts."""
    if not parts:
        return None
    return np.concatenate(parts), rate


def render_requests(requests: list, rate: int) -> Optional[tuple]:
    """Renders `clip_render_requests`' requests in order and concatenates
    them. A file that can't be read is left out."""
    parts = []
    for path, post_config, range_s in requests:
        try:
            parts.append(post.render(path, post_config, rate, range_s))
        except Exception:
            continue
    return concat_renders(parts, rate)


def rendered_clip_samples(clip, post_config: Optional[dict], rate: int) -> Optional[tuple]:
    """`(samples, rate)` for a clip's own segments concatenated and
    post-processed, or None."""
    requests = clip_render_requests(clip, lambda _clip: post_config)
    return None if requests is None else render_requests(requests, rate)


# --- export ------------------------------------------------------------------

def mix_inputs(arrangement, post_config_for_clip: Callable, nested_path_for_clip: Optional[Callable],
               sample_rate: int, engine_id: str) -> dict:
    """The keyword arguments `render_mix` and `mixdown` share. Each placed
    clip's post config, and each nested clip's mixdown file, is resolved
    here, once, so the mix itself never calls back. With
    `nested_path_for_clip` None, no nested clip has a file."""
    post_configs = {p.clip.id: post_config_for_clip(p.clip) for p in arrangement.placed}
    nested = {p.clip.id: nested_path_for_clip(p.clip)
              for p in arrangement.placed if p.clip.is_nested} if nested_path_for_clip is not None else {}
    return {
        "sample_rate": sample_rate,
        "arrangement": arrangement,
        "engine_id": engine_id,
        "post_config_for_clip": lambda clip: post_configs.get(clip.id),
        "nested_audio_path": lambda clip: nested.get(clip.id),
    }


def loudness_options(values: dict) -> Optional[dict]:
    """`mixdown(loudness=...)` for the export settings, or None when
    "Normalize loudness" is off or pyloudnorm is missing."""
    if not values.get("normalize_loudness") or not loudness_mod.available():
        return None
    if values.get("normalize_mode") == "rms":
        return {"mode": "rms", "target_rms_dbfs": values.get("target_rms_dbfs", DEFAULT_RMS_DBFS),
                "limiter_dbfs": values.get("limiter_dbfs", DEFAULT_LIMITER_DBFS)}
    return {"mode": "lufs", "target_lufs": values.get("target_lufs", DEFAULT_TARGET_LUFS),
            "ceiling_dbtp": values.get("ceiling_dbtp", DEFAULT_CEILING_DBTP)}


def export_options(values: dict, inputs: dict, *, progress: Optional[Callable] = None,
                   tags: Optional[dict] = None, checks: tuple = ()) -> dict:
    """`mixdown`'s keyword arguments for the export settings `values` over
    `mix_inputs`' `inputs`. `values` needs `format`, `srt` and
    `keep_clip_files`; the rest default. `tags` is `mixdown(tags=...)` as
    the caller built it, `checks` the preset's checks."""
    fmt = values["format"]
    return dict(
        fmt=fmt, include_srt=values["srt"], keep_clip_files=values["keep_clip_files"],
        progress=progress, channels=values.get("channels", 2),
        srt_granularity="word" if values.get("srt_words") else "clip",
        include_cue_sheet=bool(values.get("cue_sheet")), loudness=loudness_options(values),
        bitrate_kbps=values.get("m4b_bitrate_kbps" if fmt == "m4b" else "bitrate_kbps"),
        out_rate=values.get("sample_rate"),
        head_s=values.get("head_s", 0.0), tail_s=values.get("tail_s", 0.0), checks=checks,
        stems=values.get("stems"), dialogue_stem=bool(values.get("dialogue_stem")),
        extras=tuple(clean_extras(values.get("extras"))),
        transcript_speakers=bool(values.get("transcript_speakers", True)), tags=tags, **inputs,
    )


def export_mix(document, out_path: str, options: dict, *, range_s: Optional[tuple] = None, plan=None,
               out_dir: Optional[str] = None, numbered: bool = False):
    """Writes the mix to `out_path` and returns `mixdown`'s `ExportResult`.
    `options` are `export_options`'. With `plan` (a `mixdown.ChapterPlan`)
    it writes one file per chapter into `out_dir` (default: `out_path`'s
    folder) instead; `numbered` gives each the first free name, and
    `range_s` doesn't apply."""
    if plan is not None:
        folder = os.path.dirname(out_path) if out_dir is None else out_dir
        return mixdown_chapters(document, folder, plan, numbered=numbered, **options)
    return mixdown(document, out_path, range_s=range_s, **options)
