"""Record a guide take (plan 31, grill PG8).

`store_guide_take` turns a recording into a take on a clip: the samples go
to a temp WAV, `project.import_audio_file` copies it into the project's
`audio/imported/`, and one `AddTakeCommand` adds a one-segment take and
makes it the active one. The segment has no `cache_key` and its
`range` starts after the count-in, so playback and export skip the beeps.
`dirty.is_recorded_take` reads those two facts: a clip whose active take
is recorded is never stale. No `.tbaw` field is new: the take is stored
like any other, and its file is bundled with the imported audio.
"""
from __future__ import annotations

import os
import tempfile
from typing import Optional

import numpy as np

from kokoro_gui.audio import recorder as recorder_module
from kokoro_gui.daw import revision
from kokoro_gui.daw.models import Segment
from kokoro_gui.daw.undo import AddTakeCommand

# A recording that ends this soon after the count-in holds nothing to keep.
MIN_TAKE_S = 0.1


def store_guide_take(app, clip_id: str, samples, sample_rate: int,
                     count_in_s: float = recorder_module.COUNT_IN_S) -> Optional[Segment]:
    """Adds `samples` (mono float32, the count-in first) to clip `clip_id`
    as its new active take, one undo step on the clip's project. Returns
    the take's segment, or None with a status line when the clip is gone,
    the recording is too short or the file couldn't be stored."""
    import soundfile as sf

    from kokoro_gui.qt import project as project_io

    project = app.project_of_clip_id(clip_id)
    clip = project.document.get_clip(clip_id) if project is not None else None
    if clip is None:
        app.set_status("The clip was deleted while recording; the take was not kept.", "warning")
        return None
    data = np.asarray(samples, dtype=np.float32).reshape(-1)
    seconds = len(data) / float(sample_rate)
    if seconds < count_in_s + MIN_TAKE_S:
        app.set_status("The recording stopped before the count-in ended; nothing was kept.", "warning")
        return None
    if not project.project_dir:
        app.set_status("This project has no folder to store the take in.", "warning")
        return None
    fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="guide_take_")
    os.close(fd)
    try:
        sf.write(tmp, data, int(sample_rate), subtype="PCM_16")
        revision.bump_files()  # a file is about to be copied into the project
        path = project_io.import_audio_file(tmp, project.project_dir)
    except (OSError, RuntimeError, project_io.ProjectError) as e:
        app.set_status(f"Couldn't store the take: {e}", "error")
        return None
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    end_s = round(seconds, 3)
    start_s = round(float(count_in_s), 3)
    segment = Segment(order_index=0, text=" ".join(project.document.clip_text(clip).split()), cache_key=None,
                      audio_path=path, duration=round(end_s - start_s, 3), raw=True, range=[start_s, end_s])
    project.document.undo_stack.push(AddTakeCommand(clip_id, [segment]))
    if app.editor is not None:
        app.editor.rehighlight()
    app.schedule_save()
    app.refresh_timeline()
    take = int(clip.overrides.get("take", 0) or 0)
    app.set_status(f"Recorded take {take + 1} ({end_s - start_s:.1f}s).", "info")
    return segment
