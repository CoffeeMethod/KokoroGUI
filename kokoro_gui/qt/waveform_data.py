"""Min/max peak decimation for the waveform-view spike
(Workstream 3 of Claude/PLAN_daw_ui_ux_redesign.md).

Lives under `kokoro_gui/qt/` rather than `kokoro_gui/engine/`: this is
presentation-layer decimation for one specific widget (waveform_view.py),
not a general engine capability - even though the module itself has zero Qt
imports and is plain NumPy.

This is a spike-scoped module. Two simplifications are called out explicitly
rather than hidden:

- Stereo input is downmixed by a plain per-frame average across channels.
  This phase-cancels a hypothetical inverted-phase stereo signal into
  apparent silence - acceptable for a spike, worth revisiting only if it
  ever matters in practice.
- `compute_peaks` always recomputes from raw samples; there's no
  multi-resolution/mipmap cache. Fine for clip-length audio at any bucket
  count reachable from a real window size - a full-build pass should
  benchmark real multi-minute files before deciding whether that's needed.
"""
import numpy as np
import soundfile as sf


def compute_peaks(samples: np.ndarray, sample_rate: int, bucket_count: int) -> np.ndarray:
    """Returns a `(bucket_count, 2)` float32 array of `(min, max)` pairs -
    one pair per horizontal "pixel bucket", the standard technique for
    rendering a waveform without holding/redrawing every raw sample.

    `samples` may be 1-D (mono) or 2-D `(n, channels)` (stereo/multi-channel,
    downmixed to mono via a plain average across channels - see module
    docstring). `sample_rate` is accepted for interface symmetry with
    `load_peaks_from_file` but unused here - decimation only cares about
    sample *count*, not real time; duration is the caller's concern.

    Bucket count vs. sample count:
    - `bucket_count <= len(samples)`: samples are split into `bucket_count`
      near-equal groups via `np.array_split` (NumPy's own "distribute N
      items into K groups" rule), each group reduced to its own min/max.
    - `bucket_count > len(samples)` (more pixels than samples - a short
      clip in a wide view): bucket `i` for `i < len(samples)` gets
      `(samples[i], samples[i])` (a single real sample, min == max);
      buckets beyond that stay `(0.0, 0.0)`. Deliberately not carried
      forward from the last real sample - that would visually fabricate a
      continuous waveform out of a handful of real data points.

    All-silence and empty (`len(samples) == 0`) inputs both fall out to an
    all-zero result without any special-casing.
    """
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    samples = np.asarray(samples, dtype=np.float32)

    peaks = np.zeros((bucket_count, 2), dtype=np.float32)
    if samples.size == 0 or bucket_count <= 0:
        return peaks

    if bucket_count <= samples.size:
        for i, bucket in enumerate(np.array_split(samples, bucket_count)):
            peaks[i, 0] = bucket.min()
            peaks[i, 1] = bucket.max()
    else:
        n = samples.size
        peaks[:n, 0] = samples
        peaks[:n, 1] = samples

    return peaks


def load_peaks_from_file(path: str, bucket_count: int):
    """Reads `path` via `soundfile` (same `sf.read(..., dtype='float32')`
    call pattern `playback.py`/`kokoro_gui/engine/caching.py` already use),
    and returns `(compute_peaks(...), duration_seconds)`. File I/O is kept
    separate from `compute_peaks`'s pure array math so that function stays
    testable with in-memory arrays and no temp files."""
    data, sample_rate = sf.read(path, dtype="float32")
    duration = len(data) / sample_rate if sample_rate else 0.0
    return compute_peaks(data, sample_rate, bucket_count), duration
