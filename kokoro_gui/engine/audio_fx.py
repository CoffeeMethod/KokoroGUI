"""Post-processing audio FX chain (pitch, volume, time stretch, Pedalboard FX, normalize, trim)."""
import logging
import math

import numpy as np
import scipy.signal
from pedalboard import time_stretch as pedalboard_time_stretch
from pedalboard import (
    Pedalboard, Reverb, Compressor, HighShelfFilter, LowShelfFilter,
    Chorus, Distortion, Phaser, Clipping, Gain, Limiter,
    HighpassFilter, LowpassFilter, LadderFilter, Delay, PitchShift,
    GSMFullRateCompressor, Bitcrush, Convolution, Chain, Mix
)

from kokoro_gui.engine.presets import resolve_ir

logger = logging.getLogger(__name__)

# Pedalboard's Convolution (JUCE underneath) scales every impulse response to
# unit energy and then applies a fixed 0.125 (-18 dB) to the wet signal.
# The chain undoes the fixed part, so a one-sample IR is the identity at mix
# 1.0 and any IR's wet signal sits near the dry level. Measured on
# pedalboard 0.9.23 (the pinned version).
CONVOLUTION_WET_MAKEUP_DB = 20.0 * math.log10(8.0)

# (name, project_dir) pairs already reported missing, so a clip with a
# missing IR logs once, not once per segment.
_warned_missing_irs = set()

# Pitch range the Generation dock's spinbox allows (see pitch_spin.setRange
# in kokoro_gui/qt/docks/generation_dock.py). A preset's `pitch` bypasses
# that spinbox entirely (presets are untrusted JSON - see
# Claude/SECURITY_AUDIT.md), so both places that turn it into a resample
# factor (here and caching.py's ETA speed compensation) clamp to this range
# first: `2 ** (pitch/12.0)` is otherwise unbounded and can OverflowError or
# attempt a multi-GB scipy.signal.resample allocation at extreme values.
# Trim silence's threshold on |sample|. `wordtiming.silence_bounds` measures
# a segment's onset and tail against the same value at generation time, so
# the trimmed length can be computed without reading the file.
TRIM_THRESHOLD = 0.01

PITCH_SEMITONES_MIN = -12.0
PITCH_SEMITONES_MAX = 12.0

# `time_stretch` factor bounds (1.0 = none, above 1 = faster and shorter).
# Fit to slot stays well inside them; the clamp is for a hand-edited
# document.json, where 0.001 would ask for a thousandfold longer buffer.
TIME_STRETCH_MIN = 0.5
TIME_STRETCH_MAX = 2.0


def clamp_pitch_semitones(pitch_semitones):
    """Coerces `pitch_semitones` to a float and clamps it to the GUI's
    -12..12 range. Falls back to 0.0 (no pitch shift) for a non-numeric
    value rather than raising, matching the existing tolerant `config.get`
    style used throughout this pipeline."""
    try:
        pitch_semitones = float(pitch_semitones)
    except (TypeError, ValueError):
        return 0.0
    return max(PITCH_SEMITONES_MIN, min(PITCH_SEMITONES_MAX, pitch_semitones))


def _convolution_stage(config):
    """The convolution reverb for `config` (grill Q31), or None when there's
    nothing to add: no `convolution_ir`, a mix of 0, or a name that resolves
    to no file (a logged warning, never an exception). The IR is looked up
    by `resolve_ir(name, config["project_dir"])`: the project's
    `fx/ir/<name>.wav` first, then `presets/fx/ir/<name>.wav`. A mix below
    1.0 runs the dry signal and the wet chain in parallel."""
    name = config.get('convolution_ir')
    if not isinstance(name, str) or not name:
        return None
    try:
        mix = min(1.0, max(0.0, float(config.get('convolution_mix', 0.5))))
    except (TypeError, ValueError):
        mix = 0.5
    if mix <= 0.0:
        return None
    project_dir = config.get('project_dir')
    path = resolve_ir(name, project_dir)
    if path is None:
        if (name, project_dir) not in _warned_missing_irs:
            _warned_missing_irs.add((name, project_dir))
            logger.warning("Impulse response %r not found; convolution reverb skipped", name)
        return None
    # Read with soundfile rather than handing Convolution the path: given a
    # file it can't decode, Convolution loads nothing and passes the input
    # through, which the makeup gain would then turn up by 18 dB.
    try:
        import soundfile as sf

        data, ir_rate = sf.read(path, dtype="float32", always_2d=True)
    except Exception as e:  # noqa: BLE001 - an unreadable IR is skipped like a missing one
        logger.warning("Impulse response %r couldn't be read (%s); convolution reverb skipped", name, e)
        return None
    if data.size == 0 or not np.all(np.isfinite(data)) or not np.any(data):
        logger.warning("Impulse response %r is empty or silent; convolution reverb skipped", name)
        return None
    impulse = np.ascontiguousarray(data[:, 0] if data.shape[1] == 1 else data.T)
    try:
        convolution = Convolution(impulse, 1.0, float(ir_rate))
    except Exception as e:  # noqa: BLE001
        logger.warning("Impulse response %r couldn't be loaded (%s); convolution reverb skipped", name, e)
        return None
    wet = Chain([convolution, Gain(gain_db=CONVOLUTION_WET_MAKEUP_DB + 20.0 * math.log10(mix))])
    if mix >= 1.0:
        return wet
    return Mix([Gain(gain_db=20.0 * math.log10(1.0 - mix)), wet])


def clamp_time_stretch(factor):
    """`factor` as a float clamped to `TIME_STRETCH_MIN..TIME_STRETCH_MAX`;
    1.0 (no stretch) for a missing or non-numeric value."""
    try:
        factor = float(factor)
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(factor):
        return 1.0
    return max(TIME_STRETCH_MIN, min(TIME_STRETCH_MAX, factor))


def process_audio(audio, sr, config):
    """The post-processing stage, in the order the numbered comments below
    run: trim silence, volume, pitch (resample), time stretch, the
    Pedalboard FX chain, normalize. Reads only `config`, never engine state, so it runs equally
    well inside `process_chunk_task` (the whole-document path) and at read
    time from `kokoro_gui.audio.post.render` (clip playback/export, where FX
    are applied on top of the raw segment file every time the settings
    change). Returns the processed mono float array at the same `sr`."""
    # 1. Trim Silence (Simple threshold)
    if config.get('trim_silence', False):
        threshold = TRIM_THRESHOLD
        # Find first index > threshold
        mask = np.abs(audio) > threshold
        if np.any(mask):
            start = np.argmax(mask)
            end = len(audio) - np.argmax(mask[::-1])
            audio = audio[start:end]

    # 2. Volume / Gain
    vol = config.get('volume', 1.0)
    if vol != 1.0:
        audio = audio * vol

    # 3. Pitch Shift (Resampling)
    pitch_semitones = clamp_pitch_semitones(config.get('pitch', 0.0))
    if pitch_semitones != 0.0:
        factor = 2 ** (pitch_semitones / 12.0)
        new_len = int(len(audio) / factor)
        if new_len > 0:
            try:
                audio = scipy.signal.resample(audio, new_len)
            except Exception as e:
                print(f"Resample failed: {e}")

    # 4. Time stretch: length divided by the factor, pitch kept. Fit to
    # slot sets it on a clip whose engine has no speed control.
    stretch = clamp_time_stretch(config.get('time_stretch', 1.0))
    if stretch != 1.0 and len(audio):
        try:
            stretched = pedalboard_time_stretch(np.asarray(audio, dtype=np.float32), sr, stretch)
            audio = np.asarray(stretched, dtype=np.float32).reshape(-1)
        except Exception as e:
            print(f"Time stretch failed: {e}")

    # 5. Pedalboard FX
    fx_chain = []

    if config.get('apply_fx', True):
        # --- Guitar / Modulation ---
        if config.get('distortion_enabled', False):
            drive = config.get('distortion_drive', 25.0)
            fx_chain.append(Distortion(drive_db=drive))

        if config.get('chorus_enabled', False):
            fx_chain.append(Chorus(
                rate_hz=config.get('chorus_rate', 1.0),
                depth=config.get('chorus_depth', 0.25),
                mix=config.get('chorus_mix', 0.5)
            ))

        if config.get('phaser_enabled', False):
            fx_chain.append(Phaser(
                rate_hz=config.get('phaser_rate', 1.0),
                depth=config.get('phaser_depth', 0.5),
                mix=config.get('phaser_mix', 0.5)
            ))

        if config.get('clipping_enabled', False):
            fx_chain.append(Clipping(threshold_db=config.get('clipping_thresh', -6.0)))

        if config.get('bitcrush_enabled', False):
            fx_chain.append(Bitcrush(bit_depth=config.get('bitcrush_depth', 8.0)))

        if config.get('gsm_enabled', False):
            fx_chain.append(GSMFullRateCompressor())

        # --- Filters / EQ ---
        # HighPass
        if config.get('highpass_enabled', False):
            fx_chain.append(HighpassFilter(cutoff_frequency_hz=config.get('highpass_freq', 50.0)))

        # LowPass
        if config.get('lowpass_enabled', False):
            fx_chain.append(LowpassFilter(cutoff_frequency_hz=config.get('lowpass_freq', 10000.0)))

        # Shelves (Bass/Treble) - Simple EQ
        bass_db = config.get('eq_bass', 0.0)
        if bass_db != 0.0:
            fx_chain.append(LowShelfFilter(cutoff_frequency_hz=250, gain_db=bass_db))

        treble_db = config.get('eq_treble', 0.0)
        if treble_db != 0.0:
            fx_chain.append(HighShelfFilter(cutoff_frequency_hz=4000, gain_db=treble_db))

        # --- Spatial / Time ---
        if config.get('pitch_shift_enabled', False):
            # High quality pitch shifting without duration change
            semitones = config.get('pitch_shift_semitones', 0.0)
            if semitones != 0:
                fx_chain.append(PitchShift(semitones=semitones))

        if config.get('delay_enabled', False):
            fx_chain.append(Delay(
                delay_seconds=config.get('delay_time', 0.5),
                feedback=config.get('delay_feedback', 0.0),
                mix=config.get('delay_mix', 0.5)
            ))

        if config.get('reverb_enabled', False):
            fx_chain.append(Reverb(
                room_size=config.get('reverb_room_size', 0.5),
                damping=config.get('reverb_damping', 0.5),
                wet_level=config.get('reverb_wet_level', 0.3),
                dry_level=config.get('reverb_dry_level', 1.0),
                width=config.get('reverb_width', 1.0)
            ))

        # Convolution reverb goes after the algorithmic reverb and before
        # the dynamics, so the compressor and limiter see its tail.
        convolution = _convolution_stage(config)
        if convolution is not None:
            fx_chain.append(convolution)

        # --- Dynamics ---
        if config.get('comp_enabled', False):
            fx_chain.append(Compressor(
                threshold_db=config.get('comp_threshold', -20),
                ratio=config.get('comp_ratio', 4),
                attack_ms=config.get('comp_attack', 1.0),
                release_ms=config.get('comp_release', 100.0)
            ))

        if config.get('limiter_enabled', False):
            fx_chain.append(Limiter(
                threshold_db=config.get('limiter_threshold', -1.0),
                release_ms=config.get('limiter_release', 100.0)
            ))

        if config.get('gain_enabled', False):
            db = config.get('gain_db', 0.0)
            if db != 0.0:
                fx_chain.append(Gain(gain_db=db))

    if fx_chain:
        try:
            board = Pedalboard(fx_chain)
            # Pedalboard expects float32
            audio = board(audio, sr)
        except Exception as e:
            print(f"Pedalboard FX failed: {e}")

    # 6. Normalization
    if config.get('normalize', False):
        peak = np.max(np.abs(audio))
        if peak > 0:
            target_peak = 0.98
            audio = audio / peak * target_peak

    return audio


class AudioFXMixin:
    def process_audio(self, audio, sr, config):
        return process_audio(audio, sr, config)
