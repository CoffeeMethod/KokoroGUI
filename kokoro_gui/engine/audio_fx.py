"""Post-processing audio FX chain (pitch, volume, Pedalboard FX, normalize, trim)."""
import numpy as np
import scipy.signal
from pedalboard import (
    Pedalboard, Reverb, Compressor, HighShelfFilter, LowShelfFilter,
    Chorus, Distortion, Phaser, Clipping, Gain, Limiter,
    HighpassFilter, LowpassFilter, LadderFilter, Delay, PitchShift,
    GSMFullRateCompressor, Bitcrush
)


class AudioFXMixin:
    def process_audio(self, audio, sr, config):
        """
        Apply post-processing: Pitch (Resample), Volume, FX (Reverb, EQ, Comp), Normalize, Trim.
        Returns: (processed_audio, new_sr)
        """
        # 1. Trim Silence (Simple threshold)
        if config.get('trim_silence', False):
            threshold = 0.01
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
        pitch_semitones = config.get('pitch', 0.0)
        if pitch_semitones != 0.0:
            factor = 2 ** (pitch_semitones / 12.0)
            new_len = int(len(audio) / factor)
            if new_len > 0:
                try:
                    audio = scipy.signal.resample(audio, new_len)
                except Exception as e:
                    print(f"Resample failed: {e}")

        # 4. Pedalboard FX
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

        # 5. Normalization
        if config.get('normalize', False):
            peak = np.max(np.abs(audio))
            if peak > 0:
                target_peak = 0.98
                audio = audio / peak * target_peak

        return audio
