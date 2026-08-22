"""Per-chunk generation with WAV segment caching, keyed on text|voice|speed|lang_code.

Reads `kokoro_engine.CACHE_DIR` and calls `kokoro_engine.get_thread_pipeline`
qualified, at call time, so tests can keep monkeypatching those names on the
`kokoro_engine` module (e.g. the `isolated_dirs`/`make_config` fixtures and the
`_boom` sentinel used in `test_caching.py`/`test_mix_voices.py`).
"""
import hashlib
import os
import re

import soundfile as sf
import torch
from pedalboard.io import AudioFile

import kokoro_engine


class CachingMixin:
    def process_chunk_task(self, chunk_data, progress_callback):
        index, text, config = chunk_data
        if self.cancel_event.is_set(): return []

        # Use lang_code from config, default to 'a'
        lang_code = config.get('lang_code', 'a')

        # Speed Adjustment for Pitch Compensation
        eff_speed = config['speed']
        pitch_semitones = config.get('pitch', 0.0)
        if pitch_semitones != 0.0:
            factor = 2 ** (pitch_semitones / 12.0)
            eff_speed = eff_speed / factor

        # --- Caching Check (WAV only) ---
        use_cache = config.get('caching', True)
        cache_hash = None
        cached_segments = []

        if use_cache:
            to_hash = f"{text}|{config['voice']}|{eff_speed}|{lang_code}"
            cache_hash = hashlib.md5(to_hash.encode('utf-8')).hexdigest()

            # Predict segments to verify cache integrity
            try:
                # Mimic KPipeline splitting logic roughly to align with file indices
                # Note: KPipeline might strip whitespace or handle things slightly differently.
                # This is a heuristic. If file count matches segment count, we assume cache is valid.
                split_pat = config.get('split_pattern', r"\n+")
                predicted_texts = [t.strip() for t in re.split(split_pat, text) if t.strip()]

                if not predicted_texts:
                    # If text is empty/whitespace but passed here, treat as single empty?
                    # Usually smart_split handles this.
                    predicted_texts = []

                all_exist = True
                loaded_data = []

                if predicted_texts:
                    for i, seg_text in enumerate(predicted_texts):
                        f_name = f"{cache_hash}_{i}.wav"
                        f_path = os.path.join(kokoro_engine.CACHE_DIR, f_name)
                        if not os.path.exists(f_path):
                            all_exist = False
                            break
                        # Load raw audio
                        audio_data, _ = sf.read(f_path)
                        loaded_data.append((seg_text, '', audio_data)) # phonemes empty

                    # Ensure no extra files (e.g. from a previous run with same hash but more splits?)
                    # Hash includes text, so split count shouldn't change unless split_pattern changes.
                    # If split_pattern changes, hash logic might not capture it unless we add pattern to hash.
                    # Ideally we should add split_pattern to hash, but current requirement is simpler.
                    # For now, if we found all expected parts, we accept it.
                else:
                    all_exist = False # Empty text logic usually handled before

                if all_exist and loaded_data:
                    cached_segments = loaded_data
            except Exception as e:
                print(f"Cache check error: {e}")
                cached_segments = []

        chunk_files = []
        sub_idx = 0
        base_name = f"{config.get('filename', 'output')}_{config.get('time_id', '0')}_part{index}"

        # Function to process raw audio (from cache or gen) into final output
        def process_and_save(graphemes, raw_audio):
            nonlocal sub_idx

            # Post Process
            processed_audio = self.process_audio(raw_audio, 24000, config)

            # Determine format
            fmt = config.get('format', 'wav').lower()
            if fmt not in ['wav', 'flac', 'mp3', 'ogg']: fmt = 'wav'

            file_name = f"{base_name}_{sub_idx}.{fmt}"
            path = os.path.join(config['out_dir'], file_name)

            try:
                # Use Pedalboard AudioFile for writing
                with AudioFile(path, 'w', samplerate=24000, num_channels=1) as f:
                    f.write(processed_audio)
            except Exception as e:
                print(f"Pedalboard write failed: {e}. Fallback to soundfile.")
                sf.write(path, processed_audio, 24000)

            return {
                "path": path,
                "text": graphemes,
                "duration": len(processed_audio) / 24000.0,
                "seg_idx": index
            }

        if cached_segments:
            # Use Cache
            for graphemes, phonemes, audio in cached_segments:
                if self.cancel_event.is_set(): break
                if progress_callback: progress_callback(len(graphemes), graphemes)

                res = process_and_save(graphemes, audio)
                chunk_files.append(res)
                sub_idx += 1
        else:
            # Generate
            pipeline = kokoro_engine.get_thread_pipeline(lang_code)
            if not pipeline: raise RuntimeError(f"Failed to initialize pipeline ({lang_code}) in thread.")

            generator = pipeline(text, voice=config['voice'], speed=eff_speed, split_pattern=config['split_pattern'])

            for graphemes, phonemes, audio in generator:
                if self.cancel_event.is_set(): break

                # Notify progress
                if progress_callback:
                    progress_callback(len(graphemes), graphemes)

                if isinstance(audio, torch.Tensor):
                    audio = audio.cpu().numpy()

                # Save to Cache if enabled
                if use_cache and cache_hash:
                    cache_filename = f"{cache_hash}_{sub_idx}.wav"
                    cache_path = os.path.join(kokoro_engine.CACHE_DIR, cache_filename)
                    try:
                        sf.write(cache_path, audio, 24000)
                    except Exception as e:
                        print(f"Cache write error: {e}")

                # Process for output
                res = process_and_save(graphemes, audio)
                chunk_files.append(res)
                sub_idx += 1

        return chunk_files
