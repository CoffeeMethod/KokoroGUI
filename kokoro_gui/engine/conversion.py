"""Batch conversion lifecycle: single-clip preview generation, the parallel
chunked "Standard" batch pipeline (`start_conversion` -> `_process_text_async`),
and the WAV-segment combiner shared with JIT mode.

`generate_preview` calls `kokoro_engine.get_thread_pipeline` qualified, at call
time, so tests can keep monkeypatching that name on the `kokoro_engine` module.
"""
import asyncio
import concurrent.futures
import os
import threading
import time

import numpy as np
import soundfile as sf
import torch
from pedalboard.io import AudioFile

import kokoro_engine


class ConversionMixin:
    async def generate_preview(self, text, voice, speed, output_path, extra_config=None, voice_tensor=None, lang_code='a'):
        def _gen():
            # Use specific lang code for preview
            p = kokoro_engine.get_thread_pipeline(lang_code)
            if not p: return False

            try:
                ms_segments = self.parse_multispeaker_text(text)
                # Truncate to first 2 segments for preview if many
                if len(ms_segments) > 2:
                    ms_segments = ms_segments[:2]

                all_pieces = []

                for speaker_name, fx_name, segment_text in ms_segments:
                    # Apply Lexicon if provided in extra_config
                    if extra_config and 'lexicon' in extra_config:
                        segment_text = self.apply_lexicon(segment_text, extra_config['lexicon'])

                    # Truncate segment text if too long for preview
                    if len(segment_text) > 500:
                        segment_text = segment_text[:500]

                    target_voice = voice
                    target_speed = speed
                    target_extra = extra_config.copy() if extra_config else {}

                    if speaker_name:
                        preset = self.load_preset(speaker_name)
                        if preset:
                            target_voice = preset.get('voice', target_voice)
                            target_speed = preset.get('speed', target_speed)
                            if 'volume' in preset: target_extra['volume'] = preset['volume']
                            if 'pitch' in preset: target_extra['pitch'] = preset['pitch']
                            if 'normalize' in preset: target_extra['normalize'] = preset['normalize']
                            if 'trim' in preset: target_extra['trim_silence'] = preset['trim']
                            # If speaker preset has an FX preset, it can be overridden by the colon syntax
                            if 'fx_preset' in preset:
                                target_extra['fx_preset'] = preset['fx_preset']
                            if 'apply_fx' in preset:
                                target_extra['apply_fx'] = preset['apply_fx']

                    if fx_name:
                        fx_preset = self.load_fx_preset(fx_name)
                        if fx_preset:
                            target_extra.update(fx_preset)
                            target_extra['apply_fx'] = True
                            target_extra['fx_preset'] = fx_name

                    # Resolve voice
                    if voice_tensor is not None and not speaker_name:
                        # Only use voice_tensor if no speaker name (direct preview of mix)
                        actual_voice = "_preview_temp"
                        p.voices[actual_voice] = voice_tensor
                    else:
                        actual_voice = self.resolve_voice_path(target_voice)

                    # Pitch Compensation
                    eff_speed = target_speed
                    pitch_st = target_extra.get('pitch', 0.0)
                    if pitch_st != 0.0:
                        factor = 2 ** (pitch_st / 12.0)
                        eff_speed = target_speed / factor

                    # Generate
                    generator = p(segment_text, voice=actual_voice, speed=eff_speed, split_pattern=r"\n+")
                    for _, _, audio in generator:
                        if isinstance(audio, torch.Tensor):
                            audio = audio.cpu().numpy()
                        # Post Process
                        audio = self.process_audio(audio, 24000, target_extra)
                        all_pieces.append(audio)

                if not all_pieces:
                    return False

                full_audio = np.concatenate(all_pieces)

                try:
                    with AudioFile(output_path, 'w', samplerate=24000, num_channels=1) as f:
                        f.write(full_audio)
                    return True
                except Exception as e:
                    print(f"Preview write error: {e}")
                    # Fallback
                    sf.write(output_path, full_audio, 24000)
                    return True
            except Exception as e:
                print(f"Preview error: {e}")
                return False

        return await asyncio.to_thread(_gen)

    async def smart_combine(self, file_paths, output_path, update_callback):
        def combine_worker():
            total_files = len(file_paths)
            try:
                # Use Pedalboard AudioFile
                with AudioFile(output_path, 'w', samplerate=24000, num_channels=1) as out_f:
                    for i, fp in enumerate(file_paths):
                        if self.cancel_event.is_set(): break
                        try:
                            # Read with SoundFile (reliable for reading various formats)
                            data, _ = sf.read(fp)
                            out_f.write(data)
                            if update_callback: update_callback((i + 1) / total_files)
                        except Exception as e:
                            print(f"Failed to read segment {fp}: {e}")
            except Exception as e:
                print(f"Combine failed: {e}")
        await asyncio.to_thread(combine_worker)

    def start_conversion(self, text, config):
        # Resolve voice path once before distribution
        config['voice'] = self.resolve_voice_path(config['voice'])

        self.cancel_event.clear()
        self.worker.run_coro(self._process_text_async(text, config))

    async def _process_text_async(self, text, config):
        try:
            if self.on_status: self.on_status("Preparing text...", False)
            os.makedirs(config['out_dir'], exist_ok=True)

            num_workers = config.get('num_threads', 1)

            # Multispeaker Support
            ms_segments = self.parse_multispeaker_text(text)
            tasks_data = []

            lexicon = config.get('lexicon', {})

            for speaker_name, fx_name, segment_text in ms_segments:
                # Apply Lexicon
                segment_text = self.apply_lexicon(segment_text, lexicon)

                seg_config = config.copy()
                if speaker_name:
                    preset = self.load_preset(speaker_name)
                    if preset:
                        seg_config.update(preset)
                        if 'trim' in preset:
                            seg_config['trim_silence'] = preset['trim']
                        # Resolve voice path for the new voice
                        seg_config['voice'] = self.resolve_voice_path(seg_config['voice'])
                    else:
                        if self.on_status: self.on_status(f"Warning: Preset '{speaker_name}' not found.", False)

                if fx_name:
                    fx_preset = self.load_fx_preset(fx_name)
                    if fx_preset:
                        seg_config.update(fx_preset)
                        seg_config['apply_fx'] = True
                        seg_config['fx_preset'] = fx_name
                    else:
                        if self.on_status: self.on_status(f"Warning: FX Preset '{fx_name}' not found.", False)

                # Split this segment into sub-chunks for parallel processing
                # Use same character limit as original
                seg_chunks = self.smart_split(segment_text, chunk_size=5000 if num_workers > 1 else 1000000)
                for chunk in seg_chunks:
                    # (index, text, config)
                    tasks_data.append((len(tasks_data), chunk, seg_config))

            total_chunks = len(tasks_data)
            if total_chunks == 0:
                if self.on_status: self.on_status("No text to process.", False)
                if self.on_finish: self.on_finish()
                return

            total_chars = sum(len(d[1]) for d in tasks_data)
            processed_chars = 0
            start_time = time.time()
            phase_weight = 0.9 if config.get('combine', True) else 1.0

            if self.on_status: self.on_status(f"Queued {total_chunks} blocks. Starting {num_workers} workers...", False)

            # Progress tracker
            progress_lock = threading.Lock()

            def on_chunk_progress(char_count, snippet):
                nonlocal processed_chars
                with progress_lock:
                    processed_chars += char_count

                # Calculate progress and call main callback
                elapsed = time.time() - start_time
                gen_fraction = min(processed_chars / total_chars, 1.0)
                total_fraction = gen_fraction * phase_weight

                # Estimate ETA
                eta_str = "--:--"
                if total_fraction > 0.01:
                    total_est = elapsed / total_fraction
                    rem = max(0, total_est - elapsed)
                    eta_str = time.strftime('%M:%S', time.gmtime(rem))

                clean_snip = snippet.replace("\n", " ").strip()
                if len(clean_snip) > 40: clean_snip = clean_snip[:37] + "..."

                if self.on_progress:
                    self.on_progress(total_fraction * 100, elapsed, eta_str, f"Processing: {clean_snip}")

            # All generated files list
            all_generated_files = [None] * total_chunks

            loop = asyncio.get_running_loop()

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for i, data in enumerate(tasks_data):
                    fut = loop.run_in_executor(executor, self.process_chunk_task, data, on_chunk_progress)
                    futures.append(fut)

                results = await asyncio.gather(*futures, return_exceptions=True)

                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        print(f"Chunk {i} failed: {result}")
                        if self.on_status: self.on_status(f"Error in chunk {i}", True)
                    else:
                        all_generated_files[i] = result

            if self.cancel_event.is_set():
                if self.on_status: self.on_status("Conversion Cancelled.", False)
                if self.on_finish: self.on_finish()
                return

            final_segment_list = []
            for sublist in all_generated_files:
                if sublist: final_segment_list.extend(sublist)

            final_file_paths = [seg['path'] for seg in final_segment_list]

            if self.on_status: self.on_status(f"Generated {len(final_segment_list)} segments. Processing outputs...", False)

            if config.get('export_subtitles', False) and final_segment_list:
                srt_path = os.path.join(config['out_dir'], f"{config.get('filename', 'output')}_{config.get('time_id', '0')}_combined.srt")
                self.generate_srt(final_segment_list, srt_path)

            if config.get('combine', True) and final_file_paths:
                if self.on_status: self.on_status("Merging audio files...", False)

                fmt = config.get('format', 'wav').lower()
                combine_path = os.path.join(config['out_dir'], f"{config.get('filename', 'output')}_{config.get('time_id', '0')}_combined.{fmt}")

                def on_merge_progress(frac):
                    total_fraction = (1.0 * phase_weight) + (frac * (1.0 - phase_weight))
                    elapsed = time.time() - start_time
                    if self.on_progress:
                        self.on_progress(total_fraction * 100, elapsed, "00:00", f"Merging... {int(frac*100)}%")

                await self.smart_combine(final_file_paths, combine_path, on_merge_progress)

                if not config.get('separate', True):
                    for p in final_file_paths:
                        try: os.remove(p)
                        except Exception: pass

                if self.on_status: self.on_status(f"Done! Saved: {combine_path}", False)
            else:
                 if self.on_status: self.on_status("Conversion Complete!", False)

            if self.on_progress:
                self.on_progress(100, time.time() - start_time, "00:00", "Completed")

        except Exception as e:
            print(e)
            if self.on_status: self.on_status(f"Critical Error: {e}", True)
        finally:
            if self.on_finish: self.on_finish()
