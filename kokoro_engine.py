import os
import threading
import asyncio
import time
import concurrent.futures
import soundfile as sf
import torch
import numpy as np
import scipy.signal
from pedalboard import (
    Pedalboard, Reverb, Compressor, HighShelfFilter, LowShelfFilter,
    Chorus, Distortion, Phaser, Clipping, Gain, Limiter,
    HighpassFilter, LowpassFilter, LadderFilter, Delay, PitchShift,
    GSMFullRateCompressor, Bitcrush
)
from pedalboard.io import AudioFile
import pypdf
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import warnings
import re
import json
from kokoro import KPipeline

# Suppress ebooklib warnings
warnings.filterwarnings("ignore", category=UserWarning, module='ebooklib')
warnings.filterwarnings("ignore", category=FutureWarning, module='ebooklib')

CUSTOM_VOICES_DIR = "custom_voices"

# --- Thread Local Storage ---
thread_local = threading.local()

def get_thread_pipeline(lang_code="a"):
    """Get or create a KPipeline instance for the current thread."""
    current = getattr(thread_local, "pipeline", None)
    if current is None or getattr(current, "lang_code", None) != lang_code:
        try:
            thread_local.pipeline = KPipeline(lang_code=lang_code)
        except Exception as e:
            print(f"Error init pipeline in thread {threading.get_ident()}: {e}")
            return None
    return thread_local.pipeline

class AsyncLoopThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.loop = asyncio.new_event_loop()
        self.running = True

    def run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def stop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.join()

    def run_coro(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

class KokoroEngine:
    def __init__(self):
        self.worker = AsyncLoopThread()
        self.worker.start()
        self.cancel_event = threading.Event()
        self.pipeline = None # Main pipeline for single thread check or init
        
        if not os.path.exists(CUSTOM_VOICES_DIR):
            os.makedirs(CUSTOM_VOICES_DIR)
        
        # Callbacks
        self.on_progress = None # func(percentage, time_elapsed, eta, detail_text)
        self.on_status = None   # func(msg, is_error)
        self.on_finish = None   # func()

    def apply_lexicon(self, text, lexicon):
        """
        Applies a dictionary of replacements to the text.
        Case-insensitive finding, preserves case of replacement.
        """
        if not lexicon:
            return text
        
        for src, dest in lexicon.items():
            if not src: continue
            try:
                # Escape the search term to treat it as literal text
                pattern = re.compile(re.escape(src), re.IGNORECASE)
                text = pattern.sub(dest, text)
            except Exception as e:
                print(f"Lexicon error for '{src}': {e}")
                
        return text

    def resolve_voice_path(self, voice_name):
        """
        Returns the absolute path if it's a custom voice, 
        otherwise returns the name as-is (for standard voices).
        """
        # Check if it's a custom voice file
        custom_path = os.path.join(CUSTOM_VOICES_DIR, f"{voice_name}.pt")
        if os.path.exists(custom_path):
            return os.path.abspath(custom_path)
        return voice_name

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

    async def init_pipeline_async(self, lang_code="a"):
        try:
            self.pipeline = await asyncio.to_thread(KPipeline, lang_code=lang_code)
            if self.on_status: self.on_status(f"Pipeline Initialized ({lang_code}).", False)
            return True
        except Exception as e:
            msg = f"Pipeline Init Failed: {e}"
            err_str = str(e).lower()
            if lang_code == 'j' and ("fugashi" in err_str or "unidic" in err_str):
                 msg += "\n(Try: pip install fugashi unidic-lite)"
            elif lang_code == 'z' and "pypinyin" in err_str:
                 msg += "\n(Try: pip install pypinyin)"
            
            if self.on_status: self.on_status(msg, True)
            return False

    async def mix_voices(self, v1_name, v2_name, ratio, new_name):
        def _mix():
            try:
                # Ensure we have a pipeline to load voices
                # Use 'a' as default for mixing if main pipeline is not ready
                p = self.pipeline
                if not p:
                    p = get_thread_pipeline('a')
                    if not p: raise RuntimeError("No pipeline available for mixing")
                
                # Resolve inputs (handle custom vs standard)
                v1_arg = self.resolve_voice_path(v1_name)
                v2_arg = self.resolve_voice_path(v2_name)
                
                # Load tensors
                # KPipeline.load_voice returns a tensor
                t1 = p.load_voice(v1_arg)
                t2 = p.load_voice(v2_arg)
                
                if t1 is None or t2 is None:
                    raise ValueError("Failed to load one of the voices.")
                
                # Ensure they are on CPU for mixing
                if isinstance(t1, torch.Tensor): t1 = t1.cpu()
                if isinstance(t2, torch.Tensor): t2 = t2.cpu()
                
                # Check shapes
                if t1.shape != t2.shape:
                    # Try to align? Usually kokoro voices are fixed size [510, 1, 256]
                    # If different, we might fail or warn.
                    print(f"Warning: Voice shapes differ {t1.shape} vs {t2.shape}. Mixing might fail or produce garbage.")
                
                # Linear Interpolation
                # mixed = v1 * (1 - ratio) + v2 * ratio
                # ratio is mix of B. If ratio 0, full A. If ratio 1, full B.
                mixed = t1 * (1.0 - ratio) + t2 * ratio
                
                # Save
                out_path = os.path.join(CUSTOM_VOICES_DIR, f"{new_name}.pt")
                torch.save(mixed, out_path)
                return True, out_path, mixed
            except Exception as e:
                return False, str(e), None

        return await asyncio.to_thread(_mix)

    async def generate_preview(self, text, voice, speed, output_path, extra_config=None, voice_tensor=None, lang_code='a'):
        def _gen():
            # Use specific lang code for preview
            p = get_thread_pipeline(lang_code)
            if not p: return False

            try:
                ms_segments = self.parse_multispeaker_text(text)
                # Truncate to first 2 segments for preview if many
                if len(ms_segments) > 2:
                    ms_segments = ms_segments[:2]
                
                all_pieces = []

                for speaker_name, segment_text in ms_segments:
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

    def extract_text_from_file(self, fpath):
        if not os.path.exists(fpath):
            raise FileNotFoundError("File does not exist.")
        
        text_data = ""
        lower_path = fpath.lower()
        
        if lower_path.endswith(".pdf"):
            reader = pypdf.PdfReader(fpath)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text_data += extracted + "\n\n"
        
        elif lower_path.endswith(".epub"):
            book = epub.read_epub(fpath)
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text_data += soup.get_text(separator='\n\n') + "\n\n"
        else:
            # Assume text based
            with open(fpath, "r", encoding="utf-8") as f:
                text_data = f.read()
                
        return text_data

    def parse_multispeaker_text(self, text):
        """
        Parses text for [PresetName]: syntax.
        Returns a list of (preset_name, text_segment)
        """
        # Regex to find [Name]:
        # Matches [Something]: followed by text until the next [Something]: or end of string
        pattern = r"\[([^\]]+)\]:\s*"
        matches = list(re.finditer(pattern, text))
        
        if not matches:
            return [(None, text)]
            
        segments = []
        for i in range(len(matches)):
            name = matches[i].group(1)
            start = matches[i].end()
            end = matches[i+1].start() if i+1 < len(matches) else len(text)
            segment_text = text[start:end].strip()
            if segment_text:
                segments.append((name, segment_text))
        
        return segments

    def load_preset(self, name):
        """Loads a preset from the presets directory."""
        preset_path = os.path.join("presets", f"{name}.json")
        if os.path.exists(preset_path):
            try:
                with open(preset_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading preset {name}: {e}")
        return None

    def smart_split(self, text, chunk_size=3000):
        chunks = []
        current_chunk = []
        current_len = 0
        paragraphs = text.split('\n\n')
        
        for para in paragraphs:
            if len(para) > chunk_size:
                lines = para.split('\n')
                for line in lines:
                    if current_len + len(line) > chunk_size and current_chunk:
                        chunks.append("\n".join(current_chunk))
                        current_chunk = []
                        current_len = 0
                    current_chunk.append(line)
                    current_len += len(line)
            else:
                if current_len + len(para) > chunk_size and current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                current_chunk.append(para)
                current_len += len(para)
        
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        return [c for c in chunks if c.strip()]

    def generate_srt(self, segments, output_path):
        def format_time(seconds):
            millis = int((seconds - int(seconds)) * 1000)
            seconds = int(seconds)
            minutes, seconds = divmod(seconds, 60)
            hours, minutes = divmod(minutes, 60)
            return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                current_time = 0.0
                for i, seg in enumerate(segments):
                    start = current_time
                    end = current_time + seg['duration']
                    f.write(f"{i+1}\n")
                    f.write(f"{format_time(start)} --> {format_time(end)}\n")
                    f.write(f"{seg['text'].strip()}\n\n")
                    current_time = end
            return True
        except Exception as e:
            print(f"Failed to generate SRT: {e}")
            return False

    def process_chunk_task(self, chunk_data, progress_callback):
        index, text, config = chunk_data
        if self.cancel_event.is_set(): return []

        # Use lang_code from config, default to 'a'
        lang_code = config.get('lang_code', 'a')
        pipeline = get_thread_pipeline(lang_code)
        
        if not pipeline: raise RuntimeError(f"Failed to initialize pipeline ({lang_code}) in thread.")

        # Speed Adjustment for Pitch Compensation
        eff_speed = config['speed']
        pitch_semitones = config.get('pitch', 0.0)
        if pitch_semitones != 0.0:
            factor = 2 ** (pitch_semitones / 12.0)
            eff_speed = eff_speed / factor

        # Config voice should already be resolved by start_conversion
        generator = pipeline(text, voice=config['voice'], speed=eff_speed, split_pattern=config['split_pattern'])
        chunk_files = []
        sub_idx = 0
        base_name = f"{config['filename']}_{config['time_id']}_part{index}"

        for graphemes, _, audio in generator:
            if self.cancel_event.is_set(): break
            
            # Notify progress (chars processed)
            if progress_callback:
                progress_callback(len(graphemes), graphemes)

            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
            
            # Post Process
            audio = self.process_audio(audio, 24000, config)
            
            # Determine format
            fmt = config.get('format', 'wav').lower()
            if fmt not in ['wav', 'flac', 'mp3', 'ogg']: fmt = 'wav'

            file_name = f"{base_name}_{sub_idx}.{fmt}"
            path = os.path.join(config['out_dir'], file_name)
            
            try:
                # Use Pedalboard AudioFile for writing
                with AudioFile(path, 'w', samplerate=24000, num_channels=1) as f:
                    f.write(audio)
            except Exception as e:
                print(f"Pedalboard write failed: {e}. Fallback to soundfile.")
                sf.write(path, audio, 24000)
            
            chunk_files.append({
                "path": path,
                "text": graphemes,
                "duration": len(audio) / 24000.0
            })
            sub_idx += 1
            
        return chunk_files

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

    def cancel(self):
        self.cancel_event.set()

    async def _process_text_async(self, text, config):
        try:
            if self.on_status: self.on_status("Preparing text...", False)
            os.makedirs(config['out_dir'], exist_ok=True)
            
            num_workers = config.get('num_threads', 1)
            
            # Multispeaker Support
            ms_segments = self.parse_multispeaker_text(text)
            tasks_data = []
            
            lexicon = config.get('lexicon', {})

            for speaker_name, segment_text in ms_segments:
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
            phase_weight = 0.9 if config['combine'] else 1.0
            
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

            if config['export_subtitles'] and final_segment_list:
                srt_path = os.path.join(config['out_dir'], f"{config['filename']}_{config['time_id']}_combined.srt")
                self.generate_srt(final_segment_list, srt_path)

            if config['combine'] and final_file_paths:
                if self.on_status: self.on_status("Merging audio files...", False)
                
                fmt = config.get('format', 'wav').lower()
                combine_path = os.path.join(config['out_dir'], f"{config['filename']}_{config['time_id']}_combined.{fmt}")
                
                def on_merge_progress(frac):
                    total_fraction = (1.0 * phase_weight) + (frac * (1.0 - phase_weight))
                    elapsed = time.time() - start_time
                    if self.on_progress:
                        self.on_progress(total_fraction * 100, elapsed, "00:00", f"Merging... {int(frac*100)}%")
                
                await self.smart_combine(final_file_paths, combine_path, on_merge_progress)
                
                if not config['separate']:
                    for p in final_file_paths:
                        try: os.remove(p)
                        except: pass
                
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
