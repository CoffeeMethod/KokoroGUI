import os
import threading
import asyncio
import time
import concurrent.futures
import soundfile as sf
import torch
import numpy as np
import scipy.signal
import pypdf
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import warnings
from kokoro import KPipeline

# Suppress ebooklib warnings
warnings.filterwarnings("ignore", category=UserWarning, module='ebooklib')
warnings.filterwarnings("ignore", category=FutureWarning, module='ebooklib')

# --- Thread Local Storage ---
thread_local = threading.local()

def get_thread_pipeline(lang_code="a"):
    """Get or create a KPipeline instance for the current thread."""
    if not hasattr(thread_local, "pipeline"):
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
        
        # Callbacks
        self.on_progress = None # func(percentage, time_elapsed, eta, detail_text)
        self.on_status = None   # func(msg, is_error)
        self.on_finish = None   # func()

    def process_audio(self, audio, sr, config):
        """
        Apply post-processing: Pitch (Resample), Volume, Normalize, Trim.
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
        # Note: We handled the duration compensation by adjusting the generation speed beforehand.
        # Here we just do the resampling to shift the pitch back (or forth).
        # Pitch > 0 means higher pitch.
        # If user wanted higher pitch, we generated SLOWER (longer).
        # Now we play it faster (shorter) to get higher pitch and normal length.
        pitch_semitones = config.get('pitch', 0.0)
        if pitch_semitones != 0.0:
            # Factor: >1 means higher frequency (shorter duration)
            # 2^(st/12)
            factor = 2 ** (pitch_semitones / 12.0)
            
            # Target length = Original / Factor
            new_len = int(len(audio) / factor)
            if new_len > 0:
                # Scipy resample uses Fourier method usually, or we can use signal.resample
                try:
                    audio = scipy.signal.resample(audio, new_len)
                except Exception as e:
                    print(f"Resample failed: {e}")

        # 4. Normalization
        if config.get('normalize', False):
            peak = np.max(np.abs(audio))
            if peak > 0:
                target_peak = 0.98
                audio = audio / peak * target_peak

        return audio

    async def init_pipeline_async(self):
        try:
            self.pipeline = await asyncio.to_thread(KPipeline, lang_code="a")
            if self.on_status: self.on_status("Pipeline Initialized and Ready.", False)
            return True
        except Exception as e:
            if self.on_status: self.on_status(f"Pipeline Init Failed: {e}", True)
            return False

    async def generate_preview(self, text, voice, speed, output_path, extra_config=None):
        def _gen():
            if not self.pipeline:
                p = get_thread_pipeline()
                if not p: return False
            else:
                p = self.pipeline

            try:
                # Pitch Compensation Logic
                # If pitch is +2 st (factor 1.12), we want higher pitch.
                # Resampling to 1/1.12 makes it higher pitch but shorter.
                # So generate it 1.12x longer (slower).
                # Speed_eff = Speed / Factor
                
                eff_speed = speed
                pitch_semitones = 0.0
                if extra_config:
                    pitch_semitones = extra_config.get('pitch', 0.0)
                    if pitch_semitones != 0.0:
                        factor = 2 ** (pitch_semitones / 12.0)
                        eff_speed = speed / factor

                # Generate
                generator = p(text, voice=voice, speed=eff_speed, split_pattern=r"\n+")
                pieces = []
                for _, _, audio in generator:
                    if isinstance(audio, torch.Tensor):
                        audio = audio.cpu().numpy()
                    pieces.append(audio)
                
                if not pieces:
                    return False
                
                full_audio = np.concatenate(pieces)
                
                # Post Process
                if extra_config:
                    full_audio = self.process_audio(full_audio, 24000, extra_config)
                
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

        pipeline = get_thread_pipeline()
        if not pipeline: raise RuntimeError("Failed to initialize pipeline in thread.")

        # Speed Adjustment for Pitch Compensation
        eff_speed = config['speed']
        pitch_semitones = config.get('pitch', 0.0)
        if pitch_semitones != 0.0:
            factor = 2 ** (pitch_semitones / 12.0)
            eff_speed = eff_speed / factor

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

            file_name = f"{base_name}_{sub_idx}.wav"
            path = os.path.join(config['out_dir'], file_name)
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
            with sf.SoundFile(output_path, 'w', samplerate=24000, channels=1) as out_f:
                for i, fp in enumerate(file_paths):
                    if self.cancel_event.is_set(): break
                    try:
                        data, _ = sf.read(fp)
                        out_f.write(data)
                        if update_callback: update_callback((i + 1) / total_files)
                    except Exception as e:
                        print(f"Failed to read/write segment {fp}: {e}")
        await asyncio.to_thread(combine_worker)

    def start_conversion(self, text, config):
        self.cancel_event.clear()
        self.worker.run_coro(self._process_text_async(text, config))

    def cancel(self):
        self.cancel_event.set()

    async def _process_text_async(self, text, config):
        try:
            if self.on_status: self.on_status("Preparing text...", False)
            os.makedirs(config['out_dir'], exist_ok=True)
            
            num_workers = config.get('num_threads', 1)
            chunks = self.smart_split(text, chunk_size=5000 if num_workers > 1 else 1000000)
            total_chunks = len(chunks)
            
            total_chars = sum(len(c) for c in chunks)
            processed_chars = 0
            start_time = time.time()
            phase_weight = 0.9 if config['combine'] else 1.0
            
            if self.on_status: self.on_status(f"Queued {len(chunks)} blocks. Starting {num_workers} workers...", False)
            
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

            # Prepare tasks
            tasks_data = [(i, c, config) for i, c in enumerate(chunks)]
            all_generated_files = [None] * total_chunks
            
            loop = asyncio.get_running_loop()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for i, data in enumerate(tasks_data):
                    # We pass a lambda that calls our tracker
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
                combine_path = os.path.join(config['out_dir'], f"{config['filename']}_{config['time_id']}_combined.wav")
                
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
