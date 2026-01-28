import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import asyncio
import soundfile as sf
import torch
import numpy as np
import time
from kokoro import KPipeline
import concurrent.futures
import pypdf
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import warnings
import re

# Suppress ebooklib warnings about future ignores
warnings.filterwarnings("ignore", category=UserWarning, module='ebooklib')
warnings.filterwarnings("ignore", category=FutureWarning, module='ebooklib')

# --- Thread Local Storage for Parallel Pipelines ---
thread_local = threading.local()

def get_thread_pipeline(lang_code="a"):
    """Get or create a KPipeline instance for the current thread."""
    if not hasattr(thread_local, "pipeline"):
        try:
            # We assume lang_code is constant 'a' (American English) for now
            thread_local.pipeline = KPipeline(lang_code=lang_code)
        except Exception as e:
            print(f"Error init pipeline in thread {threading.get_ident()}: {e}")
            return None
    return thread_local.pipeline

# --- Async Helper ---
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

# --- Main Application ---
class TTSApp:
    def __init__(self, master):
        self.master = master
        master.title("Kokoro TTS GUI")
        master.geometry("600x750")

        # Core Variables
        self.text_var = tk.StringVar()
        self.file_path_var = tk.StringVar()
        self.voice = tk.StringVar(value="af_heart")
        self.filename = tk.StringVar(value="output")
        self.output_directory = tk.StringVar(value="audio_output")
        self.separate_files = tk.BooleanVar(value=True)
        self.timecode_format = tk.StringVar(value="%Y%m%d%H%M%S")
        self.combine_post = tk.BooleanVar(value=True)
        self.num_threads = tk.IntVar(value=1)
        self.speed_var = tk.DoubleVar(value=1.0)
        self.split_pattern_var = tk.StringVar(value=r"\n+")
        self.pipeline = None # Main pipeline for single thread
        self.is_generating = False
        self.cancel_event = threading.Event()
        self.progress_lock = threading.Lock()
        self.total_chars = 0
        self.processed_chars = 0
        self.merge_progress = 0.0
        self.phase_weight = 1.0
        self.current_snippet = ""
        self.start_time = 0

        # Background Processing
        self.worker = AsyncLoopThread()
        self.worker.start()

        # UI Setup
        self.create_widgets()
        
        # Initialize Pipeline in background (Primary pipeline)
        self.status_label.config(text="Initializing primary pipeline...")
        self.worker.run_coro(self.init_pipeline_async())

    def create_widgets(self):
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 1. Input Source
        input_group = ttk.LabelFrame(main_frame, text="Input", padding="5")
        input_group.pack(fill=tk.X, pady=5)

        self.notebook = ttk.Notebook(input_group)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=5)

        # Tab 1: Direct Text
        self.text_frame = ttk.Frame(self.notebook, padding=5)
        self.notebook.add(self.text_frame, text="Direct Text")
        self.text_entry = tk.Text(self.text_frame, height=6, width=40)
        self.text_entry.pack(fill=tk.BOTH, expand=True)

        # Tab 2: File
        self.file_frame = ttk.Frame(self.notebook, padding=5)
        self.notebook.add(self.file_frame, text="Load File")
        
        file_row = ttk.Frame(self.file_frame)
        file_row.pack(fill=tk.X)
        ttk.Entry(file_row, textvariable=self.file_path_var, width=40).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(file_row, text="Browse...", command=self.browse_file).pack(side=tk.LEFT)
        ttk.Label(self.file_frame, text="Supports .txt, .pdf, and .epub files. Ideal for books.").pack(anchor="w", pady=5)

        # 2. Configuration
        config_group = ttk.LabelFrame(main_frame, text="Configuration", padding="5")
        config_group.pack(fill=tk.X, pady=5)

        # Voice
        voice_row = ttk.Frame(config_group)
        voice_row.pack(fill=tk.X, pady=2)
        ttk.Label(voice_row, text="Voice:", width=15).pack(side=tk.LEFT)
        self.voice_options = [
            "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", 
            "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", 
            "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir", 
            "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa"
        ]
        ttk.Combobox(voice_row, textvariable=self.voice, values=self.voice_options).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Output Dir
        dir_row = ttk.Frame(config_group)
        dir_row.pack(fill=tk.X, pady=2)
        ttk.Label(dir_row, text="Output Dir:", width=15).pack(side=tk.LEFT)
        ttk.Entry(dir_row, textvariable=self.output_directory).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(dir_row, text="Browse", command=self.browse_directory).pack(side=tk.LEFT)

        # Filename
        file_name_row = ttk.Frame(config_group)
        file_name_row.pack(fill=tk.X, pady=2)
        ttk.Label(file_name_row, text="Base Filename:", width=15).pack(side=tk.LEFT)
        ttk.Entry(file_name_row, textvariable=self.filename).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Audio Speed
        speed_row = ttk.Frame(config_group)
        speed_row.pack(fill=tk.X, pady=2)
        ttk.Label(speed_row, text="Audio Speed:", width=15).pack(side=tk.LEFT)
        self.speed_spin = ttk.Spinbox(speed_row, from_=0.5, to=2.0, increment=0.1, textvariable=self.speed_var, width=5)
        self.speed_spin.pack(side=tk.LEFT)

        # Split Pattern
        split_row = ttk.Frame(config_group)
        split_row.pack(fill=tk.X, pady=2)
        ttk.Label(split_row, text="Split By:", width=15).pack(side=tk.LEFT)
        
        self.split_map = {
            "Natural (Newlines)": r"\n+",
            "Paragraphs (Double Newline)": r"\n\n+",
            "Sentences (.!?)": r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s"
        }
        # Invert map for display
        self.split_display_map = {v: k for k, v in self.split_map.items()}
        
        self.split_combo = ttk.Combobox(split_row, values=list(self.split_map.keys()), state="readonly")
        self.split_combo.set("Natural (Newlines)")
        self.split_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.split_combo.bind("<<ComboboxSelected>>", lambda e: self.split_pattern_var.set(self.split_map[self.split_combo.get()]))

        # Options
        opts_row = ttk.Frame(config_group)
        opts_row.pack(fill=tk.X, pady=5)
        self.export_subtitles = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts_row, text="Keep Separate Chunks", variable=self.separate_files).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(opts_row, text="Combine into One File", variable=self.combine_post).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(opts_row, text="Export Subtitles (.srt)", variable=self.export_subtitles).pack(side=tk.LEFT, padx=10)

        # Multithreading Config
        thread_row = ttk.Frame(config_group)
        thread_row.pack(fill=tk.X, pady=10)
        ttk.Label(thread_row, text="Parallel Processes:").pack(side=tk.LEFT, padx=(0,5))
        self.thread_spin = ttk.Spinbox(thread_row, from_=1, to=8, textvariable=self.num_threads, width=5)
        self.thread_spin.pack(side=tk.LEFT)
        ttk.Label(thread_row, text="(Warning: High RAM usage)").pack(side=tk.LEFT, padx=5)

        # 3. Actions
        action_frame = ttk.Frame(main_frame, padding="5")
        action_frame.pack(fill=tk.BOTH, expand=True)

        # Progress Info
        info_frame = ttk.Frame(action_frame)
        info_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.time_label = ttk.Label(info_frame, text="Time: 00:00 / ETA: --:--")
        self.time_label.pack(side=tk.LEFT)
        
        self.percent_label = ttk.Label(info_frame, text="0%")
        self.percent_label.pack(side=tk.RIGHT)

        self.detail_label = ttk.Label(action_frame, text="...", font=("Consolas", 8), foreground="grey")
        self.detail_label.pack(fill=tk.X, pady=(0, 10))

        btn_row = ttk.Frame(action_frame)
        btn_row.pack(fill=tk.X)
        self.convert_button = ttk.Button(btn_row, text="Start Conversion", command=self.start_conversion)
        self.convert_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.cancel_button = ttk.Button(btn_row, text="Cancel", command=self.cancel_conversion, state=tk.DISABLED)
        self.cancel_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.status_label = ttk.Label(action_frame, text="Ready", wraplength=550)
        self.status_label.pack(pady=10)

    # --- Logic ---

    def on_progress_update(self, char_count, text_snippet):
        with self.progress_lock:
            self.processed_chars += char_count
            self.current_snippet = text_snippet
        
        self.master.after(0, self._update_progress_ui)

    def _update_progress_ui(self):
        if not self.is_generating or self.total_chars == 0: return

        # Throttle updates (max 20fps)
        now = time.time()
        if hasattr(self, '_last_ui_update') and (now - self._last_ui_update < 0.05):
            return
        self._last_ui_update = now

        elapsed = time.time() - self.start_time
        
        # Generation Progress (0.0 - 1.0)
        gen_fraction = min(self.processed_chars / self.total_chars, 1.0)
        
        # Total Weighted Progress
        # If phase_weight is 0.9, generation is 0-90%, merge is 90-100%
        total_fraction = (gen_fraction * self.phase_weight) + (self.merge_progress * (1.0 - self.phase_weight))
        
        percentage = total_fraction * 100
        # Cap visual percentage at 99.9% until explicit finish
        if percentage > 99.9: percentage = 99.9
        
        # ETA Calculation
        # We use total_fraction for ETA to represent total job time
        if total_fraction > 0.01:
            total_time_est = elapsed / total_fraction
            remaining = max(0, total_time_est - elapsed)
            eta_str = time.strftime('%M:%S', time.gmtime(remaining))
        else:
            eta_str = "--:--"
            
        elapsed_str = time.strftime('%M:%S', time.gmtime(elapsed))
        
        # Update UI
        self.percent_label.config(text=f"{int(percentage)}%")
        self.time_label.config(text=f"Time: {elapsed_str} / ETA: {eta_str}")
        
        # Truncate snippet
        clean_snip = self.current_snippet.replace("\n", " ").strip()
        if len(clean_snip) > 50: clean_snip = clean_snip[:47] + "..."
        if self.merge_progress > 0 and gen_fraction >= 0.99:
             self.detail_label.config(text=f"Merging... {int(self.merge_progress*100)}%")
        else:
             self.detail_label.config(text=f"Processing: {clean_snip}")

    def browse_directory(self):
        d = filedialog.askdirectory()
        if d: self.output_directory.set(d)

    def browse_file(self):
        file_types = [
            ("Supported Files", "*.txt *.pdf *.epub"),
            ("Text Files", "*.txt"),
            ("PDF Files", "*.pdf"),
            ("EPUB Files", "*.epub"),
            ("All Files", "*.*")
        ]
        f = filedialog.askopenfilename(filetypes=file_types)
        if f: self.file_path_var.set(f)

    def update_status(self, msg, is_error=False):
        color = "red" if is_error else "black"
        self.master.after(0, lambda: self.status_label.config(text=msg, foreground=color))

    def update_ui_state(self, generating):
        state = tk.DISABLED if generating else tk.NORMAL
        cancel_state = tk.NORMAL if generating else tk.DISABLED
        self.master.after(0, lambda: self._set_ui_state(state, cancel_state, generating))

    def _set_ui_state(self, main_state, cancel_state, generating):
        self.convert_button.config(state=main_state)
        self.cancel_button.config(state=cancel_state)

    def cancel_conversion(self):
        if self.is_generating:
            self.cancel_event.set()
            self.update_status("Cancelling... please wait for current chunks.")

    async def init_pipeline_async(self):
        try:
            self.pipeline = await asyncio.to_thread(KPipeline, lang_code="a")
            self.update_status("Pipeline Initialized and Ready.")
        except Exception as e:
            self.update_status(f"Pipeline Init Failed: {e}", True)

    def extract_pdf_text(self, fpath):
        text = ""
        try:
            reader = pypdf.PdfReader(fpath)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n\n"
        except Exception as e:
            raise Exception(f"PDF Error: {e}")
        return text

    def extract_epub_text(self, fpath):
        text = ""
        try:
            book = epub.read_epub(fpath)
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    # Use BeautifulSoup to strip HTML tags
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text += soup.get_text(separator='\n\n') + "\n\n"
        except Exception as e:
            raise Exception(f"EPUB Error: {e}")
        return text

    def smart_split(self, text, chunk_size=3000):
        """Splits text into chunks respecting paragraph boundaries."""
        chunks = []
        current_chunk = []
        current_len = 0
        
        # Split by double newline first to preserve paragraphs
        paragraphs = text.split('\n\n')
        
        for para in paragraphs:
            # If a single paragraph is huge, split it by single newline
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
            
        return [c for c in chunks if c.strip()] # Remove empty chunks

    def process_chunk_task(self, chunk_data):
        """Executed by a worker thread."""
        index, text, config = chunk_data
        
        if self.cancel_event.is_set():
            return []

        # Get thread-local pipeline
        pipeline = get_thread_pipeline()
        if not pipeline:
            raise RuntimeError("Failed to initialize pipeline in thread.")

        # Generate
        generator = pipeline(text, voice=config['voice'], speed=config['speed'], split_pattern=config['split_pattern'])
        
        chunk_files = []
        sub_idx = 0
        
        base_name = f"{config['filename']}_{config['time_id']}_part{index}"
        
        for graphemes, _, audio in generator:
            if self.cancel_event.is_set(): break
            
            # Report progress (1 segment completed)
            self.on_progress_update(len(graphemes), graphemes)

            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
                
            file_name = f"{base_name}_{sub_idx}.wav"
            path = os.path.join(config['out_dir'], file_name)
            
            sf.write(path, audio, 24000)
            
            # Calculate duration in seconds (24000 Hz)
            duration = len(audio) / 24000.0
            
            chunk_files.append({
                "path": path,
                "text": graphemes,
                "duration": duration
            })
            sub_idx += 1
            
        return chunk_files

    def start_conversion(self):
        if self.is_generating: return
        
        # Gather inputs
        mode = self.notebook.index(self.notebook.select())
        text_data = ""
        
        if mode == 0: # Direct Text
            text_data = self.text_entry.get("1.0", tk.END).strip()
            if not text_data:
                messagebox.showerror("Error", "Please enter text.")
                return
        else: # File
            fpath = self.file_path_var.get()
            if not os.path.exists(fpath):
                messagebox.showerror("Error", "File does not exist.")
                return
            try:
                if fpath.lower().endswith(".pdf"):
                    text_data = self.extract_pdf_text(fpath)
                elif fpath.lower().endswith(".epub"):
                    text_data = self.extract_epub_text(fpath)
                else:
                    with open(fpath, "r", encoding="utf-8") as f:
                        text_data = f.read()
            except Exception as e:
                messagebox.showerror("Error", f"Could not read file: {e}")
                return

        if not text_data: return
        
        # Check pipeline if 1 thread
        if self.num_threads.get() == 1 and not self.pipeline:
             messagebox.showwarning("Wait", "Pipeline is still initializing...")
             return

        self.is_generating = True
        self.cancel_event.clear()
        self.update_ui_state(True)
        
        # Config
        config = {
            'voice': self.voice.get(),
            'speed': self.speed_var.get(),
            'split_pattern': self.split_pattern_var.get(),
            'filename': self.filename.get(),
            'out_dir': self.output_directory.get(),
            'separate': self.separate_files.get(),
            'combine': self.combine_post.get(),
            'export_subtitles': self.export_subtitles.get(),
            'timecode': self.timecode_format.get(),
            'time_id': time.strftime(self.timecode_format.get())
        }
        
        self.worker.run_coro(self.process_text_async(text_data, config))

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

    async def process_text_async(self, text, config):
        try:
            self.update_status("Preparing text...")
            os.makedirs(config['out_dir'], exist_ok=True)
            
            num_workers = self.num_threads.get()
            
            # 1. Split text
            chunks = self.smart_split(text, chunk_size=5000 if num_workers > 1 else 1000000)
            total_chunks = len(chunks)
            
            # Init Progress
            self.total_chars = sum(len(c) for c in chunks)
            self.processed_chars = 0
            self.merge_progress = 0.0
            self.phase_weight = 0.9 if config['combine'] else 1.0
            self.start_time = time.time()
            
            self.update_status(f"Queued {len(chunks)} blocks. Starting {num_workers} workers...")
            
            # Prepare tasks
            tasks_data = [(i, c, config) for i, c in enumerate(chunks)]
            all_generated_files = [None] * total_chunks # Placeholder for order
            
            # 2. Run Parallel
            loop = asyncio.get_running_loop()
            
            # Use ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                # Map futures to indices to reconstruct order later
                future_to_index = {}
                
                for i, data in enumerate(tasks_data):
                    fut = loop.run_in_executor(executor, self.process_chunk_task, data)
                    futures.append(fut)
                    future_to_index[fut] = i
                
                # Wait for all to complete
                # We use asyncio.gather to get them in order of submission (which is what we want)
                results = await asyncio.gather(*futures, return_exceptions=True)
                
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        print(f"Chunk {i} failed: {result}")
                        self.update_status(f"Error in chunk {i}", True)
                    else:
                        all_generated_files[i] = result

            if self.cancel_event.is_set():
                self.update_status("Conversion Cancelled.")
                return

            # Flatten file list in order
            final_segment_list = []
            for sublist in all_generated_files:
                if sublist: final_segment_list.extend(sublist)
            
            final_file_paths = [seg['path'] for seg in final_segment_list]
            
            self.update_status(f"Generated {len(final_segment_list)} segments. Combining...")

            # Export Subtitles
            if config['export_subtitles'] and final_segment_list:
                srt_path = os.path.join(config['out_dir'], f"{config['filename']}_{config['time_id']}_combined.srt")
                self.generate_srt(final_segment_list, srt_path)

            # Combine if needed
            if config['combine'] and final_file_paths:
                combine_path = os.path.join(config['out_dir'], f"{config['filename']}_{config['time_id']}_combined.wav")
                await self.smart_combine(final_file_paths, combine_path)
                
                if not config['separate']:
                    for p in final_file_paths:
                        try: os.remove(p)
                        except: pass
                
                self.update_status(f"Done! Saved: {combine_path}")
            else:
                 self.update_status("Conversion Complete!")

            # Force 100% on completion
            self.master.after(0, lambda: self.percent_label.config(text="100%"))
            self.master.after(0, lambda: self.time_label.config(text=f"Time: {time.strftime('%M:%S', time.gmtime(time.time() - self.start_time))} / ETA: 00:00"))

        except Exception as e:
            print(e)
            self.update_status(f"Error: {e}", True)
        finally:
            self.is_generating = False
            self.update_ui_state(False)

    async def smart_combine(self, file_paths, output_path):
        def combine_worker():
            total_files = len(file_paths)
            with sf.SoundFile(output_path, 'w', samplerate=24000, channels=1) as out_f:
                for i, fp in enumerate(file_paths):
                    if self.cancel_event.is_set(): break
                    try:
                        data, _ = sf.read(fp)
                        out_f.write(data)
                        
                        # Update Merge Progress
                        fraction = (i + 1) / total_files
                        self.merge_progress = fraction
                        self.master.after(0, self._update_progress_ui)
                        
                    except Exception as e:
                        print(f"Failed to read/write segment {fp}: {e}")
        
        await asyncio.to_thread(combine_worker)


if __name__ == "__main__":
    root = tk.Tk()
    app = TTSApp(root)
    root.mainloop()
