"""Real-time ("JIT") generation and playback: a generation thread fills a queue
while a playback thread drains it, with buffer management for immediate streaming.

Calls `kokoro_engine.playback.play` qualified, at call time, so tests can keep
monkeypatching `kokoro_engine.playback` to a `MagicMock()` (see the `engine`
fixture in `tests/conftest.py`) without a real audio device ever being touched.
"""
import asyncio
import os
import time

import kokoro_engine


class JITMixin:
    def start_jit_conversion(self, text, config):
        """Starts real-time generation and playback."""
        config['voice'] = self.resolve_voice_path(config['voice'])
        self.cancel_event.clear()
        self.worker.run_coro(self._process_jit_async(text, config))

    async def _process_jit_async(self, text, config):
        """
        JIT Logic:
        1. Parse text into segments.
        2. Generation thread fills a queue.
        3. Playback thread consumes the queue.
        4. Buffer management (2 mins ahead).
        """
        try:
            if self.on_status: self.on_status("JIT: Preparing...", False)
            os.makedirs(config['out_dir'], exist_ok=True)

            # 1. Parse segments
            ms_segments = self.parse_multispeaker_text(text)
            all_text_segments = []
            lexicon = config.get('lexicon', {})

            for speaker_name, fx_name, segment_text in ms_segments:
                segment_text = self.apply_lexicon(segment_text, lexicon)
                seg_config = config.copy()
                seg_config['format'] = 'wav' # Force wav for JIT playback compatibility
                if speaker_name:
                    preset = self.load_preset(speaker_name)
                    if preset:
                        seg_config.update(preset)
                        if 'trim' in preset:
                            seg_config['trim_silence'] = preset['trim']
                        seg_config['format'] = 'wav' # Ensure preset doesn't override format to non-wav
                        seg_config['voice'] = self.resolve_voice_path(seg_config['voice'])

                if fx_name:
                    fx_preset = self.load_fx_preset(fx_name)
                    if fx_preset:
                        seg_config.update(fx_preset)
                        seg_config['apply_fx'] = True
                        seg_config['fx_preset'] = fx_name

                # Split into smaller chunks for JIT (sentences/short paragraphs)
                chunks = self.smart_split(segment_text, chunk_size=500) # Small chunks for fast start
                for c in chunks:
                    all_text_segments.append((c, seg_config))

            if not all_text_segments:
                if self.on_status: self.on_status("No text for JIT.", False)
                if self.on_finish: self.on_finish()
                return

            # Queues and State
            audio_queue = asyncio.Queue()
            played_segments = []
            generated_but_unplayed = []
            total_segments = len(all_text_segments)

            playback_finished_event = asyncio.Event()

            # --- Generation Loop ---
            async def generation_loop():
                nonlocal total_segments
                try:
                    for i, (seg_text, seg_config) in enumerate(all_text_segments):
                        if self.cancel_event.is_set(): break

                        while audio_queue.qsize() > 10 and not self.cancel_event.is_set():
                            await asyncio.sleep(0.5)

                        if self.cancel_event.is_set(): break

                        if self.on_status:
                            self.on_status(f"JIT: Generating chunk {i+1}/{total_segments}...", False)

                        chunk_files = await asyncio.to_thread(self.process_chunk_task, (i, seg_text, seg_config), None)

                        for cf in chunk_files:
                            await audio_queue.put(cf)
                            generated_but_unplayed.append(cf)
                except Exception as e:
                    print(f"JIT Gen Error: {e}")
                finally:
                    # Always signal end
                    await audio_queue.put(None)

            # --- Playback Loop ---
            async def playback_loop():
                nonlocal played_segments
                start_time = time.time()
                try:
                    idx = 0
                    while not self.cancel_event.is_set():
                        # Use wait_for to allow checking cancel_event periodically
                        try:
                            item = await asyncio.wait_for(audio_queue.get(), timeout=1.0)
                        except asyncio.TimeoutError:
                            continue

                        if item is None: break # End of stream

                        idx += 1
                        if self.on_status:
                            self.on_status(f"JIT: Playing chunk {idx}...", False)

                        clean_snip = item['text'].replace("\n", " ").strip()
                        if len(clean_snip) > 40: clean_snip = clean_snip[:37] + "..."

                        elapsed = time.time() - start_time
                        if self.on_progress:
                            percent = (idx / total_segments) * 100
                            self.on_progress(percent, elapsed, "--:--", f"Playing: {clean_snip}")

                        # Play audio (Synchronously in thread)
                        await asyncio.to_thread(kokoro_engine.playback.play, item['path'], True)

                        played_segments.append(item)
                        if item in generated_but_unplayed:
                            generated_but_unplayed.remove(item)

                except Exception as e:
                    print(f"JIT Playback Error: {e}")
                finally:
                    playback_finished_event.set()

            # Start loops
            gen_task = asyncio.create_task(generation_loop())
            play_task = asyncio.create_task(playback_loop())

            await playback_finished_event.wait()

            # --- Cleanup and Save State ---
            if self.cancel_event.is_set():
                if self.on_status: self.on_status("JIT Stopped. Saving state...", False)
            else:
                if self.on_status: self.on_status("JIT Finished.", False)

            # Combine what was played/generated so far
            all_work_so_far = played_segments + generated_but_unplayed
            if all_work_so_far:
                combined_path = os.path.join(config['out_dir'], f"{config.get('filename', 'output')}_{config.get('time_id', '0')}_jit_output.wav")
                await self.smart_combine([s['path'] for s in all_work_so_far], combined_path, None)
                if self.on_status: self.on_status(f"JIT Output saved: {combined_path}", False)

            # Save remaining text
            if generated_but_unplayed:
                first_remaining_idx = generated_but_unplayed[0]['seg_idx']
            elif played_segments:
                first_remaining_idx = played_segments[-1]['seg_idx'] + 1
            else:
                first_remaining_idx = 0

            remaining_text = ""
            for i in range(first_remaining_idx, total_segments):
                remaining_text += all_text_segments[i][0] + "\n\n"

            if remaining_text:
                rem_path = os.path.join(config['out_dir'], f"{config.get('filename', 'output')}_{config.get('time_id', '0')}_remaining.txt")
                with open(rem_path, "w", encoding="utf-8") as f:
                    f.write(remaining_text)
                if self.on_status: self.on_status(f"Remaining text saved: {rem_path}", False)

        except Exception as e:
            print(f"JIT Critical Error: {e}")
            if self.on_status: self.on_status(f"JIT Error: {e}", True)
        finally:
            if self.on_finish: self.on_finish()
