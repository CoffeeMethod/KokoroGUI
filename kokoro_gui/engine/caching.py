"""Per-chunk generation with WAV segment caching, keyed on
schema_version|engine_id|engine_version|text|voice|voice_fingerprint|speed|lang_code
(see `compute_cache_key` - PLAN_qt_and_engine_abstraction.md workstream 2).

Reads `kokoro_engine.CACHE_DIR`/`kokoro_engine.CUSTOM_VOICES_DIR` qualified, at
call time, so tests can keep monkeypatching those names on the
`kokoro_engine` module (e.g. the `isolated_dirs`/`make_config` fixtures and
the `_boom` sentinel used in `test_caching.py`/`test_mix_voices.py`). The
actual synthesis call goes through `self.get_thread_pipeline(lang_code)`
rather than `kokoro_engine.get_thread_pipeline` directly - that's the one
genuinely model-specific piece of this otherwise-generic pipeline, and going
through `self` lets a non-Kokoro backend (kokoro_gui/engines/dummy.py) reuse
this whole mixin by supplying its own `get_thread_pipeline`.
"""
import hashlib
import importlib.metadata
import os
import re

import soundfile as sf
import torch
from pedalboard.io import AudioFile

import kokoro_engine
from kokoro_gui.engine.audio_fx import clamp_pitch_semitones

# Bump whenever compute_cache_key's composition or logic changes. Old cache
# entries simply stop matching (new hash algorithm -> new filenames) and
# become dead weight for whatever eventually implements cache eviction
# (ROADMAP Phase 2) - no explicit migration/cleanup needed, but a bump does
# mean the first run after upgrading regenerates the whole cache.
CACHE_SCHEMA_VERSION = 2

# path -> (mtime, fingerprint): avoids re-hashing the same custom-voice file
# on every chunk in a batch run. Mirrors the `self._lexicon_cache` compiled-
# regex cache pattern (the lexicon perf fix) but keyed on filesystem content
# rather than an engine instance, since voice files are process-wide state.
_voice_fingerprint_cache = {}


def get_engine_version(engine_id="kokoro"):
    """Best-effort version/identity string for `engine_id`, folded into the
    cache key so an upgrade that changes model output invalidates stale
    entries instead of silently serving old audio under it. Only "kokoro"
    has a concrete answer today (the installed `kokoro` package version) -
    any other engine_id (e.g. a future cloud backend, or "dummy") falls back
    to a constant so its cache entries are at least self-consistent."""
    if engine_id == "kokoro":
        try:
            return importlib.metadata.version("kokoro")
        except importlib.metadata.PackageNotFoundError:
            return "unknown"
    return "unknown"


def voice_fingerprint(voice_ref):
    """Identity string for `voice_ref` (already resolved by
    `resolve_voice_path` - a bare name for a standard voice, or an absolute
    path for a custom one).

    Standard voices are fingerprinted by name alone - they're built into the
    model and don't change. Custom voices are fingerprinted by *content*:
    remixing and re-saving a `.pt` file under the same name (a real
    workflow - see `VoiceMixingMixin.mix_voices`) changes what the voice
    sounds like without changing its name, and a name-only key can't tell
    the difference. The content hash is cached per-file-mtime so a batch run
    doesn't re-read/re-hash the same file for every chunk.
    """
    if not voice_ref or not (os.path.isabs(voice_ref) and os.path.isfile(voice_ref)):
        return voice_ref

    try:
        mtime = os.path.getmtime(voice_ref)
    except OSError:
        return voice_ref

    cached = _voice_fingerprint_cache.get(voice_ref)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    try:
        with open(voice_ref, "rb") as f:
            fp = hashlib.sha256(f.read()).hexdigest()[:16]
    except OSError:
        return voice_ref

    _voice_fingerprint_cache[voice_ref] = (mtime, fp)
    return fp


def compute_cache_key(text, voice, eff_speed, lang_code, engine_id="kokoro", engine_version=None, extra=None):
    """The segment-cache hash: schema_version, engine identity/version, text,
    voice (name + content fingerprint), effective speed, language code, and
    an optional `extra` dict of engine-specific inputs that also affect what
    gets generated.

    Takes exactly those inputs, not a whole config dict - a config dict
    also carries `out_dir`/`filename`/`format`/`normalize`/`trim_silence`/the
    FX chain/`num_threads`/etc., none of which affect what gets cached (they
    apply in `process_and_save` *after* cache read/generation, to the same
    raw segment - that's the whole point of caching pre-FX audio). Keeping
    those out of the signature, not just out of the hash, makes that
    boundary the type checker/reader can see rather than something you have
    to trust the implementation not to violate. `split_pattern` is excluded
    for the same reason: only the text used to generate a segment determines
    its content - splitting is an internal detail of how a chunk gets
    divided for parallel processing.

    `extra` exists for a backend whose "voice" isn't fully described by a
    name + resolved-file fingerprint alone - e.g. Audio8Engine's zero-shot
    voice cloning also takes a reference *transcript*, which changes what
    gets generated even when the reference wav and its name are unchanged.
    Left as `None` (the default), it's omitted from `cache_key_parts`
    entirely rather than hashed as an empty/`None` value, so Kokoro's and
    the dummy backend's existing call sites - and every cache key they've
    already written to disk - are byte-for-byte unaffected by this
    parameter's addition.
    """
    if engine_version is None:
        engine_version = get_engine_version(engine_id)

    cache_key_parts = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "engine_id": engine_id,
        "engine_version": engine_version,
        "text": text,
        "voice": voice,
        "voice_fingerprint": voice_fingerprint(voice),
        "speed": eff_speed,
        "lang_code": lang_code,
    }
    if extra:
        for k in sorted(extra):
            cache_key_parts[f"extra_{k}"] = extra[k]
    to_hash = "|".join(f"{k}={v}" for k, v in cache_key_parts.items())
    return hashlib.sha256(to_hash.encode("utf-8")).hexdigest()


class CachingMixin:
    def process_chunk_task(self, chunk_data, progress_callback):
        index, text, config = chunk_data
        if self.cancel_event.is_set(): return []

        # Use lang_code from config, default to 'a'
        lang_code = config.get('lang_code', 'a')

        # Speed Adjustment for Pitch Compensation
        eff_speed = config['speed']
        pitch_semitones = clamp_pitch_semitones(config.get('pitch', 0.0))
        if pitch_semitones != 0.0:
            factor = 2 ** (pitch_semitones / 12.0)
            eff_speed = eff_speed / factor

        # --- Caching Check (WAV only) ---
        use_cache = config.get('caching', True)
        cache_hash = None
        cached_segments = []

        if use_cache:
            engine_id = config.get('engine_id', 'kokoro')
            cache_hash = compute_cache_key(text, config['voice'], eff_speed, lang_code, engine_id)

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

        # config['raw_output'] (set by generate_clip_audio for the clip
        # paths) skips process_audio: the segment file is the raw model
        # output and kokoro_gui/audio/post.py applies FX/volume/pitch/
        # normalize/trim at read time, so changing them never regenerates.
        raw_output = bool(config.get('raw_output', False))

        # Function to process raw audio (from cache or gen) into final output
        def process_and_save(graphemes, raw_audio):
            nonlocal sub_idx

            # Post Process
            if raw_output:
                processed_audio = raw_audio
            else:
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
                "seg_idx": index,
                "raw": raw_output,
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
            pipeline = self.get_thread_pipeline(lang_code)
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
