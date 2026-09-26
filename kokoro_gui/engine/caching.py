"""Per-chunk generation with segment caching, keyed on
schema_version|engine_id|engine_version|text|voice|voice_fingerprint|speed|lang_code
plus a backend's own `extra` inputs (see `compute_cache_key` and `segment_key`).

`segment_key` is the one hash for a segment: what `Segment.cache_key`
stores, what the dirty check (kokoro_gui/daw/dirty.py) recomputes, and the
stem of the file name in a project dir. Before it existed the dirty check
and `process_chunk_task` computed their keys through two code paths that
disagreed for custom voices (one hashed the name, the other the resolved
absolute path) and for Audio8 (only one folded in the transcript). Every
backend's `process_chunk_task` calls `segment_key(text, config, self)`; the
app calls it with its backend adapter. `backend` is duck-typed: anything
with `resolve_voice_file(name, project_dir)`, `cache_key_extra(config)` and
`engine_version()`, which `CachingMixin` supplies with defaults.

Two file-naming modes, chosen by `config["segment_naming"]`:

- unset (legacy, the whole-document and JIT paths): the cache is
  `runtime.CACHE_DIR` and the output file in `out_dir` is named
  `<filename>_<time_id>_part<index>_<sub_idx>.<fmt>`, post-processed unless
  `config["raw_output"]`.
- `"cache_key"` (every clip generation, `.tbaw` plan section 3): `out_dir`
  *is* the cache. The file is `<segment_key>_<sub_idx>.<fmt>`, written raw,
  once; `CACHE_DIR` is never touched; a present file is never overwritten
  (grill TB8). The target is reserved with a `<key>.reserved` marker made
  `O_CREAT | O_EXCL`, so two identical clips generating in one batch can't
  both write the same file; when the reservation fails, or the caller asks
  for `config["regenerate"]`, the clip's take index bumps and the key
  changes. Every result dict reports the `take`, `cache_key` and
  `engine_version` it landed on, so the caller stamps segments from the
  result instead of predicting the key before dispatch.

Reads `runtime.CACHE_DIR` qualified, at call time, so tests can
monkeypatch it (the `isolated_dirs` fixture). The synthesis call is
`self._synthesize(piece, config)` (kokoro_gui/engine/runner.py's
`EngineRunner`, around the backend's `SynthesisModel`), the one
model-specific piece of this otherwise-generic pipeline.
"""
import hashlib
import os
import re

import numpy as np
import soundfile as sf
from pedalboard.io import AudioFile

from kokoro_gui.engine import runtime
from kokoro_gui.engine.audio_fx import clamp_pitch_semitones
from kokoro_gui.engine import segmenting
from kokoro_gui.engine.wordtiming import silence_bounds
from kokoro_gui.engines.registry import DEFAULT_ENGINE_ID

# Bump whenever compute_cache_key's composition or logic changes. Old cache
# entries simply stop matching (new hash algorithm -> new filenames) and
# become dead weight for whatever eventually implements cache eviction
# (ROADMAP) - a bump means the first run after upgrading regenerates the
# whole cache. 3: the voice enters as basename + content fingerprint instead
# of the resolved path, so a key is the same on every machine (the `.tbaw`
# bundle names files by it). `.json` project migration rekeys with
# `schema_version=2` to adopt segments stamped under the old key.
CACHE_SCHEMA_VERSION = 3

# Marker suffix for a reserved segment key in "cache_key" naming mode.
RESERVED_SUFFIX = ".reserved"

# path -> (mtime, fingerprint): avoids re-hashing the same custom-voice file
# on every chunk in a batch run. Mirrors the `self._lexicon_cache` compiled-
# regex cache pattern (the lexicon perf fix) but keyed on filesystem content
# rather than an engine instance, since voice files are process-wide state.
_voice_fingerprint_cache = {}

AUDIO_FORMATS = ("wav", "flac", "mp3", "ogg")


def get_engine_version(engine_id=DEFAULT_ENGINE_ID):
    """Best-effort version/identity string for `engine_id`, folded into the
    cache key so an upgrade that changes model output invalidates stale
    entries instead of silently serving old audio under it. Asks the
    registered adapter class (`package_version()`, Kokoro's installed
    package version); an engine without one, or one that isn't installed,
    gets a constant so its entries are at least self-consistent. The
    default `engine_version()` hook on every backend calls this; Audio8
    answers with its model id instead."""
    from kokoro_gui.engines import registry

    return registry.package_version(engine_id) or "unknown"


def voice_fingerprint(voice_ref):
    """Identity string for `voice_ref`: a bare name for a standard voice
    (built into the model, never changes), or a content hash for an existing
    absolute file path (a custom `.pt` or a reference wav). Remixing and
    re-saving a `.pt` under the same name changes what the voice sounds like
    without changing its name, and a name-only key can't tell the
    difference. The content hash is cached per-file-mtime so a batch run
    doesn't re-read the same file for every chunk, and so the dirty check's
    per-rehighlight cost is a stat, not a read."""
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


def compute_cache_key(text, voice, eff_speed, lang_code, engine_id=DEFAULT_ENGINE_ID, engine_version=None, extra=None,
                      schema_version=None, voice_fingerprint_value=None):
    """The segment-cache hash: schema_version, engine identity/version, text,
    voice name, voice content fingerprint, effective speed, language code,
    and an optional `extra` dict of engine-specific inputs that also affect
    what gets generated.

    `voice` is the voice's *name* (a built-in id, or a custom file's
    basename without extension), never a path: a path would make the key
    differ per machine. `voice_fingerprint_value` is the file's content hash
    for a custom voice; left `None` it's derived with `voice_fingerprint(voice)`,
    which is the bare name for a built-in. `segment_key` below is the
    normal way in; this function stays a pure formula so its tests can pin
    the format.

    Takes exactly those inputs, not a whole config dict - a config dict
    also carries `out_dir`/`filename`/`format`/`normalize`/`trim_silence`/the
    FX chain/`num_threads`/etc., none of which affect what gets cached (they
    apply after cache read/generation, to the same raw segment - that's the
    whole point of caching pre-FX audio). The segmentation settings are
    excluded for the same reason: they decide which text a segment is,
    and only that text determines its content.

    `extra` exists for a backend whose "voice" isn't fully described by a
    name + fingerprint - Audio8's zero-shot cloning also takes a reference
    *transcript* and sampling knobs. Left `None` it's omitted from the
    parts entirely rather than hashed as empty, so a Kokoro key with no
    extra is unaffected by the parameter. `schema_version` defaults to
    `CACHE_SCHEMA_VERSION`; `.json` project migration passes the old one to
    recognise segments stamped before the bump.
    """
    if engine_version is None:
        engine_version = get_engine_version(engine_id)
    if schema_version is None:
        schema_version = CACHE_SCHEMA_VERSION
    if voice_fingerprint_value is None:
        voice_fingerprint_value = voice_fingerprint(voice)

    cache_key_parts = {
        "schema_version": schema_version,
        "engine_id": engine_id,
        "engine_version": engine_version,
        "text": text,
        "voice": voice,
        "voice_fingerprint": voice_fingerprint_value,
        "speed": eff_speed,
        "lang_code": lang_code,
    }
    if extra:
        for k in sorted(extra):
            cache_key_parts[f"extra_{k}"] = extra[k]
    to_hash = "|".join(f"{k}={v}" for k, v in cache_key_parts.items())
    return hashlib.sha256(to_hash.encode("utf-8")).hexdigest()


def effective_speed(config):
    """Speed adjusted for pitch compensation: a pitch shift is done by
    resampling, so the model is asked for a correspondingly slower or faster
    take. The one place this formula lives; the dirty check imports it."""
    eff_speed = config.get("speed", 1.0)
    pitch_semitones = clamp_pitch_semitones(config.get("pitch", 0.0))
    if pitch_semitones != 0.0:
        factor = 2 ** (pitch_semitones / 12.0)
        eff_speed = eff_speed / factor
    return eff_speed


def split_segments(text, config):
    """The pieces `text` is generated as: one pipeline call and one file
    each, on every path (clips, whole document, JIT). `process_chunk_task`
    calls the engine once per piece with the pipeline's own splitting off,
    and the dirty check compares stored segment texts against this list,
    so the two can't disagree. The rule (a word target, ranked boundary
    toggles, a hard limit at 2x) is in kokoro_gui/engine/segmenting.py."""
    return segmenting.split_text(text, config)


def normalize_voice(voice, backend, project_dir=None):
    """`(name, fingerprint)` for `config["voice"]`, which may be a voice
    name or a path an earlier step already resolved. A path that exists is
    taken as the voice file; a name is looked up through
    `backend.resolve_voice_file(name, project_dir)`. The name in the key is
    the basename without extension either way, and the fingerprint is the
    file's content hash, or the bare name for a built-in voice with no file."""
    if not voice:
        return voice, voice
    if os.path.isabs(voice) and os.path.isfile(voice):
        name = os.path.splitext(os.path.basename(voice))[0]
        return name, voice_fingerprint(voice)
    name = os.path.basename(voice)
    path = backend.resolve_voice_file(name, project_dir)
    if path:
        return os.path.splitext(os.path.basename(path))[0], voice_fingerprint(path)
    return name, name


def segment_key(text, config, backend, engine_version=None):
    """The one hash for a segment: what `Segment.cache_key` stores and what
    the file is named in a project dir. `config["voice"]` may be a name or
    an already resolved path; both normalize to (basename without
    extension, content fingerprint) through `normalize_voice`.
    `backend.cache_key_extra(config)` adds the backend's own generation
    inputs (Audio8: transcript + sampling). `config.get("take", 0)` enters
    when non-zero, so nothing keyed before takes existed changes.
    `engine_version` defaults to `backend.engine_version()`.
    `config["lang_code"]` is required, not defaulted: the dirty check and
    the engine used to default it differently ("a" vs "English") and one
    key can't have two defaults. Pure over its inputs plus the voice file's
    content."""
    if "lang_code" not in config:
        raise KeyError("segment_key needs config['lang_code']; the config assembler must set it")
    name, fingerprint = normalize_voice(config.get("voice"), backend, config.get("project_dir"))
    extra = dict(backend.cache_key_extra(config) or {})
    take = int(config.get("take", 0) or 0)
    if take:
        extra["take"] = take
    if engine_version is None:
        engine_version = backend.engine_version()
    engine_id = config.get("engine_id") or getattr(backend, "id", DEFAULT_ENGINE_ID)
    return compute_cache_key(
        text, name, effective_speed(config), config["lang_code"], engine_id,
        engine_version=engine_version, extra=extra or None, voice_fingerprint_value=fingerprint,
    )


def to_numpy(audio):
    """A model's audio as a numpy array: a torch tensor (anything with
    `detach`) is moved to the CPU first, without importing torch here."""
    if hasattr(audio, "detach"):
        return audio.detach().cpu().numpy()
    return audio


def _output_format(config):
    fmt = str(config.get("format", "wav")).lower()
    return fmt if fmt in AUDIO_FORMATS else "wav"


def _write_audio(path, audio, sample_rate, label):
    try:
        with AudioFile(path, "w", samplerate=sample_rate, num_channels=1) as f:
            f.write(audio)
    except Exception as e:
        print(f"{label} write failed: {e}. Fallback to soundfile.")
        sf.write(path, audio, sample_rate)


def _audio_duration_s(path, sample_rate):
    try:
        return float(sf.info(path).duration)
    except Exception:
        try:
            data, sr = sf.read(path)
            return len(data) / float(sr or sample_rate)
        except Exception:
            return 0.0


class CachingMixin:
    """Generation-with-cache for any engine that supplies
    `_synthesize(piece, config) -> Synthesis` (`EngineRunner`) and
    optionally `SAMPLE_RATE` (default 24000) and `id`. Also the default implementation of the three `segment_key`
    hooks; a backend overrides what differs (Audio8: `engine_version` and
    `cache_key_extra`)."""

    id = DEFAULT_ENGINE_ID

    # -- segment_key hooks --------------------------------------------------

    def engine_version(self):
        """What goes into the segment key and `manifest.engines[id].version`.
        Looked up through this module's `get_engine_version` by name so a
        test can monkeypatch it."""
        return get_engine_version(getattr(self, "id", DEFAULT_ENGINE_ID))

    def cache_key_extra(self, config):
        """Backend-specific generation inputs folded into `segment_key`.
        Nothing for Kokoro: a named or mixed voice, the text and the speed
        describe the output."""
        return {}

    def resolve_voice_file(self, name, project_dir=None):
        """The file a voice name resolves to (project dir first), or `None`
        for a built-in voice. Built on the backend's own `resolve_voice_path`."""
        resolved = self.resolve_voice_path(name, project_dir)
        if resolved and os.path.isabs(resolved) and os.path.isfile(resolved):
            return resolved
        return None

    # -- generation ---------------------------------------------------------

    def process_chunk_task(self, chunk_data, progress_callback):
        index, text, config = chunk_data
        if self.cancel_event.is_set():
            return []

        sample_rate = getattr(self, "SAMPLE_RATE", 24000)
        lang_code = config.get("lang_code")
        eff_speed = effective_speed(config)
        key_naming = config.get("segment_naming") == "cache_key"
        use_cache = key_naming or bool(config.get("caching", True))
        # config['raw_output'] (set by generate_clip_audio for the clip
        # paths) skips process_audio: the segment file is the raw model
        # output and kokoro_gui/audio/post.py applies FX/volume/pitch/
        # normalize/trim at read time. Cache-key naming is raw by definition:
        # the file it writes is the cache entry.
        raw_output = key_naming or bool(config.get("raw_output", False))
        fmt = _output_format(config)
        predicted_texts = split_segments(text, config)
        take = int(config.get("take", 0) or 0)
        engine_version = self.engine_version()

        cache_hash = None
        cache_dir = None
        cache_ext = "wav"
        cached_segments = []  # [(graphemes, audio), ...]
        hit_paths = None

        if key_naming:
            cache_dir = config["out_dir"]
            cache_ext = fmt
            regenerate = bool(config.get("regenerate", False))
            if not predicted_texts:
                return []
            while True:
                cache_hash = segment_key(text, {**config, "take": take}, self, engine_version)
                expected = [os.path.join(cache_dir, f"{cache_hash}_{i}.{cache_ext}") for i in range(len(predicted_texts))]
                marker = os.path.join(cache_dir, f"{cache_hash}{RESERVED_SUFFIX}")
                if not os.path.exists(marker) and all(os.path.isfile(p) for p in expected):
                    if not regenerate:
                        hit_paths = expected
                        break
                    # A present set is never written over (another clip may
                    # play it): a regenerate lands on the next take.
                    take += 1
                    continue
                try:
                    fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.close(fd)
                except FileExistsError:
                    take += 1
                    continue
                # Reserved. A partial earlier attempt (some files present, the
                # marker gone) also lands here through the `all(...)` check
                # failing: those files get overwritten under this same key,
                # which is what a resume of the same inputs should do.
                break
        elif use_cache:
            cache_dir = runtime.CACHE_DIR
            cache_hash = segment_key(text, config, self, engine_version)
            try:
                if predicted_texts:
                    loaded = []
                    all_exist = True
                    for i, seg_text in enumerate(predicted_texts):
                        f_path = os.path.join(cache_dir, f"{cache_hash}_{i}.wav")
                        if not os.path.exists(f_path):
                            all_exist = False
                            break
                        audio_data, _ = sf.read(f_path)
                        loaded.append((seg_text, audio_data))
                    if all_exist and loaded:
                        cached_segments = loaded
            except Exception as e:
                print(f"Cache check error: {e}")
                cached_segments = []

        def result(path, graphemes, duration, raw_audio=None, words=None):
            # Onset/tail come from the raw model output (what the post stage
            # trims); a cache hit re-reads the file for them. `words` are
            # relative to this segment's start.
            if raw_audio is None:
                try:
                    raw_audio, _sr = sf.read(path, dtype="float32")
                except Exception:
                    raw_audio = None
            onset_s, tail_s = silence_bounds(raw_audio, sample_rate) if raw_audio is not None else (None, None)
            return {
                "path": path, "text": graphemes, "duration": duration, "seg_idx": index,
                "raw": raw_output, "take": take, "cache_key": cache_hash, "engine_version": engine_version,
                "words": words or [], "onset_s": onset_s, "tail_s": tail_s,
            }

        if hit_paths is not None:
            # Cache-key naming hit: the files are the segments. No write.
            chunk_files = []
            for path, graphemes in zip(hit_paths, predicted_texts):
                if self.cancel_event.is_set():
                    break
                if progress_callback:
                    progress_callback(len(graphemes), graphemes)
                chunk_files.append(result(path, graphemes, _audio_duration_s(path, sample_rate)))
            return chunk_files

        chunk_files = []
        sub_idx = 0
        base_name = f"{config.get('filename', 'output')}_{config.get('time_id', '0')}_part{index}"

        def process_and_save(graphemes, raw_audio, words=None):
            nonlocal sub_idx
            if key_naming:
                path = os.path.join(cache_dir, f"{cache_hash}_{sub_idx}.{cache_ext}")
                processed_audio = raw_audio
            else:
                path = os.path.join(config["out_dir"], f"{base_name}_{sub_idx}.{fmt}")
                processed_audio = raw_audio if raw_output else self.process_audio(raw_audio, sample_rate, config)
            _write_audio(path, processed_audio, sample_rate, type(self).__name__)
            return result(path, graphemes, len(processed_audio) / float(sample_rate), raw_audio=raw_audio,
                          words=words)

        try:
            if cached_segments:
                for graphemes, audio in cached_segments:
                    if self.cancel_event.is_set():
                        break
                    if progress_callback:
                        progress_callback(len(graphemes), graphemes)
                    chunk_files.append(process_and_save(graphemes, audio))
                    sub_idx += 1
            else:
                synth_config = {**config, "voice": config["voice"], "speed": eff_speed, "lang_code": lang_code}
                for piece in predicted_texts:
                    if self.cancel_event.is_set():
                        break
                    # One model call per piece. The model concatenates
                    # whatever it produces for the piece (KPipeline cuts at
                    # ~510 phoneme tokens), so the file count is always the
                    # predicted count. The result's text is the piece, not
                    # the model's graphemes.
                    synthesis = self._synthesize(piece, synth_config)
                    audio = np.asarray(to_numpy(synthesis.audio), dtype=np.float32).reshape(-1)
                    if self.cancel_event.is_set() or not len(audio):
                        break
                    words = list(synthesis.words or [])
                    graphemes = piece
                    if progress_callback:
                        progress_callback(len(graphemes), graphemes)

                    if use_cache and cache_hash and not key_naming:
                        try:
                            sf.write(os.path.join(cache_dir, f"{cache_hash}_{sub_idx}.wav"), audio, sample_rate)
                        except Exception as e:
                            print(f"Cache write error: {e}")

                    chunk_files.append(process_and_save(graphemes, audio, words))
                    sub_idx += 1
        finally:
            if key_naming and cache_hash:
                try:
                    os.remove(os.path.join(cache_dir, f"{cache_hash}{RESERVED_SUFFIX}"))
                except OSError:
                    pass

        return chunk_files
