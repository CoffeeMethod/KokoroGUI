"""kokoro_gui/core/keys.py: the memoized segment key function a document's
dirty check uses."""
import os

import pytest

from kokoro_gui.core import keys as core_keys
from kokoro_gui.daw import revision
from kokoro_gui.daw.models import Clip, Document, Segment
from kokoro_gui.engine import caching
from kokoro_gui.engines.missing import MissingBackend


class FileVoiceBackend:
    """Resolves every voice to one file and reports a settable
    `cache_key_extra`, like Audio8's reference transcript."""
    id = "filevoice"

    def __init__(self, voice_file):
        self.voice_file = voice_file
        self.extra = {"transcript": "one"}

    def resolve_voice_file(self, name, project_dir=None):
        return self.voice_file

    def cache_key_extra(self, config):
        return dict(self.extra)

    def engine_version(self):
        return "1"


CONFIG = {"engine_id": "filevoice", "lang_code": "a", "voice": "ref", "speed": 1.0, "project_dir": None}


@pytest.fixture
def voice_file(tmp_path):
    path = tmp_path / "ref.wav"
    path.write_bytes(b"first")
    return str(path)


@pytest.fixture
def counted(monkeypatch):
    calls = []
    real = caching.segment_key

    def counting(*args, **kwargs):
        calls.append(args[0])
        return real(*args, **kwargs)

    monkeypatch.setattr(core_keys.caching, "segment_key", counting)
    return calls


def test_a_memo_hit_skips_the_hash(voice_file, counted):
    backend = FileVoiceBackend(voice_file)
    key_fn = core_keys.make_key_fn(lambda clip: backend, lambda clip: dict(CONFIG))
    clip = Clip()
    first = key_fn("Hello.", clip)
    assert key_fn("Hello.", clip) == first
    assert counted == ["Hello."]
    assert first == caching.segment_key("Hello.", CONFIG, backend)


def test_a_memo_hit_rechecks_the_voice_files_mtime(voice_file, counted):
    backend = FileVoiceBackend(voice_file)
    key_fn = core_keys.make_key_fn(lambda clip: backend, lambda clip: dict(CONFIG))
    clip = Clip()
    first = key_fn("Hello.", clip)
    with open(voice_file, "wb") as handle:
        handle.write(b"second, re-recorded")
    stamp = os.path.getmtime(voice_file) + 5
    os.utime(voice_file, (stamp, stamp))
    revision.bump_files()
    second = key_fn("Hello.", clip)
    assert len(counted) == 2
    assert second != first


def test_a_memo_hit_rechecks_cache_key_extra(voice_file, counted):
    backend = FileVoiceBackend(voice_file)
    key_fn = core_keys.make_key_fn(lambda clip: backend, lambda clip: dict(CONFIG))
    clip = Clip()
    first = key_fn("Hello.", clip)
    backend.extra = {"transcript": "two"}
    assert key_fn("Hello.", clip) != first
    assert len(counted) == 2


def test_a_missing_backend_returns_the_stored_key(counted):
    key_fn = core_keys.make_key_fn(lambda clip: MissingBackend("audio8"), lambda clip: dict(CONFIG))
    clip = Clip(segments=[Segment(order_index=0, text="Hello.", cache_key="stored", audio_path=None)])
    assert key_fn("Hello.", clip) == "stored"
    assert key_fn("Hello.", Clip()) is None
    assert counted == []


def test_install_key_fn_sets_the_document_callbacks(voice_file):
    backend = FileVoiceBackend(voice_file)
    document = Document(runs=[], clips=[], characters=[])
    configs = []

    def dirty_config(clip):
        configs.append(clip)
        return dict(CONFIG)

    core_keys.install_key_fn(document, lambda clip: backend, lambda clip: dict(CONFIG), dirty_config,
                             lambda: "fingerprint")
    clip = Clip()
    assert document.segment_key_fn("Hello.", clip) == caching.segment_key("Hello.", CONFIG, backend)
    assert document.generation_config_fn(clip) == CONFIG
    assert configs == [clip]
    assert document.inputs_fn() == "fingerprint"


def test_each_install_starts_a_fresh_memo(voice_file, counted):
    backend = FileVoiceBackend(voice_file)
    document = Document(runs=[], clips=[], characters=[])
    for _ in range(2):
        core_keys.install_key_fn(document, lambda clip: backend, lambda clip: dict(CONFIG), lambda clip: CONFIG)
        document.segment_key_fn("Hello.", Clip())
    assert len(counted) == 2
    assert document.inputs_fn is None
