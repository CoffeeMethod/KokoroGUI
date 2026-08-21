"""Tests for KokoroEngine.resolve_voice_path (kokoro_engine.py:110-121)."""
import os

import torch


def test_standard_voice_name_passthrough(engine):
    assert engine.resolve_voice_path("af_heart") == "af_heart"


def test_custom_voice_resolves_to_abspath(engine, isolated_dirs):
    torch.save(torch.zeros(1), str(isolated_dirs.custom_voices / "foo.pt"))

    resolved = engine.resolve_voice_path("foo")

    assert os.path.isabs(resolved)
    assert os.path.exists(resolved)
    assert os.path.dirname(resolved) == str(isolated_dirs.custom_voices)


def test_path_traversal_sanitized(engine, isolated_dirs):
    torch.save(torch.zeros(1), str(isolated_dirs.custom_voices / "secrets.pt"))

    resolved = engine.resolve_voice_path("../../secrets")

    assert os.path.exists(resolved)
    assert os.path.dirname(resolved) == str(isolated_dirs.custom_voices)
