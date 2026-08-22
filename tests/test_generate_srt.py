"""Tests for KokoroEngine.generate_srt (kokoro_engine.py:545-566)."""


def test_generate_srt_writes_valid_timing_format(engine, tmp_path):
    segments = [
        {"text": "Hello.", "duration": 1.5},
        {"text": "World.", "duration": 2.0},
    ]
    out_path = str(tmp_path / "out.srt")

    ok = engine.generate_srt(segments, out_path)

    assert ok is True
    content = open(out_path, encoding="utf-8").read()
    assert "-->" in content
    assert "00:00:00,000 --> 00:00:01,500" in content
    assert "00:00:01,500 --> 00:00:03,500" in content
    assert "1\nHello." not in content  # text is on its own line, not glued to the index
    assert "Hello." in content and "World." in content


def test_generate_srt_write_failure_returns_false(engine, tmp_path):
    bad_path = str(tmp_path / "missing_dir" / "out.srt")

    ok = engine.generate_srt([{"text": "x", "duration": 1.0}], bad_path)

    assert ok is False
