"""The exports log (plan 38): daw/exports_log.py."""
import json
import os

from kokoro_gui.daw import exports_log


def _file(tmp_path):
    return str(tmp_path / "exports.json")


def _audio(tmp_path, name="mix.wav"):
    path = tmp_path / name
    path.write_bytes(b"x")
    return str(path)


def test_log_sits_beside_the_config_file(tmp_path):
    assert exports_log.log_path(str(tmp_path / "config_qt.json")) == str(tmp_path / "exports.json")


def test_record_then_entries_newest_first(tmp_path):
    log = _file(tmp_path)
    first, second = _audio(tmp_path, "a.wav"), _audio(tmp_path, "b.wav")
    exports_log.record(log, first, project_title="Book", duration_s=2.5, fmt="wav", extras=[str(tmp_path / "a.srt")],
                       when=1.0)
    exports_log.record(log, second, project_title="Book", project_path="/p/book.tbaw", fmt="wav", when=2.0)

    rows = exports_log.entries(log)

    assert [r["path"] for r in rows] == [os.path.abspath(second), os.path.abspath(first)]
    assert rows[1]["duration_s"] == 2.5 and rows[1]["project_title"] == "Book" and rows[1]["format"] == "wav"
    assert rows[1]["extras"] == [os.path.abspath(tmp_path / "a.srt")]
    assert rows[0]["project_path"] == "/p/book.tbaw"


def test_log_keeps_the_last_200(tmp_path):
    log = _file(tmp_path)
    for n in range(exports_log.MAX_ENTRIES + 5):
        exports_log.record(log, str(tmp_path / f"{n}.wav"), when=float(n))
    rows = exports_log.entries(log)
    assert len(rows) == exports_log.MAX_ENTRIES
    assert os.path.basename(rows[0]["path"]) == f"{exports_log.MAX_ENTRIES + 4}.wav"
    assert os.path.basename(rows[-1]["path"]) == "5.wav"


def test_a_deleted_file_stays_listed_as_missing(tmp_path):
    log = _file(tmp_path)
    path = _audio(tmp_path)
    exports_log.record(log, path)
    assert exports_log.entries(log)[0]["missing"] is False
    os.remove(path)
    rows = exports_log.entries(log)
    assert len(rows) == 1 and rows[0]["missing"] is True


def test_a_corrupt_or_wrong_shaped_log_reads_as_empty(tmp_path):
    log = _file(tmp_path)
    with open(log, "w") as f:
        f.write("{not json")
    assert exports_log.entries(log) == []
    with open(log, "w") as f:
        json.dump({"a": 1}, f)
    assert exports_log.entries(log) == []
    exports_log.record(log, _audio(tmp_path))  # and the next record starts a fresh log
    assert len(exports_log.entries(log)) == 1


def test_bad_fields_in_an_entry_are_cleaned(tmp_path):
    log = _file(tmp_path)
    with open(log, "w") as f:
        json.dump([5, {"path": ""}, {"path": "x.wav", "when": "soon", "extras": "no", "duration_s": True,
                                      "format": 3}], f)
    rows = exports_log.entries(log)
    assert len(rows) == 1
    assert rows[0]["when"] == 0.0 and rows[0]["extras"] == [] and rows[0]["duration_s"] == 0.0
    assert rows[0]["format"] == ""


def test_record_writes_through_a_temp_file(tmp_path, monkeypatch):
    log = _file(tmp_path)
    calls = []
    real = os.replace

    def replace(a, b):
        calls.append((a, b))
        real(a, b)

    monkeypatch.setattr(os, "replace", replace)
    exports_log.record(log, _audio(tmp_path))
    assert calls == [(log + ".tmp", log)]
    assert not os.path.exists(log + ".tmp")


def test_delete_targets_are_the_audio_and_extras_beside_it(tmp_path):
    audio, srt = _audio(tmp_path, "mix.wav"), _audio(tmp_path, "mix.srt")
    other = _audio(tmp_path, "keep.txt")
    elsewhere = tmp_path / "sub"
    elsewhere.mkdir()
    outside = str(elsewhere / "mix.csv")
    with open(outside, "w") as f:
        f.write("x")
    entry = {"path": audio, "extras": [srt, outside, str(tmp_path / "gone.srt"), str(tmp_path)]}

    targets = exports_log.delete_targets(entry)

    assert sorted(os.path.basename(t) for t in targets) == ["mix.srt", "mix.wav"]
    assert other not in targets and outside not in targets


def test_forget_drops_the_entry(tmp_path):
    log = _file(tmp_path)
    a, b = _audio(tmp_path, "a.wav"), _audio(tmp_path, "b.wav")
    exports_log.record(log, a)
    exports_log.record(log, b)
    exports_log.forget(log, a)
    assert [os.path.basename(r["path"]) for r in exports_log.entries(log)] == ["b.wav"]
