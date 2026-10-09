"""Size, clear and trim for the segment cache and Audio8's reference codes
(kokoro_gui/engine/cache_admin.py). Every test works in a tmp dir; the
point of most is what a Clear must leave alone."""
import os

import pytest

from kokoro_gui.engine import cache_admin
from kokoro_gui.engines import audio8_tts

KEY_A = "a" * 64
KEY_B = "b" * 64
KEY_C = "c" * 64


def _write(path, size=100, mtime=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


@pytest.fixture
def cache(isolated_dirs):
    """A cache dir with segment files, noise that must survive, a project
    working copy and a log."""
    root = isolated_dirs.cache_dir
    _write(root / f"{KEY_A}_0.wav", 100)
    _write(root / f"{KEY_A}_1.wav", 50)
    _write(root / f"{KEY_B}_0.flac", 200)
    _write(root / "notes.txt", 10)
    _write(root / f"{KEY_A}.reserved", 1)
    _write(root / f"{KEY_C[:60]}_0.wav", 7)  # key too short
    _write(root / f"{'G' * 64}_0.wav", 7)  # not hex
    _write(root / "projects" / "abc123" / "document.json", 300)
    _write(root / "projects" / "abc123" / "audio" / "generated" / f"{KEY_A}_0.wav", 400)
    _write(root / "logs" / "kokorogui.log", 20)
    return root


def _snapshot(root, *subdirs):
    out = {}
    for sub in subdirs:
        for dirpath, _dirs, files in os.walk(root / sub):
            for name in files:
                p = os.path.join(dirpath, name)
                with open(p, "rb") as f:
                    out[p] = f.read()
    return out


def test_usage_counts_only_segment_files(cache):
    assert cache_admin.segment_cache_usage() == (3, 350)


def test_clear_removes_only_segment_files_and_reports_them(cache):
    before = _snapshot(cache, "projects", "logs")

    removed = cache_admin.clear_segment_cache()

    assert removed == (3, 350)
    assert cache_admin.segment_cache_usage() == (0, 0)
    assert _snapshot(cache, "projects", "logs") == before
    left = sorted(p.name for p in cache.iterdir())
    assert left == sorted(["notes.txt", f"{KEY_A}.reserved", f"{KEY_C[:60]}_0.wav", f"{'G' * 64}_0.wav",
                           "projects", "logs"])


def test_clear_never_enters_a_directory_named_like_a_segment(cache):
    lookalike = cache / f"{KEY_C}_0.wav"
    lookalike.mkdir()
    _write(lookalike / "inside.txt", 5)

    cache_admin.clear_segment_cache()

    assert (lookalike / "inside.txt").exists()


def test_clear_skips_a_symlink_and_leaves_its_target(cache, tmp_path):
    target = _write(tmp_path / "outside" / "precious.wav", 42)
    link = cache / f"{KEY_C}_0.wav"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks need privileges here")

    assert cache_admin.segment_cache_usage().count == 3
    cache_admin.clear_segment_cache()

    assert target.read_bytes() == b"x" * 42
    assert os.path.islink(link)


def test_remove_refuses_a_path_that_resolves_outside_the_folder(cache, tmp_path):
    outside = _write(tmp_path / "outside" / f"{KEY_C}_0.wav", 9)
    forged = cache_admin._Entry(outside.name, str(outside), 9, 0.0)

    removed = cache_admin._remove(str(cache), [forged], cache_admin.SEGMENT_FILE)

    assert removed == (0, 0)
    assert outside.exists()


def test_clear_on_a_missing_folder_is_empty(tmp_path):
    gone = str(tmp_path / "nope")
    assert cache_admin.segment_cache_usage(gone) == (0, 0)
    assert cache_admin.clear_segment_cache(gone) == (0, 0)


def test_clear_keeps_going_past_a_file_that_cannot_be_removed(cache, monkeypatch):
    real_remove = os.remove

    def flaky(path):
        if os.path.basename(path) == f"{KEY_A}_0.wav":
            raise PermissionError("in use")
        real_remove(path)

    monkeypatch.setattr(cache_admin.os, "remove", flaky)

    removed = cache_admin.clear_segment_cache()

    assert removed == (2, 250)
    assert (cache / f"{KEY_A}_0.wav").exists()


def test_clear_stops_when_asked(cache):
    calls = iter([False, True, True, True])
    removed = cache_admin.clear_segment_cache(should_stop=lambda: next(calls))
    assert removed.count == 1


# -- trim -------------------------------------------------------------------------


def test_trim_below_the_limit_removes_nothing(cache):
    assert cache_admin.trim_segment_cache(10_000) == (0, 0)
    assert cache_admin.segment_cache_usage() == (3, 350)


def test_trim_removes_the_oldest_key_whole(isolated_dirs):
    root = isolated_dirs.cache_dir
    _write(root / f"{KEY_A}_0.wav", 100, mtime=1000)
    _write(root / f"{KEY_A}_1.wav", 100, mtime=1001)
    _write(root / f"{KEY_B}_0.wav", 100, mtime=2000)
    _write(root / f"{KEY_C}_0.wav", 100, mtime=3000)

    removed = cache_admin.trim_segment_cache(250)

    assert removed == (2, 200)
    assert sorted(p.name for p in root.iterdir()) == [f"{KEY_B}_0.wav", f"{KEY_C}_0.wav"]


def test_a_key_is_as_new_as_its_newest_file(isolated_dirs):
    """A cache hit touches every file of its key, but if only some were
    touched the key still counts as recent."""
    root = isolated_dirs.cache_dir
    _write(root / f"{KEY_A}_0.wav", 100, mtime=1000)
    _write(root / f"{KEY_A}_1.wav", 100, mtime=9000)  # used last
    _write(root / f"{KEY_B}_0.wav", 100, mtime=2000)

    cache_admin.trim_segment_cache(250)

    assert sorted(p.name for p in root.iterdir()) == [f"{KEY_A}_0.wav", f"{KEY_A}_1.wav"]


def test_trim_leaves_everything_else_alone(cache):
    before = _snapshot(cache, "projects", "logs")

    cache_admin.trim_segment_cache(0)

    assert cache_admin.segment_cache_usage() == (0, 0)
    assert _snapshot(cache, "projects", "logs") == before
    assert (cache / "notes.txt").exists()


@pytest.mark.parametrize("value", [0, -5, None, "2048", True, float("nan")])
def test_trim_to_setting_treats_zero_and_junk_as_unlimited(cache, value):
    assert cache_admin.trim_to_setting(value) == (0, 0)
    assert cache_admin.segment_cache_usage().count == 3


def test_trim_to_setting_reads_megabytes(isolated_dirs):
    root = isolated_dirs.cache_dir
    mib = cache_admin.MIB
    _write(root / f"{KEY_A}_0.wav", mib, mtime=1000)
    _write(root / f"{KEY_B}_0.wav", mib, mtime=2000)

    removed = cache_admin.trim_to_setting(1)

    assert removed == (1, mib)
    assert (root / f"{KEY_B}_0.wav").exists()


# -- reference codes ---------------------------------------------------------------


@pytest.fixture
def refs(tmp_path, monkeypatch):
    base = tmp_path / "audio8_refs"
    monkeypatch.setattr(audio8_tts, "AUDIO8_REFS_DIR", str(base))
    codes = base / ".ref_codes_cache"
    _write(codes / ("0123456789abcdef.npy"), 64)
    _write(codes / ("fedcba9876543210.npy"), 32)
    _write(codes / "0123456789abcdef.npy.tmp", 5)
    _write(base / "narrator.wav", 500)
    _write(base / "narrator.txt", 20)
    return base


def test_ref_codes_usage_and_clear_touch_only_npy_files(refs):
    assert cache_admin.ref_codes_dir() == os.path.abspath(refs / ".ref_codes_cache")
    assert cache_admin.ref_codes_usage() == (2, 96)

    removed = cache_admin.clear_ref_codes()

    assert removed == (2, 96)
    assert cache_admin.ref_codes_usage() == (0, 0)
    assert (refs / "narrator.wav").read_bytes() == b"x" * 500
    assert (refs / "narrator.txt").exists()
    assert (refs / ".ref_codes_cache" / "0123456789abcdef.npy.tmp").exists()


def test_ref_codes_with_no_folder_yet_are_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(audio8_tts, "AUDIO8_REFS_DIR", str(tmp_path / "none"))
    assert cache_admin.ref_codes_usage() == (0, 0)
    assert cache_admin.clear_ref_codes() == (0, 0)


# -- project working copies ---------------------------------------------------------


def test_projects_usage_counts_everything_below_projects(cache):
    assert cache_admin.projects_dir() == os.path.abspath(cache / "projects")
    assert cache_admin.projects_usage() == (2, 700)


def test_there_is_no_clear_for_project_working_copies():
    assert not [name for name in dir(cache_admin) if "project" in name and ("clear" in name or "delete" in name)]


# -- formatting ---------------------------------------------------------------------


def test_sizes_read_naturally():
    assert cache_admin.format_size(0) == "0 B"
    assert cache_admin.format_size(1536) == "2 KB"
    assert cache_admin.format_size(5 * 1024 * 1024) == "5.0 MB"
    assert cache_admin.format_size(3 * 1024 ** 3) == "3.00 GB"
    assert cache_admin.describe(cache_admin.Usage(1, 2048)) == "1 file, 2 KB"
    assert cache_admin.describe(cache_admin.Usage(0, 0)) == "0 files, 0 B"
