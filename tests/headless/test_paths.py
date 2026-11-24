import os
import re
from datetime import datetime

import pytest

from kokoro_gui.headless import paths
from kokoro_gui.headless.errors import HeadlessError


def _symlink(target, link):
    try:
        os.symlink(target, link, target_is_directory=os.path.isdir(target))
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not permitted here")


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "audio_output" / "headless"
    r.mkdir(parents=True)
    return str(r)


# --- slug ---------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "", "   ", "...", "a/b", "a\\b", "..\\..\\x", "name. ", "con", "CON", "nul.txt", "Com1", "lpt9 test",
    "x" * 500, "../../etc/passwd", "tab\tname", "a\0b", "\u202eevil", "naïve café", "日本語", "trailing. . ",
])
def test_slug_is_a_safe_file_name(text):
    slug = paths.slugify(text)
    assert slug and len(slug) <= paths.MAX_SLUG
    assert re.fullmatch(r"[A-Za-z0-9_-]+", slug)
    assert not re.match(r"(?i)^(con|prn|aux|nul|com\d|lpt\d)($|[-_.])", slug)


def test_slug_keeps_readable_text():
    assert paths.slugify("Chapter One: The Start") == "Chapter-One-The-Start"


def test_slug_empty_uses_fallback():
    assert paths.slugify("") == "render"
    assert paths.slugify("///", fallback="speak") == "speak"


def test_slug_length_bound():
    assert len(paths.slugify("a" * 1000)) == paths.MAX_SLUG
    assert len(paths.slugify("a" * 1000, max_len=8)) == 8


# --- per-call folder ------------------------------------------------------

def test_new_render_dir_name_and_location(root):
    path = paths.new_render_dir("My Story", root=root, now=datetime(2026, 11, 23, 14, 5, 9))
    assert os.path.isdir(path)
    assert os.path.dirname(path) == os.path.abspath(root)
    assert os.path.basename(path) == "20261123-140509_My-Story"


def test_new_render_dir_never_reuses(root):
    now = datetime(2026, 1, 2, 3, 4, 5)
    made = [paths.new_render_dir("x", root=root, now=now) for _ in range(4)]
    assert len(set(made)) == 4
    assert all(os.path.isdir(p) for p in made)
    assert os.path.basename(made[1]).endswith("_x-2")


def test_new_render_dir_does_not_touch_existing_files(root):
    now = datetime(2026, 1, 2, 3, 4, 5)
    first = paths.new_render_dir("x", root=root, now=now)
    keep = os.path.join(first, "keep.txt")
    open(keep, "w").close()
    paths.new_render_dir("x", root=root, now=now)
    assert os.path.exists(keep)


def test_default_root_is_under_cwd(tmp_path):
    assert paths.renders_root() == str(tmp_path / "audio_output" / "headless")
    made = paths.new_render_dir("hello")
    assert os.path.dirname(made) == paths.renders_root()


def test_hostile_slug_stays_inside_root(root):
    made = paths.new_render_dir("..\\..\\..\\evil", root=root)
    assert os.path.dirname(made) == os.path.abspath(root)


# --- out guard --------------------------------------------------------------

def test_relative_out_under_cwd(tmp_path):
    assert paths.check_out_path("a.wav") == os.path.realpath(tmp_path / "a.wav")
    nested = os.path.join("sub", "dir", "a.wav")
    assert paths.check_out_path(nested) == os.path.realpath(tmp_path / "sub" / "dir" / "a.wav")


def test_cwd_itself_is_allowed(tmp_path):
    assert paths.check_out_path(".") == os.path.realpath(tmp_path)


def test_absolute_out_under_root(root):
    p = os.path.join(root, "x", "a.wav")
    assert paths.check_out_path(p, root=root) == os.path.realpath(p)


def test_root_outside_cwd_is_allowed(tmp_path, root):
    other = tmp_path / "elsewhere"
    other.mkdir()
    cwd = str(other)
    assert paths.check_out_path(os.path.join(root, "a.wav"), root=root, cwd=cwd)
    with pytest.raises(HeadlessError):
        paths.check_out_path(str(tmp_path / "audio_output" / "a.wav"), root=root, cwd=cwd)


@pytest.mark.parametrize("bad", ["..", os.path.join("..", "a.wav"), os.path.join("sub", "..", "..", "a.wav")])
def test_dotdot_escape_refused(tmp_path, bad):
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(HeadlessError) as exc:
        paths.check_out_path(bad, cwd=str(work), root=str(work / "audio_output" / "headless"))
    assert exc.value.code == "out_not_allowed"
    assert exc.value.exit_code == 2


def test_dotdot_that_stays_inside_is_fine(tmp_path):
    assert paths.check_out_path(os.path.join("a", "..", "b.wav")) == os.path.realpath(tmp_path / "b.wav")


def test_absolute_outside_refused(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(HeadlessError):
        paths.check_out_path(str(tmp_path / "other.wav"), cwd=str(work))


def test_sibling_with_shared_prefix_refused(tmp_path):
    base = tmp_path / "work"
    (base / "audio_output" / "headless").mkdir(parents=True)
    evil = base / "audio_output" / "headless_evil"
    evil.mkdir()
    root = str(base / "audio_output" / "headless")
    cwd = str(base / "sub")
    os.mkdir(cwd)
    with pytest.raises(HeadlessError):
        paths.check_out_path(str(evil / "a.wav"), root=root, cwd=cwd)
    # the cwd check compares whole components too
    work2 = tmp_path / "w"
    work2.mkdir()
    (tmp_path / "w_evil").mkdir()
    with pytest.raises(HeadlessError):
        paths.check_out_path(str(tmp_path / "w_evil" / "a.wav"), cwd=str(work2), root=root)


def test_symlink_escaping_refused(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    _symlink(str(outside), str(work / "link"))
    with pytest.raises(HeadlessError):
        paths.check_out_path(os.path.join("link", "a.wav"), cwd=str(work), root=str(work / "r"))


def test_symlink_staying_inside_allowed(tmp_path):
    (tmp_path / "real").mkdir()
    _symlink(str(tmp_path / "real"), str(tmp_path / "link"))
    assert paths.check_out_path(os.path.join("link", "a.wav")) == os.path.realpath(tmp_path / "real" / "a.wav")


def test_dangling_symlink_to_outside_refused(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    _symlink(str(tmp_path / "nowhere" / "x.wav"), str(work / "out.wav"))
    with pytest.raises(HeadlessError):
        paths.check_out_path("out.wav", cwd=str(work), root=str(work / "r"))


@pytest.mark.parametrize("bad", ["", "   ", "a\0b.wav"])
def test_empty_and_nul_refused(bad):
    with pytest.raises(HeadlessError):
        paths.check_out_path(bad)


@pytest.mark.parametrize("bad", ["\\\\server\\share\\a.wav", "//server/share/a.wav", "\\\\?\\C:\\a.wav",
                                 "\\\\.\\pipe\\x"])
def test_unc_and_device_paths_refused(bad):
    with pytest.raises(HeadlessError) as exc:
        paths.check_out_path(bad)
    assert exc.value.code == "out_not_allowed"


def test_non_path_refused():
    with pytest.raises(HeadlessError):
        paths.check_out_path(None)
    with pytest.raises(HeadlessError):
        paths.check_out_path(b"a.wav")


@pytest.mark.skipif(os.name != "nt", reason="drive letters")
def test_other_drive_refused(tmp_path):
    drive = os.path.splitdrive(str(tmp_path))[0].upper()
    other = "D:" if drive != "D:" else "C:"
    with pytest.raises(HeadlessError):
        paths.check_out_path(other + "\\a.wav")


@pytest.mark.skipif(os.name != "nt", reason="case-insensitive file system")
def test_case_difference_is_allowed_on_windows(tmp_path):
    swapped = str(tmp_path).swapcase()
    assert paths.check_out_path(os.path.join(swapped, "a.wav"), cwd=str(tmp_path))


@pytest.mark.skipif(os.name == "nt", reason="case-sensitive file system")
def test_case_difference_is_refused_on_posix(tmp_path):
    work = tmp_path / "Work"
    work.mkdir()
    with pytest.raises(HeadlessError):
        paths.check_out_path(str(tmp_path / "work" / "a.wav"), cwd=str(work), root=str(work / "r"))
