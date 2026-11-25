"""headless/update.py: `render_script(update=...)` on a project that is already
there. What is kept and what is generated again, the refusals, and that a
failure puts the bundle, the mixdown and the working folder back."""
import datetime
import json
import os
import shutil
import zipfile

import pytest

from kokoro_gui.daw import mixdown
from kokoro_gui.daw.models import Character, Clip, Run, Segment, Track
from kokoro_gui.headless import render as render_module
from kokoro_gui.headless import update as update_module
from kokoro_gui.headless.host import Host
from kokoro_gui.headless.render import render_script
from kokoro_gui.qt import project as project_io

from .support import SCRIPT, failing, files_under, keys_of, reopen, working_dirs

pytestmark = pytest.mark.usefixtures("offline")

CAST = {"Narrator": "v1", "Sam": {"voice": "v2", "speed": 1.2}, "Ann": "v1"}
EDITED = SCRIPT.replace("Who is there?", "Who is it?")


@pytest.fixture
def book(fake):
    """A rendered project; `fake.spoken` starts empty."""
    result = render_script(SCRIPT, engine="fake", cast=CAST, title="Book")
    fake.spoken.clear()
    return result


def update(book, script=EDITED, **kwargs):
    return render_script(script, engine="fake", update=book.project_path, **kwargs)


def contents(folder):
    out = {}
    for name in files_under(folder):
        with open(os.path.join(folder, name), "rb") as f:
            out[name] = f.read()
    return out


def own_files(folder):
    return {name: data for name, data in contents(folder).items() if "/" not in name and name != "lock"}


def edit_saved(path, change):
    """Opens the project as the app would, changes its document, saves it."""
    loaded = reopen(path)
    change(loaded.document, loaded.project_settings)
    info = project_io.inspect_bundle(path)
    project_io.autosave_to_dir(loaded.document, loaded.project_settings, loaded.project_dir)
    project_io.save_project(loaded.document, path, loaded.project_settings, loaded.project_dir, info.project_id,
                            previous_manifest=info.manifest)


# -- what it does --------------------------------------------------------------------


@pytest.mark.parametrize("working_folder", ["kept", "deleted"])
def test_one_changed_paragraph_is_one_clip(fake, book, working_folder):
    if working_folder == "deleted":
        project_io.delete_project_dir(working_dirs()[0])
    before = keys_of(book.project_path)
    result = update(book, cast=CAST)
    assert (result.clips, result.regenerated) == (3, 1)
    assert [s[0] for s in fake.spoken] == ["Who is it?"]
    after = keys_of(book.project_path)
    assert (after[0], after[2]) == (before[0], before[2]) and after[1] != before[1]
    assert (result.project_path, result.mixdown_path) == (book.project_path, book.mixdown_path)
    assert files_under(book.out_dir) == ["Book.tbaw", "Book.wav"]
    assert [w["word"] for w in result.words] == "The door opens. Who is it? Only me, with the post.".split()


class _Later(datetime.datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2031, 5, 6, 7, 8, 9)


def test_it_reopens_with_every_clip_generated_and_the_same_project_id(fake, book, monkeypatch):
    first = project_io.inspect_bundle(book.project_path)
    monkeypatch.setattr(project_io.datetime, "datetime", _Later)
    update(book, cast=CAST)
    info = project_io.inspect_bundle(book.project_path)
    assert info.project_id == first.project_id and info.manifest["created"] == first.manifest["created"]
    assert info.manifest["modified"] == "2031-05-06T07:08:09"
    (folder,) = working_dirs()
    session = project_io.read_session(folder)
    assert not project_io.is_locked(folder) and session["dirty"] is False
    assert project_io.open_action(session, info, folder, []) == "reuse"
    assert project_io.load_project_dir(folder).document.text == EDITED.replace("[Sam]: ", "").replace("[Ann]: ", "")
    loaded = reopen(book.project_path)
    assert loaded.notices == [] and all(c.segments and c.status == "generated" for c in loaded.document.clips)


def test_the_same_script_generates_nothing(fake, book):
    result = update(book, SCRIPT, cast=CAST)
    assert result.regenerated == 0 and fake.spoken == []
    assert result.words == book.words and result.duration_s == pytest.approx(book.duration_s)


def test_moved_added_and_removed_paragraphs(fake, book):
    result = update(book, "[Ann]: Only me, with the post.\n\n[Sam]: Who is there?\n\nNew line.", cast=CAST)
    assert result.regenerated == 1 and [s[0] for s in fake.spoken] == ["New line."]


def test_a_missing_segment_file_is_generated_again(fake, book):
    (folder,) = working_dirs()
    loaded = project_io.load_project_dir(folder)
    os.remove(loaded.document.clips[0].segments[0].audio_path)
    assert update(book, SCRIPT, cast=CAST).regenerated == 1


def test_a_new_voice_in_the_cast_generates_that_speaker_again(fake, book):
    result = update(book, SCRIPT, cast={**CAST, "Sam": "v1"})
    assert result.regenerated == 1 and fake.spoken == [("Who is there?", "v1", 1.0, "a")]
    sam = reopen(book.project_path).document.get_character_by_name("Sam")
    assert sam.preset_data == {"voice": "v1"}


def test_a_speaker_the_cast_does_not_name_keeps_the_projects_settings(fake, book):
    result = update(book)
    assert fake.spoken == [("Who is it?", "v2", 1.2, "a")] and result.warnings == []
    assert update(book, EDITED + "\n\n[Eve]: Hi.").warnings == ["Speaker 'Eve' is not in the cast; using the default voice."]


def test_settings_identities_and_mixers_are_kept(fake, book):
    ids = {}

    def change(document, settings):
        document.settings["gap_s"] = 0.5
        document.settings["markers"] = []
        sam = document.get_character_by_name("Sam")
        ids["sam"] = sam.id
        sam.highlight_color, sam.variants, sam.extra = "#123456", {"angry": "v1"}, {"later": 1}
        track = next(t for t in document.tracks if t.character_id == sam.id)
        track.gain, track.pan, track.mute, track.automation = 0.5, -0.25, True, [[0.0, 1.0]]
        ids["track"] = track.id
        settings["note"] = "mine"

    edit_saved(book.project_path, change)
    update(book, cast=CAST)
    loaded = reopen(book.project_path)
    document = loaded.document
    sam = document.get_character_by_name("Sam")
    assert (sam.id, sam.highlight_color, sam.variants, sam.extra) == (ids["sam"], "#123456", {"angry": "v1"}, {"later": 1})
    track = document.get_track(ids["track"])
    assert (track.gain, track.pan, track.mute, track.automation) == (0.5, -0.25, True, [[0.0, 1.0]])
    assert [c.track_id for c in document.clips if c.character_id == sam.id] == [track.id]
    assert document.settings["gap_s"] == 0.5 and loaded.project_settings == {"title": "Book", "note": "mine"}
    assert all(document.get_track(c.track_id) is not None for c in document.clips)


def test_the_unified_layout_keeps_its_lanes(fake, book):
    def change(document, _settings):
        document.settings["track_layout"] = {"mode": "unified", "lanes": 2}
        document.tracks = [Track("Lane 1", lane=1, gain=0.7), Track("Lane 2", lane=2)]
        for clip in document.clips:
            clip.track_id = document.tracks[0].id

    edit_saved(book.project_path, change)
    update(book, cast=CAST)
    document = reopen(book.project_path).document
    assert all(document.get_track(c.track_id).lane for c in document.clips)
    assert next(t for t in document.tracks if t.lane == 1).gain == 0.7


def test_a_parked_takes_audio_is_used_again(fake, book):
    def change(document, _settings):
        clip = document.clips[0]
        clip.takes[0] = clip.segments
        clip.segments = []
        clip.overrides["take"] = 1

    edit_saved(book.project_path, change)
    assert update(book, SCRIPT, cast=CAST).regenerated == 0


def test_a_hostile_projects_character_values_are_cleaned(fake, book):
    def change(document, _settings):
        document.get_character_by_name("Sam").preset_data = {"voice": "v2", "speed": 1e9, "pitch": "x", "volume": True}

    edit_saved(book.project_path, change)
    update(book)
    assert fake.spoken[0][:3] == ("Who is it?", "v2", 4.0)


def test_a_projects_voice_the_engine_lacks_is_unknown(fake, book):
    edit_saved(book.project_path, lambda d, _s: d.get_character_by_name("Sam").preset_data.update(voice="gone"))
    assert failing(EDITED, update=book.project_path).code == "unknown_voice"
    assert update(book, cast=CAST).regenerated == 1


def test_another_engines_character_gets_the_cast_or_the_default(fake, book):
    edit_saved(book.project_path, lambda d, _s: setattr(d.get_character_by_name("Sam"), "backend_id", "kokoro"))
    result = update(book)
    assert result.warnings == ["Speaker 'Sam' is not in the cast; using the default voice."]
    assert reopen(book.project_path).document.get_character_by_name("Sam").backend_id == "fake"


def test_a_preset_only_the_project_has(fake, book, tmp_path):
    folder = tmp_path / "presets" / "fx"
    folder.mkdir(parents=True)
    (folder / "Radio.json").write_text(json.dumps({"highpass_enabled": True}))
    cast = {**CAST, "Sam": {"voice": "v2", "fx_preset": "Radio"}}
    update(book, cast=cast)
    os.remove(folder / "Radio.json")
    project_io.delete_project_dir(working_dirs()[0])
    update(book, SCRIPT, cast=cast)
    with zipfile.ZipFile(book.project_path) as zf:
        assert "fx/Radio.json" in zf.namelist()


def test_a_title_replaces_the_projects(fake, book):
    update(book, cast=CAST, title="Second")
    assert reopen(book.project_path).project_settings == {"title": "Second"}
    assert files_under(book.out_dir) == ["Book.tbaw", "Book.wav"]


# -- the mixdown beside it -------------------------------------------------------------


def test_outside_the_renders_root_a_wav_is_never_replaced(fake, tmp_path):
    result = render_script(SCRIPT, engine="fake", cast=CAST, out="mine", title="Book")
    (tmp_path / "mine" / "Book.wav").write_bytes(b"recorded by hand")
    again = render_script(EDITED, engine="fake", cast=CAST, update=result.project_path)
    assert (tmp_path / "mine" / "Book.wav").read_bytes() == b"recorded by hand"
    assert os.path.basename(again.mixdown_path) == "Book (2).wav"
    assert files_under(tmp_path / "mine") == ["Book (2).wav", "Book.tbaw", "Book.wav"]


# -- refusals ----------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["missing.tbaw", "book.json", "folder.tbaw", "..\\..\\x.tbaw", "//host/share/x.tbaw"])
def test_update_has_to_be_a_bundle_in_an_allowed_place(fake, tmp_path, name):
    (tmp_path / "book.json").write_text("{}")
    (tmp_path / "folder.tbaw").mkdir()
    assert failing(update=name).code in ("bad_arguments", "out_not_allowed")
    assert fake.spoken == [] and working_dirs() == []


@pytest.mark.parametrize("data", [b"", b"not a zip", b"PK\x05\x06" + b"\0" * 18])
def test_a_file_that_is_no_bundle(fake, tmp_path, data):
    (tmp_path / "x.tbaw").write_bytes(data)
    assert failing(update="x.tbaw").code == "bad_arguments"
    assert (tmp_path / "x.tbaw").read_bytes() == data and working_dirs() == []


def test_a_bundle_whose_document_is_not_one(fake, book):
    broken = os.path.join(book.out_dir, "Broken.tbaw")
    with zipfile.ZipFile(book.project_path) as src, zipfile.ZipFile(broken, "w") as dst:
        for name in src.namelist():
            dst.writestr(name, b"[1]" if name == "document.json" else src.read(name))
    project_io.delete_project_dir(working_dirs()[0])
    assert failing(update=broken).code == "bad_arguments"
    assert not any(project_io.is_locked(d) for d in working_dirs())


def test_out_and_update_together(fake, book):
    assert failing(update=book.project_path, out="x").code == "bad_arguments"


def test_a_project_open_elsewhere_is_locked(fake, book):
    lock = project_io.ProjectLock(working_dirs()[0]).acquire()
    try:
        before = contents(book.out_dir)
        error = failing(EDITED, update=book.project_path)
        assert (error.code, error.exit_code) == ("project_locked", 6)
        assert contents(book.out_dir) == before and lock.held
    finally:
        lock.release()


def test_a_project_the_app_opened_through_a_link_is_locked_too(fake, book, tmp_path):
    link = tmp_path / "link"
    try:
        os.symlink(book.out_dir, link, target_is_directory=True)
    except OSError:
        try:
            import _winapi

            _winapi.CreateJunction(book.out_dir, str(link))
        except (ImportError, OSError):
            pytest.skip("no links here")
    (folder,) = working_dirs()
    session = project_io.read_session(folder)
    project_io.write_session(folder, {**session, "source_path": str(link / "Book.tbaw")})
    lock = project_io.ProjectLock(folder).acquire()
    try:
        assert failing(EDITED, update=book.project_path).code == "project_locked"
    finally:
        lock.release()
    assert update(book, cast=CAST).regenerated == 1 and len(working_dirs()) == 1


def test_unsaved_changes_are_refused_and_kept(fake, book):
    (folder,) = working_dirs()
    loaded = project_io.load_project_dir(folder)
    loaded.document.clips[0].note = "unsaved"
    project_io.autosave_to_dir(loaded.document, {}, folder)
    project_io.write_session(folder, {**project_io.read_session(folder), "dirty": True})
    before, output = own_files(folder), contents(book.out_dir)
    error = failing(EDITED, update=book.project_path)
    assert (error.code, error.exit_code) == ("unsaved_session", 2)
    assert own_files(folder) == before and contents(book.out_dir) == output and fake.spoken == []

    result = update(book, cast=CAST, discard_session=True)
    assert result.regenerated == 1 and project_io.read_session(folder)["dirty"] is False


def _bed(document):
    clip = Clip(source="imported", original_audio_path="x.wav")
    document.clips.append(clip)
    document.runs.append(Run("x.wav", clip.id, "placeholder"))


def _guide_take(document):
    clip = document.clips[0]
    path = os.path.join(working_dirs()[0], "audio", "imported", "guide.wav")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    shutil.copy(clip.segments[0].audio_path, path)
    clip.takes[1] = [Segment(0, "guide", None, path, 0.3)]


def _nested(document):
    clip = Clip(source="nested", child={"kind": "embedded", "id": "abc"})
    document.clips.append(clip)
    document.runs.append(Run("Part", clip.id, "placeholder"))


@pytest.mark.parametrize("add", [_bed, _guide_take, _nested])
@pytest.mark.parametrize("working_folder", ["kept", "deleted"])
def test_a_project_with_audio_a_script_cannot_make_again_is_refused(fake, book, add, working_folder):
    edit_saved(book.project_path, lambda d, _s: add(d))
    if working_folder == "deleted":
        project_io.delete_project_dir(working_dirs()[0])
    before = contents(book.out_dir)
    error = failing(EDITED, update=book.project_path, discard_session=True)
    assert error.code == "bad_arguments" and "would drop" in error.message
    assert contents(book.out_dir) == before and fake.spoken == []
    assert not any(project_io.is_locked(d) for d in working_dirs())


def test_a_refused_project_keeps_its_unsaved_changes(fake, book):
    edit_saved(book.project_path, lambda d, _s: _bed(d))
    (folder,) = working_dirs()
    project_io.write_session(folder, {**project_io.read_session(folder), "dirty": True})
    before = {**contents(folder), "lock": b""}
    assert failing(EDITED, update=book.project_path, discard_session=True).code == "bad_arguments"
    assert contents(folder) == before


# -- a failure puts everything back ----------------------------------------------------------


def disk_full(*_args, **_kwargs):
    raise OSError("disk full")


def keyboard(*_args, **_kwargs):
    raise KeyboardInterrupt


def whole_bundle_then_fail(*args, _real=project_io.write_bundle, **kwargs):
    _real(*args, **kwargs)
    raise OSError("disk full")


FAILURES = {
    "build": (render_module, "build_document", lambda *a, **k: 1 / 0),
    "carry over": (render_module, "carry_over", disk_full),
    "reuse": (render_module, "reuse_segments", disk_full),
    "arrange": (render_module, "compute_arrangement", disk_full),
    "mix": (mixdown, "write_audio", disk_full),
    "mix, Ctrl+C": (mixdown, "write_audio", keyboard),
    "plan": (project_io, "plan_save", disk_full),
    "save": (project_io, "write_bundle", disk_full),
    "save, after the write": (project_io, "write_bundle", whole_bundle_then_fail),
    "save, Ctrl+C": (project_io, "write_bundle", keyboard),
    "session": (project_io, "record_save", disk_full),
}


def assert_as_it_was(book, output, folder, own):
    assert contents(book.out_dir) == output
    assert own_files(folder) == own and not project_io.is_locked(folder)
    info = project_io.inspect_bundle(book.project_path)
    assert project_io.open_action(project_io.read_session(folder), info, folder, []) == "reuse"


@pytest.mark.parametrize("stage", FAILURES)
def test_a_failed_update_leaves_the_project_as_it_was(fake, book, monkeypatch, stage):
    (folder,) = working_dirs()
    output, own = contents(book.out_dir), own_files(folder)
    with monkeypatch.context() as patch:
        patch.setattr(*FAILURES[stage])
        failing(EDITED, update=book.project_path, cast=CAST)
    assert_as_it_was(book, output, folder, own)
    assert update(book, cast=CAST).clips == 3


def test_the_last_rename_failing_puts_the_bundle_back(fake, book, monkeypatch):
    (folder,) = working_dirs()
    output, own = contents(book.out_dir), own_files(folder)
    real = project_io._replace_with_retries

    def replace(src, dst):
        if dst.endswith(".wav"):
            raise PermissionError("held")
        real(src, dst)

    monkeypatch.setattr(project_io, "_replace_with_retries", replace)
    assert failing(EDITED, update=book.project_path, cast=CAST).code == "generation_failed"
    assert_as_it_was(book, output, folder, own)


@pytest.mark.parametrize("stage", ["generate", "mix", "save"])
def test_a_cancelled_update_leaves_the_project_as_it_was(fake, book, stage):
    (folder,) = working_dirs()
    output, own = contents(book.out_dir), own_files(folder)
    seen = []
    failing(EDITED, update=book.project_path, cast=CAST, cancel=lambda: bool(seen),
            on_progress=lambda name, *_a: seen.append(name) if name == stage else None)
    assert seen
    assert_as_it_was(book, output, folder, own)


def test_a_failed_generate_leaves_the_project_as_it_was(fake, book):
    (folder,) = working_dirs()
    output, own = contents(book.out_dir), own_files(folder)
    fake.speak = lambda text: 1 / 0
    error = failing(EDITED, update=book.project_path, cast=CAST)
    assert error.code == "generation_failed" and "paragraph 2 of 3" in error.message
    assert_as_it_was(book, output, folder, own)


def test_unknown_voice_before_anything_is_touched(fake, book):
    (folder,) = working_dirs()
    output, own = contents(book.out_dir), own_files(folder)
    assert failing(EDITED, update=book.project_path, cast={"Sam": "nope"}).code == "unknown_voice"
    assert_as_it_was(book, output, folder, own)


def test_the_folder_is_marked_dirty_while_the_bundle_is_written(fake, book, monkeypatch):
    """A process killed in there must leave a folder the app offers to
    recover, not one it takes for a clean copy."""
    (folder,) = working_dirs()
    seen = []
    real = project_io.write_bundle

    def write(*args, **kwargs):
        seen.append(project_io.read_session(folder)["dirty"])
        return real(*args, **kwargs)

    monkeypatch.setattr(project_io, "write_bundle", write)
    update(book, cast=CAST)
    assert seen == [True] and project_io.read_session(folder)["dirty"] is False


def test_without_hard_links_the_backup_is_a_copy(fake, book, monkeypatch):
    (folder,) = working_dirs()
    output, own = contents(book.out_dir), own_files(folder)

    def no_links(*_args, **_kwargs):
        raise OSError("no hard links")

    monkeypatch.setattr(os, "link", no_links)
    with monkeypatch.context() as patch:
        patch.setattr(project_io, "record_save", disk_full)
        failing(EDITED, update=book.project_path, cast=CAST)
    assert_as_it_was(book, output, folder, own)
    update(book, cast=CAST)
    assert files_under(book.out_dir) == ["Book.tbaw", "Book.wav"]


def test_a_project_saved_by_the_app_is_updated(fake, book):
    """One with a character no line uses and text outside any clip."""
    def change(document, _settings):
        document.characters.append(Character.from_preset_dict("Eve", {"voice": "gone"}, backend_id="fake"))
        document.runs.append(Run("\n\nnotes nobody reads"))

    edit_saved(book.project_path, change)
    result = update(book, cast=CAST)
    assert result.speakers == ["Narrator", "Sam", "Ann"] and result.regenerated == 1


def test_update_on_a_given_host_twice(fake, book):
    with Host("fake") as host:
        assert render_script(EDITED, host=host, update=book.project_path).regenerated == 1
        assert render_script(EDITED, host=host, update=book.project_path).regenerated == 0
    assert update_module.check_update(None) is None
