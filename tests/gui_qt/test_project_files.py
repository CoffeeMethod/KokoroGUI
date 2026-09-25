"""Tests for the File menu / project lifecycle and kokoro_gui/qt/project.py:
the `.tbaw` bundle (Claude/old/PLAN_tbaw_bundle.md, grill TB1-TB15)."""
import json
import os
import sys
import zipfile

import pytest
from PySide6.QtGui import QTextCursor

from kokoro_gui.daw.dirty import build_segments_from_results
from kokoro_gui.daw.models import Character, Clip, Document, Run, Segment
from kokoro_gui.qt import project as project_io


def _type(editor, text):
    cursor = editor.textCursor()
    cursor.select(QTextCursor.SelectionType.Document)
    cursor.insertText(text)


def _save_as(qt_app, path):
    qt_app.save_project_as(path)
    qt_app.wait_for_project_io()
    return qt_app.project_path


def _open(qt_app, path):
    qt_app.open_project(path)
    qt_app.wait_for_project_io()


def _simulate_crash(qt_app):
    """The process died: the OS released the lock, the dir stays as it was
    (dirty), and the next window starts with no project."""
    qt_app._project_lock.release()
    qt_app._project_lock = None
    qt_app.project_dir = None
    qt_app.project_id = None
    qt_app.project_path = None


def _generated_clip(qt_app, text="hello world", seconds=0.1):
    """A clean clip whose one segment is a real raw wav inside the project
    dir, named by its key, the way a generate leaves it."""
    import numpy as np
    import soundfile as sf

    qt_app.document.text = text
    character = qt_app.document.characters[0]
    clip = qt_app.document.assign_character_to_range(0, len(text), character.id)
    key = qt_app.document.segment_key_fn(text, clip)
    generated = os.path.join(qt_app.project_dir, "audio", "generated")
    os.makedirs(generated, exist_ok=True)
    path = os.path.join(generated, f"{key}_0.wav")
    sf.write(path, np.full(int(24000 * seconds), 0.2, dtype=np.float32), 24000)
    clip.segments = build_segments_from_results(key, [{"text": text, "path": path, "duration": seconds,
                                                       "cache_key": key,
                                                       "engine_version": qt_app.backend.engine_version()}])
    return clip


# -- project.py: recent list ---------------------------------------------------------


def test_remember_recent_moves_to_front_dedupes_and_caps(tmp_path):
    settings = {}
    for i in range(12):
        project_io.remember_recent(settings, str(tmp_path / f"p{i}.tbaw"))
    project_io.remember_recent(settings, str(tmp_path / "p3.tbaw"))

    recent = settings["recent_projects"]
    assert len(recent) == project_io.MAX_RECENT
    assert recent[0].endswith("p3.tbaw")
    assert sum(1 for p in recent if p.endswith("p3.tbaw")) == 1
    assert settings["last_project"].endswith("p3.tbaw")


def test_clear_recent_empties_list_but_keeps_last_project(tmp_path):
    settings = {}
    project_io.remember_recent(settings, str(tmp_path / "a.tbaw"))
    project_io.clear_recent(settings)
    assert settings["recent_projects"] == []
    assert settings["last_project"].endswith("a.tbaw")


def test_new_document_inherits_characters_as_copies():
    alice = Character.from_preset_dict("Alice", {"voice": "af_heart"})
    previous = Document.from_plain_text("old text", characters=[alice])

    fresh = project_io.new_document_from(previous)

    assert fresh.text == ""
    assert [c.name for c in fresh.characters] == ["Alice"]
    assert fresh.characters[0] is not alice
    assert fresh.tracks[0].character_id == fresh.characters[0].id


# -- project.py: the bundle without an app -------------------------------------------


def _document_with_audio(project_dir, text="hello", key="abc"):
    generated = os.path.join(project_dir, "audio", "generated")
    os.makedirs(generated, exist_ok=True)
    seg = os.path.join(generated, f"{key}_0.wav")
    with open(seg, "wb") as f:
        f.write(b"RIFF" + b"\0" * 60)
    character = Character.from_preset_dict("A", {"voice": "af_heart", "fx_preset": "warm"})
    clip = Clip(character_id=character.id, segments=[Segment(0, text, key, seg, 1.5)])
    return Document(runs=[Run(text, clip.id, "generated")], clips=[clip], characters=[character],
                    settings={"x": 1}), seg


def test_bundle_round_trips_document_settings_audio_and_assets(tmp_path, isolated_dirs):
    fx_dir = tmp_path / "presets" / "fx"
    fx_dir.mkdir(parents=True)
    (fx_dir / "warm.json").write_text('{"gain_db": 2.0}', encoding="utf-8")
    project_dir, project_id = project_io.create_project_dir()
    doc, seg = _document_with_audio(project_dir)
    path = str(tmp_path / "proj.tbaw")

    result = project_io.save_project(doc, path, {"export": {"format": "flac"}}, project_dir, project_id,
                                     fx_presets_dir=str(fx_dir))

    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        manifest = json.loads(zf.read("manifest.json"))
        document = json.loads(zf.read("document.json"))
    assert names == {"manifest.json", "document.json", "project.json", "audio/generated/abc_0.wav", "fx/warm.json"}
    assert manifest["format"] == "tbaw" and manifest["version"] == 1 and manifest["requires"] == []
    assert manifest["project_id"] == project_id
    assert manifest["stats"] == {"clips": 1, "characters": 1, "duration_s": 1.5}
    assert manifest["assets"]["fx/warm.json"].startswith("sha256:")
    assert document["clips"][0]["segments"][0]["audio_path"] == "audio/generated/abc_0.wav"
    assert result.asset_index["fx/warm.json"][2] == manifest["assets"]["fx/warm.json"]
    assert project_io.read_session(project_dir)["saved_digest"] == result.saved_digest

    # A second extraction elsewhere reads back the same document with
    # absolute paths inside its own dir.
    info = project_io.inspect_bundle(path)
    other_dir = str(tmp_path / "other")
    project_io.extract_small(info, other_dir)
    project_io.extract_audio(info, other_dir)
    loaded = project_io.finish_open(info, other_dir)
    assert loaded.document.text == "hello"
    assert loaded.project_settings == {"export": {"format": "flac"}}
    assert loaded.document.clips[0].segments[0].audio_path == os.path.join(other_dir, "audio", "generated", "abc_0.wav")
    assert os.path.isfile(os.path.join(other_dir, "fx", "warm.json"))
    assert loaded.notices == []


def test_missing_audio_on_open_leaves_the_segment_pathless_and_says_so(tmp_path, isolated_dirs):
    project_dir, project_id = project_io.create_project_dir()
    doc, _seg = _document_with_audio(project_dir)
    path = str(tmp_path / "proj.tbaw")
    project_io.save_project(doc, path, {"bundle": {"include_generated_audio": False}}, project_dir, project_id)

    with zipfile.ZipFile(path) as zf:
        assert not any(n.startswith("audio/") for n in zf.namelist())
        assert json.loads(zf.read("manifest.json"))["includes"]["generated_audio"] is False
    info = project_io.inspect_bundle(path)
    other = str(tmp_path / "other")
    project_io.extract_small(info, other)
    loaded = project_io.finish_open(info, other)
    assert loaded.document.clips[0].segments[0].audio_path is None
    assert any("missing" in n for n in loaded.notices)
    assert loaded.document.dirty_clips() == [loaded.document.clips[0]]


def test_open_drops_an_audio_path_that_points_outside_the_project_dir(tmp_path, isolated_dirs):
    """A `document.json` is untrusted input: a segment naming a file
    elsewhere on the machine reads as missing rather than as that file,
    which the next Save would otherwise copy into the bundle."""
    secret = tmp_path / "secret.txt"
    secret.write_bytes(b"not audio")
    other = str(tmp_path / "other")
    os.makedirs(other)
    with open(os.path.join(other, "document.json"), "w", encoding="utf-8") as f:
        json.dump({"runs": [], "characters": [],
                   "clips": [{"id": "c1", "character_id": "a", "segments": [
                       {"order_index": 0, "text": "hi", "cache_key": "k", "audio_path": str(secret),
                        "duration": 1.0}]}]}, f)
    info = project_io.BundleInfo(path=str(tmp_path / "x.tbaw"), manifest={}, project_id="x", entries=[],
                                 audio_bytes=0, zip_size=0, zip_mtime=0.0)
    loaded = project_io.finish_open(info, other)
    assert loaded.document.clips[0].segments[0].audio_path is None
    assert any("missing" in n for n in loaded.notices)
    assert secret.read_bytes() == b"not audio"


def test_save_bundles_only_audio_inside_the_project_dir(tmp_path, isolated_dirs):
    project_dir, project_id = project_io.create_project_dir()
    doc, _seg = _document_with_audio(project_dir)
    outside = tmp_path / "elsewhere.wav"
    outside.write_bytes(b"RIFF" + b"\0" * 60)
    doc.clips[0].segments[0].audio_path = str(outside)
    path = str(tmp_path / "proj.tbaw")

    project_io.save_project(doc, path, {}, project_dir, project_id)

    with zipfile.ZipFile(path) as zf:
        assert not any(n.startswith("audio/") for n in zf.namelist())
        document = json.loads(zf.read("document.json"))
    assert document["clips"][0]["segments"][0]["audio_path"] == str(outside).replace("\\", "/")


def test_open_never_extracts_the_dirs_own_session_or_lock(tmp_path, isolated_dirs):
    path = str(tmp_path / "planted.tbaw")
    _write_bundle(path, _manifest(), {"session.json": b'{"dirty": true, "source_path": "/elsewhere"}',
                                      "lock": b"x", "session.json.tmp": b"{}"})
    info = project_io.inspect_bundle(path)
    project_dir = project_io.choose_project_dir(info.project_id, info.path)
    lock = project_io.ProjectLock(project_dir).acquire()
    try:
        project_io.extract_small(info, project_dir)
        loaded = project_io.finish_open(info, project_dir)
    finally:
        lock.release()
    assert loaded.document.text == ""
    session = project_io.read_session(project_dir)
    assert session["source_path"] == info.path and session["dirty"] is False
    assert not os.path.exists(os.path.join(project_dir, "session.json.tmp"))


def _write_bundle(path, manifest, extra_entries=None):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest))
        zf.writestr("document.json", json.dumps({"runs": [], "clips": [], "characters": []}))
        for name, data in (extra_entries or {}).items():
            zf.writestr(name, data)


def _manifest(**overrides):
    base = {"format": "tbaw", "version": 1, "requires": [], "project_id": "0123456789abcdef"}
    base.update(overrides)
    return base


def test_open_rejects_newer_version_and_unknown_requires_by_name(tmp_path, isolated_dirs):
    newer = str(tmp_path / "newer.tbaw")
    _write_bundle(newer, _manifest(version=2))
    with pytest.raises(project_io.ProjectError, match="newer KokoroGUI"):
        project_io.inspect_bundle(newer)

    needs = str(tmp_path / "needs.tbaw")
    _write_bundle(needs, _manifest(requires=["chapters"]))
    with pytest.raises(project_io.ProjectError, match="chapters"):
        project_io.inspect_bundle(needs)

    not_ours = str(tmp_path / "other.tbaw")
    _write_bundle(not_ours, {"format": "zip-of-things", "version": 1})
    with pytest.raises(project_io.ProjectError):
        project_io.inspect_bundle(not_ours)


@pytest.mark.parametrize("name", ["../escape.txt", "/abs/escape.txt", "C:evil.txt", "audio/../../up.txt"])
def test_open_rejects_zip_slip_and_drive_relative_entries(tmp_path, isolated_dirs, name):
    path = str(tmp_path / "bad.tbaw")
    _write_bundle(path, _manifest(), {name: b"x"})
    with pytest.raises(project_io.ProjectError, match="Refusing"):
        project_io.inspect_bundle(path)


def test_open_rejects_symlink_entries(tmp_path, isolated_dirs):
    path = str(tmp_path / "link.tbaw")
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("manifest.json", json.dumps(_manifest()))
        zf.writestr("document.json", "{}")
        info = zipfile.ZipInfo("audio/generated/link.wav")
        info.external_attr = (0o120777 << 16)
        zf.writestr(info, "../../etc/passwd")
    with pytest.raises(project_io.ProjectError, match="symlink"):
        project_io.inspect_bundle(path)


def test_unknown_entries_and_manifest_keys_survive_open_and_save(tmp_path, isolated_dirs):
    path = str(tmp_path / "future.tbaw")
    _write_bundle(path, _manifest(future_key={"a": 1}),
                  {"chapters/ch1.json": b'{"future": true}', "engines/neweng/model.bin": b"\x00\x01"})
    info = project_io.inspect_bundle(path)
    assert info.manifest["future_key"] == {"a": 1}
    project_dir = project_io.choose_project_dir(info.project_id, info.path)
    project_io.extract_small(info, project_dir)
    loaded = project_io.finish_open(info, project_dir)
    assert os.path.isfile(os.path.join(project_dir, "chapters", "ch1.json"))

    project_io.save_project(loaded.document, path, loaded.project_settings, project_dir, info.project_id,
                            known_engine_ids=("kokoro", "audio8", "dummy"))
    with zipfile.ZipFile(path) as zf:
        assert zf.read("chapters/ch1.json") == b'{"future": true}'
        assert zf.read("engines/neweng/model.bin") == b"\x00\x01"


def test_save_is_atomic_when_the_write_fails(tmp_path, isolated_dirs, monkeypatch):
    project_dir, project_id = project_io.create_project_dir()
    doc, _seg = _document_with_audio(project_dir)
    path = str(tmp_path / "proj.tbaw")
    project_io.save_project(doc, path, {}, project_dir, project_id)
    before = open(path, "rb").read()

    real_write = zipfile.ZipFile.write

    def _boom(self, *a, **k):
        raise OSError("disk on fire")

    monkeypatch.setattr(zipfile.ZipFile, "write", _boom)
    with pytest.raises(OSError):
        project_io.save_project(doc, path, {}, project_dir, project_id)
    monkeypatch.setattr(zipfile.ZipFile, "write", real_write)

    assert open(path, "rb").read() == before
    assert not os.path.exists(path + ".tmp")


def test_save_and_open_refuse_on_short_disk_before_writing(tmp_path, isolated_dirs, monkeypatch):
    project_dir, project_id = project_io.create_project_dir()
    doc, _seg = _document_with_audio(project_dir)
    path = str(tmp_path / "proj.tbaw")
    monkeypatch.setattr(project_io, "free_space", lambda _p: 10)
    with pytest.raises(project_io.ProjectError, match="free space"):
        project_io.save_project(doc, path, {}, project_dir, project_id)
    assert not os.path.exists(path) and not os.path.exists(path + ".tmp")

    monkeypatch.setattr(project_io, "free_space", lambda _p: 10 ** 12)
    project_io.save_project(doc, path, {}, project_dir, project_id)
    monkeypatch.setattr(project_io, "free_space", lambda _p: 10)
    info = project_io.inspect_bundle(path)
    with pytest.raises(project_io.ProjectError, match="free space"):
        project_io.check_free_space(project_io.projects_root(), info.audio_bytes, "open the project")


def test_save_deletes_nothing_in_the_project_dir(tmp_path, isolated_dirs):
    project_dir, project_id = project_io.create_project_dir()
    doc, seg = _document_with_audio(project_dir)
    orphan = os.path.join(project_dir, "audio", "generated", "orphan_0.wav")
    open(orphan, "wb").write(b"RIFF")
    project_io.save_project(doc, str(tmp_path / "p.tbaw"), {}, project_dir, project_id)
    assert os.path.isfile(orphan) and os.path.isfile(seg)


def test_close_time_gc_removes_orphans_and_keeps_a_takes_files(tmp_path, isolated_dirs):
    project_dir, _project_id = project_io.create_project_dir()
    doc, seg = _document_with_audio(project_dir)
    generated = os.path.dirname(seg)
    orphan = os.path.join(generated, "orphan_0.wav")
    take_file = os.path.join(generated, "take2_0.wav")
    marker = os.path.join(generated, "stale.reserved")
    for p in (orphan, take_file, marker):
        open(p, "wb").write(b"RIFF")
    other = Clip(segments=[Segment(0, "x", "take2", take_file, 0.5)])
    doc.clips.append(other)

    removed = project_io.gc_project_dir(project_dir, doc)

    assert sorted(os.path.basename(r) for r in removed) == ["orphan_0.wav", "stale.reserved"]
    assert os.path.isfile(seg) and os.path.isfile(take_file)


def test_eviction_keeps_only_the_last_project_dir_and_skips_dirty_and_locked(isolated_dirs):
    keep, _ = project_io.create_project_dir("keep")
    gone, _ = project_io.create_project_dir("gone")
    dirty, _ = project_io.create_project_dir("dirty")
    locked, _ = project_io.create_project_dir("locked")
    project_io.write_session(dirty, {"dirty": True, "source_path": None})
    lock = project_io.ProjectLock(locked).acquire()
    try:
        removed = project_io.evict_project_dirs(keep)
    finally:
        lock.release()
    assert removed == [gone]
    assert os.path.isdir(keep) and os.path.isdir(dirty) and os.path.isdir(locked)
    assert not os.path.exists(gone)


def test_sweep_removes_clean_dirs_whose_file_is_gone(isolated_dirs, tmp_path):
    orphan, _ = project_io.create_project_dir("orphan")
    project_io.write_session(orphan, {"dirty": False, "source_path": str(tmp_path / "moved.tbaw")})
    kept, _ = project_io.create_project_dir("kept")
    existing = tmp_path / "here.tbaw"
    existing.write_bytes(b"PK")
    project_io.write_session(kept, {"dirty": False, "source_path": str(existing)})
    assert project_io.sweep_orphan_dirs() == [orphan]
    assert os.path.isdir(kept)


@pytest.mark.parametrize("source", ["moved.tbaw", "C:moved.tbaw", "", 7])
def test_sweep_leaves_a_dir_whose_session_source_is_not_an_absolute_path(isolated_dirs, source):
    """A session the app wrote always has an absolute `source_path`; anything
    else is corrupt and must not drive a delete."""
    odd, _ = project_io.create_project_dir("odd")
    project_io.write_session(odd, {"dirty": False, "source_path": source})
    assert project_io.sweep_orphan_dirs() == []
    assert os.path.isdir(odd)


def test_second_open_of_a_locked_project_is_refused_and_the_lock_clears(isolated_dirs):
    project_dir, _ = project_io.create_project_dir()
    holder = project_io.ProjectLock(project_dir).acquire()
    assert project_io.is_locked(project_dir)
    with pytest.raises(project_io.ProjectLockedError):
        project_io.ProjectLock(project_dir).acquire()
    holder.release()
    assert not project_io.is_locked(project_dir)
    project_io.ProjectLock(project_dir).acquire().release()


def test_recovery_wipe_succeeds_with_the_lock_held(isolated_dirs):
    project_dir, _ = project_io.create_project_dir()
    holder = project_io.ProjectLock(project_dir).acquire()
    try:
        open(os.path.join(project_dir, "document.json"), "w").write("{}")
        os.makedirs(os.path.join(project_dir, "fx"))
        open(os.path.join(project_dir, "fx", "a.json"), "w").write("{}")
        project_io.wipe_project_dir(project_dir)
        assert os.listdir(project_dir) == ["lock"]
        assert holder.held
    finally:
        holder.release()


def test_choose_project_dir_keys_on_id_and_suffixes_a_clean_dir_of_another_path(isolated_dirs, tmp_path):
    pid = "feedfacefeedface"
    first = os.path.join(project_io.projects_root(), pid)
    assert project_io.choose_project_dir(pid, str(tmp_path / "a.tbaw")) == first
    os.makedirs(first)
    project_io.write_session(first, {"dirty": False, "source_path": str(tmp_path / "a.tbaw")})
    # Same file, moved through the file manager: still this dir (dirty or not).
    project_io.write_session(first, {"dirty": True, "source_path": str(tmp_path / "old-name.tbaw")})
    assert project_io.choose_project_dir(pid, str(tmp_path / "renamed.tbaw")) == first
    # A clean dir belonging to a Save As sibling gets out of the way.
    project_io.write_session(first, {"dirty": False, "source_path": str(tmp_path / "a.tbaw")})
    assert project_io.choose_project_dir(pid, str(tmp_path / "copy.tbaw")) == first + "-2"


def test_project_summary_reads_only_the_manifest_and_still_summarises_json(tmp_path, isolated_dirs, monkeypatch):
    project_dir, project_id = project_io.create_project_dir()
    doc, _ = _document_with_audio(project_dir)
    path = str(tmp_path / "s.tbaw")
    project_io.save_project(doc, path, {}, project_dir, project_id)

    reads = []
    real_read = zipfile.ZipFile.read
    monkeypatch.setattr(zipfile.ZipFile, "read", lambda self, name, *a: (reads.append(name), real_read(self, name, *a))[1])
    summary = project_io.project_summary(path)
    assert reads == ["manifest.json"]
    assert summary["clips"] == 1 and summary["characters"] == 1 and summary["duration_s"] == 1.5
    assert summary["path"] == os.path.abspath(path)

    legacy = str(tmp_path / "s.json")
    with open(legacy, "w", encoding="utf-8") as f:
        json.dump({"runs": [], "clips": [{}, {}], "characters": [{}, {}, {}]}, f)
    legacy_summary = project_io.project_summary(legacy)
    assert legacy_summary["clips"] == 2 and legacy_summary["characters"] == 3
    assert project_io.project_summary(str(tmp_path / "ghost.tbaw")) is None
    bad = tmp_path / "bad.tbaw"
    bad.write_bytes(b"not a zip")
    assert project_io.project_summary(str(bad)) is None


def test_fourth_backend_engine_version_reaches_keys_and_manifest(tmp_path, isolated_dirs):
    from kokoro_gui.engine.caching import segment_key
    from kokoro_gui.engines.base import BackendHooksMixin, EngineCapabilities

    class FourthBackend(BackendHooksMixin):
        id = "fourth"
        display_name = "Fourth"
        capabilities = EngineCapabilities()
        engine = None

        def engine_version(self):
            return "fourth-9.9"

        def get_voices(self, lang_code=None):
            return []

    backend = FourthBackend()
    config = {"voice": "v", "speed": 1.0, "lang_code": "a", "engine_id": "fourth"}
    assert segment_key("hi", config, backend) != segment_key("hi", {**config, "engine_id": "kokoro"}, backend)

    character = Character.from_preset_dict("F", {"voice": "v"}, backend_id="fourth")
    doc = Document(runs=[], clips=[], characters=[character])
    project_dir, project_id = project_io.create_project_dir()
    path = str(tmp_path / "f.tbaw")
    project_io.save_project(doc, path, {}, project_dir, project_id, backend_for=lambda _id: backend)
    with zipfile.ZipFile(path) as zf:
        assert json.loads(zf.read("manifest.json"))["engines"] == {"fourth": {"version": "fourth-9.9", "meta": {}}}


def test_stats_duration_is_the_sum_of_segment_durations():
    clip = Clip(segments=[Segment(0, "a", "k", None, 1.25), Segment(1, "b", "k", None, 0.5)])
    doc = Document(runs=[], clips=[clip, Clip()], characters=[Character.from_preset_dict("A", {})])
    assert project_io.project_stats(doc) == {"clips": 2, "characters": 1, "duration_s": 1.75}


@pytest.mark.slow
def test_entry_over_4gb_round_trips(tmp_path, isolated_dirs):
    """Zip64 is on by default; a book-length bundle crosses 4 GB."""
    project_dir, project_id = project_io.create_project_dir()
    generated = os.path.join(project_dir, "audio", "generated")
    os.makedirs(generated, exist_ok=True)
    big = os.path.join(generated, "big_0.wav")
    with open(big, "wb") as f:
        f.truncate(4 * 1024 ** 3 + 1024)
    clip = Clip(segments=[Segment(0, "x", "big", big, 1.0)])
    doc = Document(runs=[Run("x", clip.id, "generated")], clips=[clip])
    path = str(tmp_path / "big.tbaw")
    project_io.save_project(doc, path, {}, project_dir, project_id)
    info = project_io.inspect_bundle(path)
    assert info.audio_bytes == 4 * 1024 ** 3 + 1024


# -- app-level ----------------------------------------------------------------------------


def test_launch_starts_untitled_in_a_locked_project_dir(qt_app):
    assert qt_app.project_path is None
    assert qt_app.settings.get("last_project") is None
    assert os.path.isfile(os.path.join(qt_app.project_dir, "document.json"))
    assert project_io.is_locked(qt_app.project_dir)
    assert qt_app.project_dir.startswith(project_io.projects_root())
    assert qt_app.windowTitle() == "Untitled - KokoroGUI"


def test_save_as_then_open_restores_text_and_clips(qt_app, tmp_path):
    _type(qt_app.editor, "hello world")
    alice = qt_app.document.characters[0]
    clip = qt_app.document.assign_character_to_range(0, 5, alice.id)
    target = str(tmp_path / "story")

    _save_as(qt_app, target)

    assert qt_app.project_path.endswith("story.tbaw")
    assert os.path.exists(qt_app.project_path)
    assert qt_app.windowTitle().startswith("story")
    assert qt_app.settings["recent_projects"][0] == qt_app.project_path
    assert not qt_app.is_project_dirty()

    story_dir = qt_app.project_dir
    qt_app.new_project()
    assert qt_app.document.text == ""
    assert qt_app.editor.toPlainText() == ""
    assert qt_app.project_path is None
    assert qt_app.project_dir != story_dir

    _open(qt_app, target + ".tbaw")
    assert qt_app.document.text == "hello world"
    assert qt_app.editor.toPlainText() == "hello world"
    assert qt_app.document.clip_covering(2).id == clip.id
    assert qt_app.project_dir == story_dir  # keyed by project_id
    assert not qt_app.is_project_dirty()


def test_save_as_json_path_writes_a_tbaw(qt_app, tmp_path):
    _save_as(qt_app, str(tmp_path / "legacy.json"))
    assert qt_app.project_path.endswith("legacy.tbaw")
    assert zipfile.is_zipfile(qt_app.project_path)


def test_autosave_writes_the_project_dir_never_the_zip_and_dirty_is_by_digest(qt_app, tmp_path):
    path = _save_as(qt_app, str(tmp_path / "auto"))
    stamp = (os.path.getsize(path), os.path.getmtime(path))
    assert qt_app.is_project_dirty() is False

    _type(qt_app.editor, "typed later")
    qt_app.save_settings()

    with open(os.path.join(qt_app.project_dir, "document.json"), encoding="utf-8") as f:
        data = json.load(f)
    assert "".join(r["text"] for r in data["runs"]) == "typed later"
    assert (os.path.getsize(path), os.path.getmtime(path)) == stamp
    assert qt_app.is_project_dirty() is True
    assert project_io.read_session(qt_app.project_dir)["dirty"] is True
    assert qt_app.windowTitle() == "auto* - KokoroGUI"

    qt_app.save_project()
    qt_app.wait_for_project_io()
    assert qt_app.is_project_dirty() is False
    assert qt_app.windowTitle() == "auto - KokoroGUI"
    assert project_io.load_project(path).document.text == "typed later"


def test_open_followed_by_no_edit_leaves_dirty_false(qt_app, tmp_path):
    _type(qt_app.editor, "some text")
    path = _save_as(qt_app, str(tmp_path / "quiet"))
    qt_app.new_project()
    _open(qt_app, path)
    qt_app.save_settings()  # the trailing autosave after _switch_document
    assert qt_app.is_project_dirty() is False


def test_generated_clip_lands_in_the_project_dir_named_by_key_and_is_bundled(qt_app, tmp_path):
    clip = _generated_clip(qt_app)
    config = qt_app._assemble_clip_config(clip)
    assert config["out_dir"] == os.path.join(qt_app.project_dir, "audio", "generated")
    assert config["segment_naming"] == "cache_key"
    assert config["project_dir"] == qt_app.project_dir
    assert qt_app.document.dirty_clips() == []

    path = _save_as(qt_app, str(tmp_path / "gen"))
    with zipfile.ZipFile(path) as zf:
        audio = [n for n in zf.namelist() if n.startswith("audio/generated/")]
        assert audio == [f"audio/generated/{clip.segments[0].cache_key}_0.wav"]
        assert zf.getinfo(audio[0]).compress_type == zipfile.ZIP_STORED


def test_save_as_keeps_project_dir_and_audio_paths(qt_app, tmp_path):
    clip = _generated_clip(qt_app)
    before = clip.segments[0].audio_path
    project_dir = qt_app.project_dir
    _save_as(qt_app, str(tmp_path / "first"))
    _save_as(qt_app, str(tmp_path / "second"))
    assert qt_app.project_dir == project_dir
    assert clip.segments[0].audio_path == before
    assert project_io.read_session(project_dir)["source_path"].endswith("second.tbaw")
    assert os.path.isfile(str(tmp_path / "first.tbaw")) and os.path.isfile(str(tmp_path / "second.tbaw"))


def test_regenerate_save_undo_leaves_clip_dirty_rather_than_silent(qt_app, tmp_path):
    from kokoro_gui.daw.undo import TextEditCommand

    clip = _generated_clip(qt_app, text="hello world")
    old_segment_path = clip.segments[0].audio_path
    # A text edit snapshots the clip (segments included) for undo.
    qt_app.document.undo_stack.push(TextEditCommand(6, 5, 5, "hello WORLD"))
    assert qt_app.document.clip_text(clip) == "hello WORLD"
    # "Regenerate" the edited clip: a new file, the old one now an orphan.
    text = qt_app.document.clip_text(clip)
    key = qt_app.document.segment_key_fn(text, clip)
    new_path = os.path.join(os.path.dirname(old_segment_path), f"{key}_0.wav")
    with open(new_path, "wb") as f:
        f.write(open(old_segment_path, "rb").read())
    clip.segments = build_segments_from_results(key, [{"text": text, "path": new_path, "duration": 0.1,
                                                       "cache_key": key}])
    _save_as(qt_app, str(tmp_path / "undo"))
    assert os.path.isfile(old_segment_path)  # Save deletes nothing (TB11)

    qt_app.document.undo_stack.undo()
    restored = qt_app.document.get_clip(clip.id)
    assert restored.segments[0].audio_path == old_segment_path
    assert qt_app.document.dirty_clips() == []  # the file is still there
    os.remove(old_segment_path)  # what a close-time GC would have done
    assert [c.id for c in qt_app.document.dirty_clips()] == [clip.id]


def test_clean_close_gcs_orphans_keeps_last_project_dir_and_deletes_others(qt_app, tmp_path):
    clip = _generated_clip(qt_app)
    orphan = os.path.join(qt_app.project_dir, "audio", "generated", "orphan_0.wav")
    open(orphan, "wb").write(b"RIFF")
    other_dir, _ = project_io.create_project_dir("other")
    path = _save_as(qt_app, str(tmp_path / "keepme"))
    project_dir = qt_app.project_dir

    qt_app.close()

    assert os.path.isdir(project_dir)
    assert os.path.isfile(clip.segments[0].audio_path)
    assert not os.path.exists(orphan)
    assert not os.path.exists(other_dir)
    assert not project_io.is_locked(project_dir)
    assert qt_app.settings["last_project"] == path


def test_discard_on_close_deletes_the_project_dir(qt_app, tmp_path, monkeypatch):
    _save_as(qt_app, str(tmp_path / "d"))
    project_dir = qt_app.project_dir
    _type(qt_app.editor, "unsaved")
    monkeypatch.setattr(type(qt_app), "_ask_close_choice", lambda self: "discard")
    qt_app.close()
    assert not os.path.exists(project_dir)


def test_cancel_on_close_keeps_the_window_open(qt_app, tmp_path, monkeypatch):
    _save_as(qt_app, str(tmp_path / "c"))
    _type(qt_app.editor, "unsaved")
    monkeypatch.setattr(type(qt_app), "_ask_close_choice", lambda self: "cancel")
    qt_app.close()
    assert qt_app.project_dir is not None and project_io.is_locked(qt_app.project_dir)


def test_save_on_close_writes_the_bundle_then_closes(qt_app, tmp_path, monkeypatch):
    path = _save_as(qt_app, str(tmp_path / "s"))
    _type(qt_app.editor, "kept")
    monkeypatch.setattr(type(qt_app), "_ask_close_choice", lambda self: "save")
    qt_app.close()
    qt_app.wait_for_project_io()
    assert project_io.load_project(path).document.text == "kept"
    assert qt_app.project_dir is None


def test_recover_prompt_is_driven_by_dirty_and_found_by_id_after_a_rename(qt_app, tmp_path, monkeypatch):
    _type(qt_app.editor, "saved text")
    path = _save_as(qt_app, str(tmp_path / "crash"))
    project_dir = qt_app.project_dir
    _type(qt_app.editor, "saved text plus unsaved")
    qt_app.save_settings()
    assert project_io.read_session(project_dir)["dirty"] is True

    # Simulate a crash: the lock goes away, the dir stays dirty, the file is renamed.
    _simulate_crash(qt_app)
    renamed = str(tmp_path / "renamed.tbaw")
    os.rename(path, renamed)

    asked = []
    monkeypatch.setattr(type(qt_app), "_ask_recover_choice",
                        lambda self, session, info, pdir: (asked.append((session, pdir)), "keep")[1])
    _open(qt_app, renamed)
    assert asked and asked[0][1] == project_dir
    assert qt_app.project_dir == project_dir
    assert qt_app.document.text == "saved text plus unsaved"
    assert qt_app.is_project_dirty() is True


def test_recover_take_file_wipes_and_extracts_fresh(qt_app, tmp_path, monkeypatch):
    _type(qt_app.editor, "saved text")
    path = _save_as(qt_app, str(tmp_path / "crash2"))
    project_dir = qt_app.project_dir
    _type(qt_app.editor, "unsaved edits")
    qt_app.save_settings()
    _simulate_crash(qt_app)

    monkeypatch.setattr(type(qt_app), "_ask_recover_choice", lambda self, session, info, pdir: "take")
    _open(qt_app, path)
    assert qt_app.project_dir == project_dir
    assert qt_app.document.text == "saved text"
    assert qt_app.is_project_dirty() is False


def test_open_with_audio_runs_behind_is_busy_and_refuses_a_generate(qt_app, tmp_path, monkeypatch):
    _generated_clip(qt_app)
    path = _save_as(qt_app, str(tmp_path / "busy"))
    qt_app.new_project()

    import threading

    gate = threading.Event()
    real_extract = project_io.extract_audio

    def slow_extract(*args, **kwargs):
        gate.wait(5)
        return real_extract(*args, **kwargs)

    monkeypatch.setattr(project_io, "extract_audio", slow_extract)
    qt_app.open_project(path)
    assert qt_app.is_busy()
    assert qt_app.editor.isReadOnly()
    qt_app.on_generate_clicked()
    assert not qt_app.engine.generate_dirty_clips.called
    gate.set()
    qt_app.wait_for_project_io()
    assert not qt_app.is_busy()
    assert not qt_app.editor.isReadOnly()
    assert qt_app.document.clips and qt_app.document.dirty_clips() == []


def test_open_of_a_bundle_with_another_engine_version_is_clean_with_a_status_line(qt_app, tmp_path, monkeypatch):
    clip = _generated_clip(qt_app)
    path = _save_as(qt_app, str(tmp_path / "ver"))
    with zipfile.ZipFile(path) as zf:
        manifest = json.loads(zf.read("manifest.json"))
    assert manifest["engines"]["kokoro"]["version"] == qt_app.backend.engine_version()
    qt_app.new_project()

    monkeypatch.setattr(type(qt_app.backend), "engine_version", lambda self: "99.0-other")
    qt_app._install_segment_key_fn()
    _open(qt_app, path)
    assert qt_app.document.get_clip(clip.id) is not None
    assert qt_app.document.dirty_clips() == []
    assert "99.0-other" in qt_app.transport_dock.status_text()


def test_json_project_migrates_to_tbaw_adopting_matching_segments(qt_app, tmp_path):
    from kokoro_gui.engine.caching import effective_speed

    # A 4.0-preview project: one clean clip (legacy key, file present), one whose
    # key already disagreed, one whose file is gone.
    audio_dir = tmp_path / "audio_output"
    audio_dir.mkdir()
    character = Character.from_preset_dict("Old", {"voice": "af_heart"})
    doc = Document(runs=[], clips=[], characters=[character])
    texts = ["clean clip", "stale clip", "gone clip"]
    doc.text = "\n".join(texts)
    clips = []
    pos = 0
    for text in texts:
        clips.append(doc.assign_character_to_range(pos, pos + len(text), character.id))
        pos += len(text) + 1
    qt_app.document = doc  # so the app's generation config sees these characters
    qt_app._install_segment_key_fn()
    for clip, text in zip(clips, texts):
        config = qt_app._assemble_generation_config(clip)
        legacy = project_io.legacy_segment_key(text, config)
        src = audio_dir / f"{text.replace(' ', '_')}.wav"
        src.write_bytes(b"RIFF" + b"\0" * 40)
        key = legacy if text != "stale clip" else "stale-key"
        clip.segments = [Segment(0, text, key, str(src), 1.0)]
    os.remove(audio_dir / "gone_clip.wav")
    legacy_path = str(tmp_path / "old.json")
    project_io.save_json_project(doc, legacy_path, {"export": {"format": "ogg"}})
    qt_app.new_project()

    _open(qt_app, legacy_path)

    assert qt_app.project_path == str(tmp_path / "old.tbaw")
    assert os.path.isfile(legacy_path)  # left where it was
    assert zipfile.is_zipfile(qt_app.project_path)
    assert qt_app.project_settings == {"export": {"format": "ogg"}}
    assert qt_app.settings["last_project"] == qt_app.project_path
    assert not any(p.endswith("old.json") for p in qt_app.settings["recent_projects"])
    by_text = {qt_app.document.clip_text(c): c for c in qt_app.document.clips}
    clean = by_text["clean clip"]
    assert clean.segments[0].audio_path.startswith(qt_app.project_dir)
    assert os.path.isfile(clean.segments[0].audio_path)
    assert clean.segments[0].cache_key == qt_app.document.segment_key_fn("clean clip", clean)
    assert by_text["stale clip"].segments[0].audio_path is None
    assert by_text["gone clip"].segments[0].audio_path is None
    dirty_ids = {c.id for c in qt_app.document.dirty_clips()}
    assert dirty_ids == {by_text["stale clip"].id, by_text["gone clip"].id}


def test_json_project_in_unwritable_dir_falls_through_to_save_as(qt_app, tmp_path, monkeypatch):
    doc = Document.from_plain_text("ro", characters=[Character.from_preset_dict("A", {})])
    legacy_path = str(tmp_path / "ro.json")
    project_io.save_json_project(doc, legacy_path, {})
    monkeypatch.setattr(os, "access", lambda p, mode: False)
    elsewhere = str(tmp_path / "elsewhere" / "moved")
    monkeypatch.setattr(type(qt_app), "_save_as_path_dialog", lambda self: project_io.bundle_path_for(elsewhere))
    _open(qt_app, legacy_path)
    assert qt_app.project_path == elsewhere + ".tbaw"
    assert os.path.isfile(qt_app.project_path)


def test_project_local_asset_shadows_the_global_one(qt_app, tmp_path, monkeypatch):
    import kokoro_engine

    global_dir = tmp_path / "custom_voices"
    global_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(kokoro_engine, "CUSTOM_VOICES_DIR", str(global_dir))
    (global_dir / "Mix.pt").write_bytes(b"global")
    local_dir = os.path.join(qt_app.project_dir, "engines", "kokoro", "voices")
    os.makedirs(local_dir)
    with open(os.path.join(local_dir, "Mix.pt"), "wb") as f:
        f.write(b"project-local")

    assert qt_app.backend.resolve_voice_file("Mix", qt_app.project_dir) == os.path.join(local_dir, "Mix.pt")
    assert [v.id for v in qt_app.backend.get_voices()] == ["Mix"]
    assert qt_app.backend.project_dir == qt_app.project_dir

    # And a Save bundles the project copy.
    qt_app.document.characters[0].preset_data["voice"] = "Mix"
    path = _save_as(qt_app, str(tmp_path / "shadow"))
    with zipfile.ZipFile(path) as zf:
        assert zf.read("engines/kokoro/voices/Mix.pt") == b"project-local"


def test_bundle_toggles_live_in_the_export_dialog_and_feed_the_clip_config(qt_app, tmp_path):
    from kokoro_gui.qt.docks.export_dialog import ExportDialog, run_export

    dialog = ExportDialog(qt_app)
    assert dialog.bundle_audio_check.isChecked() is True
    assert dialog.bundle_format_combo.currentText() == "wav"
    dialog.bundle_audio_check.setChecked(False)
    dialog.bundle_format_combo.setCurrentText("flac")
    run_export(qt_app, dialog.values(), bundle=dialog.bundle_values())  # no clips: nothing scheduled

    assert qt_app.project_settings["bundle"] == {
        "include_generated_audio": False, "include_imported_audio": True, "audio_format": "flac",
    }
    clip = _generated_clip(qt_app)
    assert qt_app._assemble_clip_config(clip)["format"] == "flac"
    path = _save_as(qt_app, str(tmp_path / "toggles"))
    with zipfile.ZipFile(path) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["audio"]["format"] == "flac"
        assert manifest["includes"]["generated_audio"] is False
        assert not any(n.startswith("audio/") for n in zf.namelist())
    assert project_io.load_project(path).project_settings["bundle"]["audio_format"] == "flac"


def test_recent_menu_lists_projects_and_opens_them(qt_app, tmp_path):
    _save_as(qt_app, str(tmp_path / "one"))
    _save_as(qt_app, str(tmp_path / "two"))

    texts = [a.text() for a in qt_app.recent_menu.actions()]
    assert texts[:2] == ["two", "one"]

    next(a for a in qt_app.recent_menu.actions() if a.text() == "one").trigger()
    qt_app.wait_for_project_io()
    assert qt_app.project_path.endswith("one.tbaw")


def test_new_project_inherits_characters_and_clears_selection(qt_app):
    bob = Character.from_preset_dict("Bob", {})
    qt_app.document.characters.append(bob)
    _type(qt_app.editor, "hello")
    qt_app.document.assign_character_to_range(0, 5, bob.id)
    qt_app.editor.rehighlight()
    qt_app.selection.select_clip(qt_app.document.clips[0].id)

    qt_app.new_project()

    assert [c.name for c in qt_app.document.characters] == ["Default", "Bob"]
    assert qt_app.document.clips == []
    assert qt_app.selection.kind == "none"
    assert qt_app.transcript_dock.character_combo.findData(qt_app.document.characters[1].id) >= 0


def test_open_missing_project_warns_and_forgets_it(qt_app, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    warned = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: warned.append(a)))
    ghost = str(tmp_path / "ghost.tbaw")
    project_io.remember_recent(qt_app.settings, ghost)
    qt_app._rebuild_recent_menu()

    qt_app.open_project(ghost)

    assert warned
    assert not any(p.endswith("ghost.tbaw") for p in qt_app.settings["recent_projects"])


def test_import_text_add_inserts_at_caret_on_native_undo(qt_app, tmp_path):
    _type(qt_app.editor, "start ")
    qt_app.engine.extract_text_from_file.return_value = "imported words"
    src = tmp_path / "in.txt"
    src.write_text("imported words", encoding="utf-8")

    qt_app.import_text(str(src), target="add")

    assert qt_app.document.text == "start imported words"
    assert qt_app.editor.toPlainText() == "start imported words"
    qt_app.undo()
    assert qt_app.document.text == "start "


def test_import_text_new_starts_a_fresh_project_with_the_text(qt_app, tmp_path):
    _type(qt_app.editor, "old")
    qt_app.engine.extract_text_from_file.return_value = "chapter one"
    src = tmp_path / "in.txt"
    src.write_text("chapter one", encoding="utf-8")

    qt_app.import_text(str(src), target="new")

    assert qt_app.document.text == "chapter one"
    assert qt_app.project_path is None


@pytest.mark.skipif(sys.platform != "win32", reason="the replace retry is a Windows behaviour")
def test_replace_retries_on_permission_error(tmp_path, monkeypatch):
    calls = []
    real_replace = os.replace

    def flaky(src, dst):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError("held by a scanner")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    monkeypatch.setattr(project_io.time, "sleep", lambda _s: None)
    src = tmp_path / "a.tmp"
    src.write_bytes(b"x")
    project_io._replace_with_retries(str(src), str(tmp_path / "a"))
    assert len(calls) == 3 and (tmp_path / "a").exists()


# --- phase 2: parked takes and variants in the bundle ---------------------------


def test_a_bundle_with_parked_takes_round_trips_them_and_requires_takes(tmp_path, isolated_dirs):
    project_dir, project_id = project_io.create_project_dir()
    doc, _seg = _document_with_audio(project_dir)
    parked = os.path.join(project_dir, "audio", "generated", "old_0.wav")
    with open(parked, "wb") as f:
        f.write(b"RIFF" + b"\0" * 60)
    doc.clips[0].takes = {0: [Segment(0, "hello", "old", parked, 2.0)]}
    doc.clips[0].overrides["take"] = 1
    path = str(tmp_path / "takes.tbaw")

    project_io.save_project(doc, path, {}, project_dir, project_id, fx_presets_dir=str(tmp_path / "none"))

    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        manifest = json.loads(zf.read("manifest.json"))
    assert "audio/generated/old_0.wav" in names
    assert manifest["requires"] == ["takes"]

    info = project_io.inspect_bundle(path)
    other_dir = str(tmp_path / "other")
    project_io.extract_small(info, other_dir)
    project_io.extract_audio(info, other_dir)
    loaded = project_io.finish_open(info, other_dir)
    take = loaded.document.clips[0].takes[0][0]
    assert take.audio_path == os.path.join(other_dir, "audio", "generated", "old_0.wav")
    assert take.duration == 2.0


def test_a_bundle_without_takes_requires_nothing(tmp_path, isolated_dirs):
    project_dir, project_id = project_io.create_project_dir()
    doc, _seg = _document_with_audio(project_dir)
    path = str(tmp_path / "plain.tbaw")
    project_io.save_project(doc, path, {}, project_dir, project_id, fx_presets_dir=str(tmp_path / "none"))
    with zipfile.ZipFile(path) as zf:
        assert json.loads(zf.read("manifest.json"))["requires"] == []


def test_close_time_gc_keeps_a_parked_takes_files(tmp_path, isolated_dirs):
    project_dir, _project_id = project_io.create_project_dir()
    doc, seg = _document_with_audio(project_dir)
    parked = os.path.join(os.path.dirname(seg), "old_0.wav")
    open(parked, "wb").write(b"RIFF")
    doc.clips[0].takes = {0: [Segment(0, "hello", "old", parked, 1.0)]}

    assert project_io.gc_project_dir(project_dir, doc) == []
    assert os.path.isfile(parked)


def test_used_voice_names_include_every_variant_reference():
    character = Character.from_preset_dict("A", {"voice": "calm_ref"}, backend_id="audio8")
    character.variants = {"angry": "angry_ref", "whisper": "soft_ref"}
    doc = Document(characters=[character])
    assert project_io.used_voice_names(doc) == {"audio8": {"calm_ref", "angry_ref", "soft_ref"}}
