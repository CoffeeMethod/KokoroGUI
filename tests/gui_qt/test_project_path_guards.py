"""Path guards in kokoro_gui/qt/project.py: a crafted `document.json` can
only name audio under the project dir's `audio/` (segments) or
`audio/imported/` (source track, recording sources, music beds), never the
dir's own bookkeeping (`session.json`, `lock`, `document.json`)."""
import json
import os
import zipfile

import numpy as np
import pytest
import soundfile as sf

from kokoro_gui.daw import serialization
from kokoro_gui.daw.models import Character, Clip, Document, Run, Segment
from kokoro_gui.daw.undo import ImportBedCommand
from kokoro_gui.qt import project as project_io

_PRIVATE_NAMES = ["session.json", "lock", "document.json"]


def _victim_session(project_dir):
    project_io.write_session(project_dir, {"source_path": "/home/victim/secret/Project.tbaw",
                                           "video_trusted": "/home/victim/Videos/private.mp4"})
    with open(os.path.join(project_dir, "lock"), "w", encoding="utf-8") as f:
        f.write("12345")


def _crafted_document(project_dir, tmp_path, named):
    """A document whose source track, recording source, music bed and
    generated segment all name `named`."""
    bed_src = tmp_path / "theme.wav"
    sf.write(str(bed_src), np.zeros(2400, dtype=np.float32), 24000)
    bed_path = project_io.import_audio_file(str(bed_src), project_dir)
    doc = Document()
    doc.characters.append(Character.from_preset_dict("Ann", {"voice": "af_heart"}))
    generated = Clip(character_id=doc.characters[0].id)
    generated.segments = [Segment(order_index=0, text="hi", cache_key="k", audio_path=named, duration=0.1)]
    recording = Clip(source="imported")
    doc.clips.extend([generated, recording])
    doc.runs = [Run(text="hi. ", clip_id=generated.id, kind="generated"),
                Run(text="there", clip_id=recording.id, kind="imported", words=[[0, 5, "s", 0.0, 0.5]])]
    doc.undo_stack.push(ImportBedCommand(bed_path, "theme", 0.0))
    doc.settings["sources"] = {"s": {"path": named, "sample_rate": 24000, "duration_s": 1.0}}
    doc.settings["source_track"] = {"path": named, "offset_s": 0.0}
    bed = next(c for c in doc.clips if c.is_bed)
    bed.original_audio_path = named
    return doc


@pytest.mark.parametrize("absolute", [False, True])
@pytest.mark.parametrize("name", _PRIVATE_NAMES)
def test_plan_save_never_bundles_the_project_dirs_own_files(tmp_path, isolated_dirs, name, absolute):
    project_dir, project_id = project_io.create_project_dir()
    _victim_session(project_dir)
    project_io.autosave_to_dir(Document(), {}, project_dir)
    named = os.path.join(project_dir, name) if absolute else name
    doc = _crafted_document(project_dir, tmp_path, named)

    assert project_io.source_track_path(doc, project_dir) is None
    plan, _warnings = project_io.plan_save(doc, {}, str(tmp_path / "out.tbaw"), project_dir, project_id,
                                           lambda _i: None, str(tmp_path / "fx"), project_io.read_session(project_dir))
    names = [n for n, _src in plan.audio_files]
    sources = {os.path.realpath(src) for _n, src in plan.audio_files}
    assert name not in names
    assert os.path.realpath(os.path.join(project_dir, name)) not in sources
    assert all(n.startswith("audio/") for n in names)

    path = str(tmp_path / "out.tbaw")
    project_io.save_project(doc, path, {}, project_dir, project_id)
    with zipfile.ZipFile(path) as zf:
        entries = zf.namelist()
    assert "session.json" not in entries and "lock" not in entries
    assert entries.count("document.json") == 1
    assert not any(n.endswith("/" + name) for n in entries)


@pytest.mark.parametrize("name", _PRIVATE_NAMES)
def test_open_reads_every_path_naming_a_private_file_as_missing(tmp_path, isolated_dirs, name):
    project_dir, _project_id = project_io.create_project_dir()
    doc = _crafted_document(project_dir, tmp_path, name)
    other = str(tmp_path / "other")
    os.makedirs(os.path.join(other, "audio", "imported"))
    for file_name in os.listdir(os.path.join(project_dir, "audio", "imported")):
        os.link(os.path.join(project_dir, "audio", "imported", file_name),
                os.path.join(other, "audio", "imported", file_name))
    _victim_session(other)
    data = serialization.document_to_dict(doc)
    with open(os.path.join(other, "document.json"), "w", encoding="utf-8") as f:
        json.dump(data, f)
    info = project_io.BundleInfo(path=str(tmp_path / "x.tbaw"), manifest={}, project_id="x", entries=[],
                                 audio_bytes=0, zip_size=0, zip_mtime=0.0)

    loaded = project_io.finish_open(info, other)

    document = loaded.document
    assert project_io.source_track_path(document, other) is None
    assert document.source_path("s") is None
    assert not any(c.is_bed for c in document.clips)
    assert all(c.original_audio_path is None for c in document.clips)
    assert all(s.audio_path is None for c in document.clips for s in c.segments)
    assert any("missing" in n for n in loaded.notices)
    assert any("source track is missing" in n for n in loaded.notices)


def test_a_segment_outside_audio_or_an_imported_path_outside_audio_imported_is_refused(tmp_path, isolated_dirs):
    project_dir, project_id = project_io.create_project_dir()
    stray = os.path.join(project_dir, "fx", "stray.wav")
    generated = os.path.join(project_dir, "audio", "generated", "gen.wav")
    for path in (stray, generated):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sf.write(path, np.zeros(240, dtype=np.float32), 24000)
    doc = _crafted_document(project_dir, tmp_path, stray)
    # A generated file is a fine segment but not an imported file.
    doc.settings["sources"]["s"]["path"] = generated
    doc.settings["source_track"]["path"] = "audio/generated/gen.wav"

    assert project_io.source_track_path(doc, project_dir) is None
    plan, _warnings = project_io.plan_save(doc, {}, str(tmp_path / "out.tbaw"), project_dir, project_id,
                                           lambda _i: None, str(tmp_path / "fx"))
    assert [n for n, _src in plan.audio_files] == []

    doc.clips[0].segments[0].audio_path = generated
    plan, _warnings = project_io.plan_save(doc, {}, str(tmp_path / "out.tbaw"), project_dir, project_id,
                                           lambda _i: None, str(tmp_path / "fx"))
    assert [n for n, _src in plan.audio_files] == ["audio/generated/gen.wav"]


def test_write_bundle_refuses_a_private_entry_or_source(tmp_path, isolated_dirs):
    project_dir, project_id = project_io.create_project_dir()
    _victim_session(project_dir)
    plan, _warnings = project_io.plan_save(Document(), {}, str(tmp_path / "out.tbaw"), project_dir, project_id,
                                           lambda _i: None, str(tmp_path / "fx"))
    session = os.path.join(project_dir, "session.json")
    plan.audio_files.append(("session.json", session))
    plan.audio_files.append(("lock", os.path.join(project_dir, "lock")))
    plan.audio_files.append(("audio/generated/x.wav", session))

    project_io.write_bundle(plan, [])

    with zipfile.ZipFile(plan.path) as zf:
        entries = zf.namelist()
        assert "session.json" not in entries and "lock" not in entries
        assert "audio/generated/x.wav" not in entries
