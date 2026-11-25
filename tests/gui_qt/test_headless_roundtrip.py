"""A project rendered headless opens in the app with nothing stale (H5 of the
headless plan): both compute the same config and key for every clip."""
import os

import kokoro_engine
import pytest

from kokoro_gui.headless.render import render_script
from kokoro_gui.qt import project as project_io

SCRIPT = ("The door opens.\n\n[Sam]: Who is there? It is late, and I was asleep.\n\n[pause:0.8]\n\n"
          "[Ann:Radio]: Only me, with the post.\n\n[Sam]: At this hour?")
CAST = {"Narrator": "dummy", "Sam": {"voice": "dummy", "speed": 1.2, "pitch": 2.0},
        "Ann": {"voice": "dummy", "speed": 0.9}}

RealKokoro = kokoro_engine.KokoroEngine  # `qt_app` swaps in a stub


@pytest.fixture
def rendered(qt_app, tmp_path):
    """Rendered with the app open, in its working directory."""
    folder = tmp_path / "presets" / "fx"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "Radio.json").write_text('{"highpass_enabled": true, "highpass_freq": 900}')
    return render_script(SCRIPT, engine="dummy", cast=CAST)


def _open(qt_app, path):
    qt_app.open_project(path)
    qt_app.wait_for_project_io()
    assert qt_app.project_path == path


@pytest.mark.parametrize("working_folder", ["kept", "deleted"])
def test_a_headless_project_opens_with_nothing_stale(qt_app, rendered, working_folder):
    if working_folder == "deleted":
        project_id = project_io.inspect_bundle(rendered.project_path).project_id
        project_io.delete_project_dir(os.path.join(project_io.projects_root(), project_id))
    _open(qt_app, rendered.project_path)
    document = qt_app.document
    assert [c.name for c in document.characters] == ["Narrator", "Sam", "Ann"]
    assert len(document.clips) == rendered.clips == 4
    assert document.dirty_clips() == []
    for clip in document.clips:
        assert all(os.path.isfile(s.audio_path) for s in clip.segments)
        assert qt_app.backend_for(clip).id == "dummy"
        assert clip.segments[0].cache_key == document.segment_key_fn(document.clip_text(clip), clip)
    assert not qt_app.is_project_dirty()


def test_the_app_places_the_clips_where_the_mixdown_has_them(qt_app, rendered):
    _open(qt_app, rendered.project_path)
    arrangement = qt_app.build_arrangement(qt_app.focus)
    assert arrangement.total_duration_s == pytest.approx(rendered.duration_s, abs=0.001)
    by_text = {qt_app.document.clip_text(p.clip): p for p in arrangement.placed}
    words = {w["word"]: w for w in rendered.words}
    assert by_text["Only me, with the post."].start_s == pytest.approx(words["Only"]["start"], abs=0.001)
    assert qt_app.post_config_for_clip(by_text["Only me, with the post."].clip)["highpass_enabled"] is True


def test_an_edit_makes_only_its_own_clip_stale(qt_app, rendered):
    _open(qt_app, rendered.project_path)
    clip = qt_app.document.clips[1]
    clip.overrides["speed"] = 1.5
    qt_app.document.touch()
    assert qt_app.document.dirty_clips() == [clip]


def test_the_open_app_is_untouched_by_a_render(qt_app, tmp_path):
    before = (qt_app.project_dir, qt_app.document, dict(qt_app.settings))
    render_script("Hello.", engine="dummy")
    assert (qt_app.project_dir, qt_app.document, dict(qt_app.settings)) == before
    assert project_io.is_locked(qt_app.project_dir)


def test_a_kokoro_project_opens_with_nothing_stale(qt_app, tmp_path, fake_pipeline, monkeypatch):
    stub = kokoro_engine.KokoroEngine
    monkeypatch.setattr(kokoro_engine, "KokoroEngine", RealKokoro)
    result = render_script("Nobody spoke.\n\n[Sam]: Hello there.\n\n[Ann]: Good morning.",
                           engine="kokoro", cast={"Sam": "am_adam", "Ann": {"voice": "af_bella", "speed": 0.8}})
    monkeypatch.setattr(kokoro_engine, "KokoroEngine", stub)
    assert [w["word"] for w in result.words] == ["Nobody", "spoke.", "Hello", "there.", "Good", "morning."]
    _open(qt_app, result.project_path)
    voices = [qt_app.document.get_character(c.character_id).preset_data["voice"] for c in qt_app.document.clips]
    assert voices == ["af_heart", "am_adam", "af_bella"]
    assert qt_app.document.dirty_clips() == []
