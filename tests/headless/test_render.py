"""headless/render.py: a script to a `.tbaw` and a mixdown with Dummy and a
fake engine, the checks before anything is generated, and what a failure or a
cancel leaves behind at each stage. No model and no network."""
import json
import os
import socket
import sys
import threading
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from kokoro_gui.audio import post
from kokoro_gui.core import keys as core_keys
from kokoro_gui.core import render as core_render
from kokoro_gui.daw import arrangement, library, mixdown, revision
from kokoro_gui.engine import asr, runtime
from kokoro_gui.engine.runner import EngineRunner, ModelBase
from kokoro_gui.engines import registry
from kokoro_gui.engines.base import (
    BackendHooksMixin, ConfigField, ConfigFieldType, EngineCapabilities, Synthesis, VoiceInfo, common_fields,
)
from kokoro_gui.headless import render as render_module
from kokoro_gui.headless.errors import HeadlessError
from kokoro_gui.headless.host import ALLOW_DOWNLOAD_ENV, Host
from kokoro_gui.headless.render import render_script
from kokoro_gui.qt import project as project_io

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = "The door opens.\n\n[Sam]: Who is there?\n\n[Ann]: Only me, with the post."
VOICES = {"a": ["v1", "v2"], "b": ["w1"]}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """No network. Loopback passes: asyncio's self-pipe uses it on Windows."""
    real = socket.socket.connect
    attempts = []

    def connect(self, address):
        if isinstance(address, tuple) and address and address[0] in ("127.0.0.1", "::1", "localhost"):
            return real(self, address)
        attempts.append(address)
        raise OSError(f"network: {address}")

    monkeypatch.setattr(socket.socket, "connect", connect, raising=False)
    monkeypatch.delenv(ALLOW_DOWNLOAD_ENV, raising=False)
    yield
    assert attempts == []


@pytest.fixture
def fake(isolated_dirs):
    """An engine "fake" whose load and speak run the test's functions. Speech
    is 2400 frames a word, and `words` decides whether it has word times."""
    control = SimpleNamespace(load=None, speak=None, words=True, spoken=[], voices=VOICES, closed=0)

    class Model(ModelBase):
        engine_id = "fake"
        sample_rate = 24000

        def load(self, lang_code, device):
            if control.load:
                control.load()
            return True

        def synthesize(self, text, voice, speed, lang_code, params):
            control.spoken.append((text, voice, speed, lang_code))
            if control.speak:
                control.speak(text)
            spoken = text.split()
            t = np.arange(2400 * len(spoken)) / 24000.0
            words = [[w, i * 0.1, (i + 1) * 0.1] for i, w in enumerate(spoken)] if control.words else []
            return Synthesis((0.3 * np.sin(2 * np.pi * 330 * t)).astype(np.float32), words)

        def engine_version(self):
            return "1"

    class Adapter(BackendHooksMixin):
        id = "fake"
        display_name = "Fake"
        capabilities = EngineCapabilities()

        def __init__(self):
            self._engine = EngineRunner(Model())

        @property
        def engine(self):
            return self._engine

        @classmethod
        def get_config_schema(cls):
            return [ConfigField("lang_code", "Language", ConfigFieldType.CHOICE, default="a", group="Generation"),
                    ConfigField("voice", "Voice", ConfigFieldType.CHOICE, default="v1", group="Generation"),
                    *common_fields(caching_default=False)]

        def builtin_voices(self, lang_code=None):
            return [VoiceInfo(id=name, display_name=name, lang_code=code) for code, names in control.voices.items()
                    for name in names if lang_code in (None, code)]

        def cancel(self):
            self._engine.cancel()

        def stop(self):
            control.closed += 1
            super().stop()

    registry.register_engine("fake", Adapter, display_name="Fake")
    yield control
    registry.unregister_engine("fake")


@pytest.fixture
def presets(tmp_path):
    folder = tmp_path / "presets" / "fx"
    folder.mkdir(parents=True)
    (folder / "Radio.json").write_text(json.dumps({"highpass_enabled": True, "highpass_freq": 900}))
    return folder


def files_under(folder):
    return sorted(os.path.relpath(os.path.join(d, f), folder).replace("\\", "/")
                  for d, _dirs, names in os.walk(folder) for f in names)


def working_dirs():
    root = project_io.projects_root()
    return [os.path.join(root, name) for name in os.listdir(root)] if os.path.isdir(root) else []


def reopen(path):
    """The bundle from scratch: its working folder goes first."""
    info = project_io.inspect_bundle(path)
    project_dir = os.path.join(project_io.projects_root(), info.project_id)
    project_io.delete_project_dir(project_dir)
    os.makedirs(project_dir)
    project_io.extract_small(info, project_dir)
    project_io.extract_audio(info, project_dir)
    return project_io.finish_open(info, project_dir)


def keys_of(path):
    with zipfile.ZipFile(path) as zf:
        data = json.loads(zf.read("document.json"))
    return [s["cache_key"] for clip in data["clips"] for s in clip["segments"]]


def failing(script=SCRIPT, **kwargs):
    if "host" not in kwargs:
        kwargs.setdefault("engine", "fake")
    with pytest.raises(HeadlessError) as caught:
        render_script(script, **kwargs)
    return caught.value


def assert_nothing_left(tmp_path, working_folder):
    """No file or folder at the output, no lock, no samples kept."""
    assert not (tmp_path / "audio_output").exists()
    folders = working_dirs()
    assert len(folders) == (1 if working_folder else 0)
    assert not any(project_io.is_locked(d) for d in folders)
    assert post._RENDER_CACHE == {}


# -- what a render makes -----------------------------------------------------------


def test_the_bundle_is_valid_and_reopens_with_every_clip_generated(fake):
    result = render_script(SCRIPT, engine="fake", cast={"Sam": "v2", "Ann": {"voice": "v1", "speed": 1.5}})
    assert (result.clips, result.regenerated, result.engine, result.channels) == (3, 3, "fake", 1)
    assert result.speakers == ["Narrator", "Sam", "Ann"] and result.warnings == []
    assert files_under(result.out_dir) == sorted(map(os.path.basename, (result.project_path, result.mixdown_path)))

    info = project_io.inspect_bundle(result.project_path)
    assert info.manifest["requires"] == [] and info.manifest["engines"]["fake"]["version"] == "1"
    loaded = reopen(result.project_path)
    assert loaded.notices == []
    document = loaded.document
    assert document.text == "The door opens.\n\nWho is there?\n\nOnly me, with the post."
    assert all(clip.status == "generated" and clip.segments for clip in document.clips)
    assert all(os.path.isfile(s.audio_path) for clip in document.clips for s in clip.segments)
    with Host("fake") as host:
        again = render_module._Render(host, host.backend(), document, [], title="", poll=None, on_progress=None)
        again.project_dir = loaded.project_dir
        core_keys.install_key_fn(document, again._backend_for, again._generation_config, again._generation_config)
        assert document.dirty_clips() == []


def test_the_working_folder_stays_unlocked_and_open_reuses_it(fake):
    result = render_script(SCRIPT, engine="fake")
    info = project_io.inspect_bundle(result.project_path)
    (folder,) = working_dirs()
    assert os.path.basename(folder) == info.project_id and not project_io.is_locked(folder)
    session = project_io.read_session(folder)
    assert session["source_path"] == result.project_path and session["dirty"] is False
    assert project_io.open_action(session, info, folder, []) == "reuse"
    loaded = project_io.load_project_dir(folder)
    assert all(clip.segments and os.path.isfile(clip.segments[0].audio_path) for clip in loaded.document.clips)
    assert session["saved_digest"] == project_io.project_digest(loaded.document, {}, folder)


def test_the_mixdown_is_as_long_as_its_clips_at_gap_zero(fake):
    result = render_script(SCRIPT, engine="fake")
    document = reopen(result.project_path).document
    total = sum(s.duration for clip in document.clips for s in clip.segments)
    info = sf.info(result.mixdown_path)
    assert total == pytest.approx(1.1) and info.frames == round(total * 24000)
    assert result.duration_s == pytest.approx(total) and result.sample_rate == info.samplerate == 24000
    samples, _rate = sf.read(result.mixdown_path)
    start = 0
    for clip in document.clips:
        raw, _rate = sf.read(clip.segments[0].audio_path)
        assert np.allclose(samples[start:start + len(raw)], raw, atol=2e-4)
        start += len(raw)


def test_a_pause_is_silence_before_the_next_clip(fake):
    result = render_script("One two.\n\n[pause:1.5]\n\nThree.", engine="fake")
    assert result.duration_s == pytest.approx(0.2 + 1.5 + 0.1)
    assert [w["start"] for w in result.words] == pytest.approx([0.0, 0.1, 1.7])


def test_mono_by_default_and_stereo_on_request(fake):
    assert sf.info(render_script(SCRIPT, engine="fake").mixdown_path).channels == 1
    stereo = render_script(SCRIPT, engine="fake", channels=2)
    assert stereo.channels == 2 and sf.info(stereo.mixdown_path).channels == 2


def test_words_are_in_the_mixdowns_seconds(fake):
    result = render_script("One two.\n\nThree.", engine="fake")
    assert result.words == [{"word": "One", "start": 0.0, "end": 0.1}, {"word": "two.", "start": 0.1, "end": 0.2},
                            {"word": "Three.", "start": 0.2, "end": 0.3}]
    json.dumps(result.to_json())


def test_a_repeated_paragraph_has_words_and_one_model_call(fake):
    result = render_script("Same line.\n\nSame line.", engine="fake")
    assert len(fake.spoken) == 1 and len(set(keys_of(result.project_path))) == 1
    assert [w["word"] for w in result.words] == ["Same", "line."] * 2
    assert result.words[2]["start"] == pytest.approx(0.2)


def test_each_landed_clip_moves_the_files_revision(fake):
    before = revision.FILES
    render_script(SCRIPT, engine="fake")
    assert revision.FILES - before == 3


def test_speed_and_pitch_in_the_cast_reach_the_engine_and_the_project(fake):
    result = render_script("[Sam]: Hi.", engine="fake", cast={"sam": {"voice": "v2", "speed": 2.0, "pitch": 12.0}})
    assert fake.spoken == [("Hi.", "v2", 1.0, "a")]  # 2.0, made up for the octave the resample adds
    assert result.duration_s == pytest.approx(0.05, abs=0.001)
    (sam,) = reopen(result.project_path).document.characters
    assert sam.preset_data == {"voice": "v2", "speed": 2.0, "pitch": 12.0} and sam.backend_id == "fake"


def test_title_names_the_files_and_the_project(fake):
    result = render_script(SCRIPT, engine="fake", title=" My  Book: one\n")
    assert os.path.basename(result.project_path) == "My-Book-one.tbaw"
    assert os.path.basename(result.mixdown_path) == "My-Book-one.wav"
    assert reopen(result.project_path).project_settings == {"title": "My Book: one"}


def test_without_a_title_the_first_words_name_the_files_and_nothing_is_stored(fake, tmp_path):
    result = render_script(SCRIPT, engine="fake")
    folder = os.path.relpath(result.out_dir, tmp_path).replace("\\", "/")
    assert folder.startswith("audio_output/headless/") and folder.endswith("_The-door-opens-Who-is-there-Only-me-with")
    assert reopen(result.project_path).project_settings == {}


def test_progress_is_reported_per_stage(fake):
    seen = []
    render_script(SCRIPT, engine="fake", on_progress=lambda *a: seen.append(a))
    assert [e for e in seen if e[0] == "generate"] == [("generate", n, 3) for n in (1, 2, 3)]
    assert ("mix", 100, 100) in seen and seen[-1] == ("save", 1, 1)


def test_a_progress_hook_that_raises_changes_nothing(fake):
    def boom(*_args):
        raise RuntimeError("hook")

    assert os.path.isfile(render_script(SCRIPT, engine="fake", on_progress=boom).mixdown_path)


def test_dummy_renders_without_a_model():
    result = render_script("[Sam]: Hello.", engine="dummy", cast={"Sam": "dummy"})
    assert result.engine == "dummy" and result.duration_s > 0.2 and len(result.words) == 1


def test_kokoro_renders_a_built_in_voice_and_one_of_the_users(isolated_dirs, fake_pipeline):
    (isolated_dirs.custom_voices / "mine.pt").write_bytes(b"voice")
    result = render_script("[Sam]: Hello.\n\n[Ann]: Bye.", engine="kokoro", cast={"Sam": "mine", "Ann": "bf_emma"})
    assert len(result.warnings) == 1 and "'bf_emma' is not a 'a' voice" in result.warnings[0]
    with zipfile.ZipFile(result.project_path) as zf:
        assert [n for n in zf.namelist() if n.startswith("engines/")] == ["engines/kokoro/voices/mine.pt"]
    assert "af_heart" in failing(engine="kokoro", cast={"Sam": "custom_voices/mine"}).valid


# -- the host ------------------------------------------------------------------------


def test_its_own_host_is_closed_and_a_given_one_stays_open(fake):
    render_script(SCRIPT, engine="fake")
    assert fake.closed == 1
    with Host("fake") as host:
        first = render_script(SCRIPT, host=host)
        second = render_script(SCRIPT, host=host)
        assert fake.closed == 1 and first.out_dir != second.out_dir
    assert fake.closed == 2


def test_its_own_host_is_closed_after_a_failure(fake):
    failing(cast={"Sam": "nobody"})
    assert fake.closed == 1


def test_the_render_memo_is_emptied(fake):
    render_script(SCRIPT, engine="fake")
    assert post._RENDER_CACHE == {}


# -- what is read ----------------------------------------------------------------------


def test_the_users_settings_characters_and_lexicon_are_not_read(fake, tmp_path, monkeypatch):
    plain = keys_of(render_script(SCRIPT, engine="fake").project_path)
    (tmp_path / "config_qt.json").write_text(json.dumps({
        "speed": 1.7, "pitch": 3.0, "volume": 0.2, "lexicon": {"door": "gate"},
        "engines": {"fake": {"voice": "v2", "lang_code": "b"}}}))
    os.makedirs(library.LIBRARY_DIR)
    with open(os.path.join(library.LIBRARY_DIR, "Sam.json"), "w") as f:
        json.dump({"name": "Sam", "preset_data": {"voice": "v2", "speed": 0.5}}, f)
    monkeypatch.setattr(arrangement, "recorded_chars_per_second", lambda *a, **k: pytest.fail("stats were read"))
    assert keys_of(render_script(SCRIPT, engine="fake").project_path) == plain


def test_settings_passed_in_give_the_lexicon_and_the_engine_values(fake):
    settings = {"lexicon": {"door": "gate"}, "engines": {"fake": {"voice": "w1", "lang_code": "b"}}, "speed": 1.7}
    result = render_script(SCRIPT, engine="fake", settings=settings)
    assert fake.spoken[0] == ("The gate opens.", "w1", 1.0, "b")
    assert reopen(result.project_path).document.text.startswith("The door opens.")


def test_settings_belong_to_the_host(fake):
    with Host("fake") as host:
        assert failing(host=host, settings={}).code == "bad_arguments"
    assert failing(settings=[]).code == "bad_arguments"


def test_fx_presets_are_not_looked_at_unless_one_is_named(fake, presets, monkeypatch):
    monkeypatch.setattr(render_module._FxPresets, "_stems", lambda self: pytest.fail("the folder was read"))
    render_script(SCRIPT, engine="fake")


# -- FX --------------------------------------------------------------------------------


def test_fx_changes_the_mixdown_and_no_key(fake, presets):
    dry = render_script(SCRIPT, engine="fake", cast={"Sam": "v1"})
    wet = render_script(SCRIPT, engine="fake", cast={"Sam": {"voice": "v1", "fx_preset": "Radio"}})
    assert keys_of(dry.project_path) == keys_of(wet.project_path)
    a, b = sf.read(dry.mixdown_path)[0], sf.read(wet.mixdown_path)[0]
    sam = np.zeros(len(a), dtype=bool)
    sam[int(0.3 * 24000):int(0.6 * 24000)] = True
    assert len(a) == len(b) and not np.allclose(a[sam], b[sam], atol=1e-2)
    assert np.array_equal(a[~sam], b[~sam])
    with zipfile.ZipFile(wet.project_path) as zf:
        assert "fx/Radio.json" in zf.namelist()


def test_a_tags_fx_preset_is_used_when_it_exists_and_a_warning_when_not(fake, presets):
    result = render_script("[Sam:Radio]: One.\n\n[Sam:Nope]: Two.", engine="fake", cast={"Sam": "v1"})
    assert result.warnings == ["Speaker 'Sam': FX preset 'Nope' does not exist; ignored."]
    document = reopen(result.project_path).document
    assert [c.overrides for c in document.clips] == [{"fx_preset": "Radio"}, {}]


@pytest.mark.parametrize("name", ["Nope", "radio", "../fx/Radio", "sub\\Radio", "C:Radio", "Broken", "Empty", "List"])
def test_a_cast_fx_preset_that_is_not_there_stops_the_render(fake, presets, tmp_path, name):
    (presets / "Broken.json").write_text("{")
    (presets / "Empty.json").write_text("{}")
    (presets / "List.json").write_text("[1]")
    error = failing(cast={"Sam": {"voice": "v1", "fx_preset": name}})
    assert (error.code, error.exit_code, error.valid) == ("bad_arguments", 2, ["Radio"])
    assert repr(name) in error.message and fake.spoken == []
    assert_nothing_left(tmp_path, working_folder=False)


def test_a_preset_linked_out_of_the_folder_does_not_count(fake, presets, tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"gain_db": 3}))
    try:
        os.symlink(outside, presets / "Link.json")
    except OSError:
        pytest.skip("no symlinks here")
    assert failing(cast={"Sam": {"voice": "v1", "fx_preset": "Link"}}).valid == ["Radio"]


# -- voices ----------------------------------------------------------------------------


@pytest.mark.parametrize("voice", ["nobody", "V1", "../v1", "v1.pt", "\\\\host\\share\\v1.wav"])
def test_a_voice_the_engine_does_not_list(fake, tmp_path, voice):
    error = failing(cast={"Sam": voice})
    assert (error.code, error.exit_code, error.valid) == ("unknown_voice", 4, ["v1", "v2", "w1"])
    assert "'Sam'" in error.message and fake.spoken == []
    assert_nothing_left(tmp_path, working_folder=False)


def test_a_voice_file_path_is_not_a_voice(fake, tmp_path):
    path = tmp_path / "voice.wav"
    sf.write(path, np.zeros(2400), 24000)
    assert failing(cast={"Sam": {"voice": str(path)}}).code == "unknown_voice"


def test_a_cast_entry_no_line_uses_is_checked_too(fake):
    assert failing("Hello.", cast={"Nobody": "nope"}).code == "unknown_voice"


def test_an_engine_with_no_voice_for_an_unmapped_speaker(fake):
    fake.voices = {}
    error = failing("Hello.")
    assert (error.code, error.valid) == ("unknown_voice", [])


def test_an_unmapped_speaker_gets_the_default_voice_and_a_warning(fake):
    result = render_script("[Sam]: Hi.", engine="fake")
    assert fake.spoken[0][1] == "v1"
    assert result.warnings == ["Speaker 'Sam' is not in the cast; using the default voice."]


def test_another_languages_voice_is_a_warning(fake):
    result = render_script("[Sam]: Hi.", engine="fake", cast={"Sam": "w1"})
    assert result.warnings == ["Speaker 'Sam': 'w1' is not a 'a' voice; it is read with that language's rules."]
    assert fake.spoken[0][1:] == ("w1", 1.0, "a")


# -- arguments ---------------------------------------------------------------------------


@pytest.mark.parametrize("script", ["", "  \n\n ", "[Sam]:", "[pause:2]", None, 7, b"hi"])
def test_a_script_with_nothing_to_speak(fake, tmp_path, script):
    error = failing(script)
    assert (error.code, error.exit_code) == ("bad_arguments", 2)
    assert_nothing_left(tmp_path, working_folder=False)


@pytest.mark.parametrize("kwargs", [{"channels": 3}, {"channels": True}, {"channels": "1"}, {"title": 4},
                                    {"cast": ["Sam"]}, {"cast": {"Sam": {"speed": "fast"}}},
                                    {"cast": {"Sam": {"voice": "v1", "speed": float("nan")}}}])
def test_bad_arguments(fake, tmp_path, kwargs):
    assert failing(**kwargs).code == "bad_arguments"
    assert_nothing_left(tmp_path, working_folder=False)


def test_an_engine_that_is_not_installed(isolated_dirs):
    error = failing(engine="nope")
    assert (error.code, error.exit_code) == ("engine_not_installed", 3)


# -- the output folder ---------------------------------------------------------------------


def test_out_is_a_folder_made_when_missing(fake, tmp_path):
    result = render_script(SCRIPT, engine="fake", out="a/b", title="Book")
    assert files_under(tmp_path / "a") == ["b/Book.tbaw", "b/Book.wav"]
    assert result.out_dir == os.path.realpath(tmp_path / "a" / "b")
    render_script(SCRIPT, engine="fake", out="a/b", title="Other")
    assert len(files_under(tmp_path / "a")) == 4


@pytest.mark.parametrize("out", ["..", "../elsewhere", "\\\\host\\share", "//host/share", "nul", "a:b", ""])
def test_out_outside_the_allowed_roots(fake, tmp_path, monkeypatch, out):
    inner = tmp_path / "inner"
    inner.mkdir()
    monkeypatch.chdir(inner)
    before = files_under(tmp_path)
    error = failing(out=out)
    assert (error.code, error.exit_code) == ("out_not_allowed", 2)
    assert fake.spoken == [] and files_under(tmp_path) == before and os.listdir(inner) == []


def test_an_absolute_out_elsewhere(fake, tmp_path, monkeypatch):
    inner = tmp_path / "inner"
    inner.mkdir()
    monkeypatch.chdir(inner)
    assert failing(out=str(tmp_path / "elsewhere")).code == "out_not_allowed"
    assert not (tmp_path / "elsewhere").exists()


@pytest.mark.parametrize("name", ["book.tbaw", "book.WAV", "taken"])
def test_out_that_is_a_file_name_or_a_file(fake, tmp_path, name):
    (tmp_path / "taken").write_text("mine")
    assert failing(out=name).code == "bad_arguments"
    assert (tmp_path / "taken").read_text() == "mine" and fake.spoken == []


@pytest.mark.parametrize("name", ["Book.tbaw", "Book.wav", "Book.tbaw.tmp"])
def test_a_file_already_there_is_never_replaced(fake, tmp_path, name):
    (tmp_path / "out").mkdir(exist_ok=True)
    (tmp_path / "out" / name).write_text("mine")
    error = failing(out="out", title="Book")
    assert error.code == "bad_arguments" and name in error.message
    assert (tmp_path / "out" / name).read_text() == "mine" and files_under(tmp_path / "out") == [name]
    assert fake.spoken == [] and working_dirs() == []


@pytest.mark.parametrize("name", ["Book.tbaw", "Book.wav"])
def test_a_file_that_appears_while_rendering_is_kept(fake, tmp_path, name):
    fake.speak = lambda text: (tmp_path / "out" / name).write_text("mine")
    assert failing(out="out", title="Book").code == "bad_arguments"
    assert (tmp_path / "out" / name).read_text() == "mine" and files_under(tmp_path / "out") == [name]


def test_a_failure_removes_the_folders_it_made_and_keeps_the_others(fake, tmp_path):
    (tmp_path / "mine").mkdir()
    fake.speak = lambda text: 1 / 0
    failing(out="mine/new/deep")
    assert (tmp_path / "mine").is_dir() and os.listdir(tmp_path / "mine") == []
    (tmp_path / "mine" / "keep.txt").write_text("x")
    failing(out="mine")
    assert files_under(tmp_path / "mine") == ["keep.txt"]


def test_an_output_folder_that_cannot_be_made(fake, monkeypatch):
    def denied(*_args, **_kwargs):
        raise PermissionError("no")

    monkeypatch.setattr(render_module.paths, "new_render_dir", denied)
    assert failing().code == "bad_arguments"
    assert working_dirs() == []


# -- failures per stage ------------------------------------------------------------------------


def disk_full(*_args, **_kwargs):
    raise OSError("disk full")


def not_downloaded(*_args):
    from huggingface_hub.errors import LocalEntryNotFoundError

    raise LocalEntryNotFoundError("weights.pth is not there")


def test_a_failure_while_building(fake, tmp_path, monkeypatch):
    monkeypatch.setattr(render_module, "build_document", lambda *a, **k: 1 / 0)
    error = failing()
    assert (error.code, error.exit_code) == ("generation_failed", 7)
    assert_nothing_left(tmp_path, working_folder=False)


def test_a_model_that_is_not_downloaded_leaves_nothing(fake, tmp_path):
    fake.load = not_downloaded
    error = failing()
    assert (error.code, error.exit_code) == ("model_not_downloaded", 5)
    assert_nothing_left(tmp_path, working_folder=False)


def test_a_file_missing_at_the_first_generate(fake, tmp_path):
    fake.speak = not_downloaded
    error = failing()
    assert error.code == "model_not_downloaded" and working_dirs()[0] in error.message
    assert_nothing_left(tmp_path, working_folder=True)


def test_a_failed_load(fake, tmp_path):
    def load():
        raise RuntimeError("no memory for it")

    fake.load = load
    error = failing()
    assert error.code == "generation_failed" and "no memory for it" in error.message
    assert_nothing_left(tmp_path, working_folder=False)


def test_a_held_lock_is_project_locked(fake, tmp_path, monkeypatch):
    monkeypatch.setattr(project_io, "new_project_id", lambda: "taken")
    folder, _id = project_io.create_project_dir()
    lock = project_io.ProjectLock(folder).acquire()
    try:
        error = failing()
        assert (error.code, error.exit_code) == ("project_locked", 6)
        assert lock.held and project_io.is_locked(folder)
    finally:
        lock.release()
    assert fake.spoken == [] and not (tmp_path / "audio_output").exists()


def test_the_working_folder_is_locked_while_rendering(fake):
    seen = []
    fake.speak = lambda text: seen.append(project_io.is_locked(working_dirs()[0]))
    render_script(SCRIPT, engine="fake")
    assert seen == [True] * 3


def test_a_first_autosave_that_fails(fake, tmp_path, monkeypatch):
    monkeypatch.setattr(project_io, "autosave_to_dir", disk_full)
    error = failing()
    assert error.code == "generation_failed" and "disk full" in error.message
    assert working_dirs()[0] in error.message
    assert_nothing_left(tmp_path, working_folder=True)


def test_a_paragraph_that_fails_is_named_and_the_others_stay_in_the_working_folder(fake, tmp_path):
    def speak(text):
        if text.startswith("Who"):
            raise RuntimeError("cannot say that")

    fake.speak = speak
    error = failing()
    assert error.code == "generation_failed" and "paragraph 2 of 3: cannot say that" in error.message
    cause = error
    while cause.__cause__ is not None:
        cause = cause.__cause__
    assert str(cause) == "cannot say that" and isinstance(cause, RuntimeError)
    assert_nothing_left(tmp_path, working_folder=True)
    document = project_io.load_project_dir(working_dirs()[0]).document
    assert [bool(c.segments) for c in document.clips] == [True, False, True]


def half_a_mixdown(path, *_args, **_kwargs):
    with open(path, "wb") as f:
        f.write(b"RIFF")
    raise OSError("disk full")


def out_of_memory(*_args, **_kwargs):
    raise MemoryError()


def half_a_bundle(plan, *_args, **_kwargs):
    with open(plan.path + ".tmp", "wb") as f:
        f.write(b"PK")
    raise OSError("disk full")


def whole_bundle_then_fail(*args, _real=project_io.write_bundle, **kwargs):
    _real(*args, **kwargs)
    raise OSError("disk full")


def held(*_args, **_kwargs):
    raise PermissionError("held")


STAGES = {
    "arrange": (render_module, "compute_arrangement", disk_full),
    "mix": (mixdown, "write_audio", half_a_mixdown),
    "mix, out of memory": (mixdown, "_mix_clips", out_of_memory),
    "save, the plan": (project_io, "plan_save", disk_full),
    "save, half written": (project_io, "write_bundle", half_a_bundle),
    "save, after the write": (project_io, "write_bundle", whole_bundle_then_fail),
    "save, the session": (project_io, "record_save", disk_full),
    "save, held": (project_io, "_replace_with_retries", held),
}


@pytest.mark.parametrize("stage", STAGES)
def test_a_failure_leaves_no_output_and_no_lock(fake, tmp_path, monkeypatch, stage):
    monkeypatch.setattr(*STAGES[stage])
    error = failing()
    assert (error.code, error.exit_code) == ("generation_failed", 7)
    assert "failed" in error.message and working_dirs()[0] in error.message
    assert ".tmp" not in error.message and error.__cause__ is not None
    assert_nothing_left(tmp_path, working_folder=True)


def test_a_failure_of_the_last_rename_takes_the_bundle_back(fake, tmp_path, monkeypatch):
    real = project_io._replace_with_retries

    def replace(src, dst):
        if dst.endswith(".wav"):
            raise PermissionError("held")
        real(src, dst)

    monkeypatch.setattr(project_io, "_replace_with_retries", replace)
    assert failing().code == "generation_failed"
    assert_nothing_left(tmp_path, working_folder=True)


def test_a_segment_that_cannot_be_read_back_is_a_failure(fake, tmp_path, monkeypatch):
    monkeypatch.setattr(mixdown, "_clip_samples", lambda *a, **k: None)
    assert "3 clip(s) couldn't be read" in failing().message
    assert_nothing_left(tmp_path, working_folder=True)


def test_the_next_render_on_the_same_host_works_after_a_failure(fake):
    fake.speak = lambda text: 1 / 0
    with Host("fake") as host:
        failing(host=host)
        fake.speak = None
        assert os.path.isfile(render_script(SCRIPT, host=host).mixdown_path)


# -- cancel per stage ----------------------------------------------------------------------------


def test_cancelled_before_anything(fake, tmp_path):
    error = failing(cancel=lambda: True)
    assert (error.code, error.exit_code) == ("interrupted", 130)
    assert_nothing_left(tmp_path, working_folder=False)


def test_cancelled_while_building(fake, tmp_path, monkeypatch):
    stop = threading.Event()
    real = render_module.build_document

    def build(*args, **kwargs):
        stop.set()
        return real(*args, **kwargs)

    monkeypatch.setattr(render_module, "build_document", build)
    assert failing(cancel=stop.is_set).code == "interrupted"
    assert_nothing_left(tmp_path, working_folder=False)


def test_cancelled_while_loading(fake, tmp_path):
    stop = threading.Event()
    fake.load = stop.set
    assert failing(cancel=stop.is_set).code == "interrupted"
    assert_nothing_left(tmp_path, working_folder=False)


def test_cancelled_while_generating(fake, tmp_path):
    speaking = threading.Event()
    told = threading.Event()
    polls = []

    def speak(text):
        speaking.set()
        told.wait(5)  # until the host has told the engine, so the next clip doesn't start

    def cancel():
        if speaking.is_set():
            polls.append(1)
        if len(polls) > 2:
            told.set()
        return speaking.is_set()

    fake.speak = speak
    error = failing(cancel=cancel)
    assert error.code == "interrupted" and working_dirs()[0] in error.message
    assert len(fake.spoken) == 1
    assert_nothing_left(tmp_path, working_folder=True)


@pytest.mark.parametrize("stage", ["generate", "mix", "save"])
def test_cancelled_when_a_stage_reports(fake, tmp_path, stage):
    stop = threading.Event()

    def on_progress(name, done, total):
        if name == stage and (done == total or stage == "mix"):
            stop.set()

    assert failing(on_progress=on_progress, cancel=stop.is_set).code == "interrupted"
    assert_nothing_left(tmp_path, working_folder=True)


def test_a_cancel_hook_that_raises_counts_as_a_cancel(fake, tmp_path):
    def cancel():
        if fake.spoken:
            raise RuntimeError("hook")
        return False

    assert failing(cancel=cancel).code == "interrupted"
    assert_nothing_left(tmp_path, working_folder=True)


@pytest.mark.parametrize("owner, name", [(mixdown, "write_audio"), (project_io, "write_bundle")],
                         ids=["mix", "save"])
def test_ctrl_c_is_interrupted_and_leaves_nothing(fake, tmp_path, monkeypatch, owner, name):
    def stop(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(owner, name, stop)
    assert failing().code == "interrupted"
    assert_nothing_left(tmp_path, working_folder=True)


def test_system_exit_passes_through_and_leaves_nothing(fake, tmp_path, monkeypatch):
    def stop(*_args, **_kwargs):
        raise SystemExit(3)

    monkeypatch.setattr(core_render, "mixdown", stop)
    with pytest.raises(SystemExit):
        render_script(SCRIPT, engine="fake")
    assert_nothing_left(tmp_path, working_folder=True)


# -- alignment -----------------------------------------------------------------------------------


@pytest.fixture
def whisper(monkeypatch):
    """Whisper as a stub: whether it is cached, and what it hears."""
    state = SimpleNamespace(cached=True, heard=[("the", 0.0, 0.1), ("door", 0.1, 0.2)], calls=[], error=None,
                            offline=[])

    def transcribe(path, engine_id="whisper"):
        state.calls.append((path, engine_id))
        state.offline.append(os.environ.get("HF_HUB_OFFLINE"))
        if state.error:
            raise state.error
        return state.heard

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(asr, "whisper_model_cached", lambda name=None: state.cached)
    monkeypatch.setattr(asr, "transcribe_wav_words", transcribe)
    return state


def test_align_adds_words_where_the_engine_gave_none(fake, whisper):
    fake.words = False
    result = render_script("The door.", engine="fake", align=True)
    assert [c[1] for c in whisper.calls] == ["whisper"]
    assert result.words == [{"word": "The", "start": 0.0, "end": 0.1}, {"word": "door.", "start": 0.1, "end": 0.2}]
    assert reopen(result.project_path).document.clips[0].segments[0].words == [["The", 0.0, 0.1], ["door.", 0.1, 0.2]]


def test_the_working_folder_holds_what_was_saved_words_too(fake, whisper):
    """Open may read its `document.json` instead of the bundle's."""
    fake.words = False
    render_script("The door.", engine="fake", align=True)
    (folder,) = working_dirs()
    document = project_io.load_project_dir(folder).document
    assert document.clips[0].segments[0].words
    assert project_io.read_session(folder)["saved_digest"] == project_io.project_digest(document, {}, folder)


def test_no_align_without_asking_or_with_the_engines_own_words(fake, whisper):
    render_script("The door.", engine="fake", align=True)
    fake.words = False
    assert render_script("A door.", engine="fake").words == []
    assert whisper.calls == []


def test_align_is_skipped_when_whisper_is_not_downloaded(fake, whisper):
    fake.words = whisper.cached = False
    result = render_script("The door.", engine="fake", align=True)
    assert whisper.calls == [] and result.words == []
    assert len(result.warnings) == 1 and "not aligned" in result.warnings[0]


def test_align_downloads_whisper_when_that_is_allowed(fake, whisper, monkeypatch):
    fake.words = whisper.cached = False
    assert render_script("The door.", engine="fake", align=True, allow_download=True).words
    with Host("fake", allow_download=True) as host:
        assert render_script("The door.", host=host, align=True).words
        assert not render_script("The door.", host=host, align=True, allow_download=False).words
    monkeypatch.setenv(ALLOW_DOWNLOAD_ENV, "1")
    assert render_script("The door.", engine="fake", align=True).words
    assert whisper.offline == [None] * 3


def test_align_holds_the_download_gate(fake, whisper):
    fake.words = False
    render_script("The door.", engine="fake", align=True)
    assert whisper.offline == ["1"] and "HF_HUB_OFFLINE" not in os.environ


def test_a_transcription_that_fails_is_a_warning(fake, whisper):
    fake.words = False
    whisper.error = RuntimeError("bad file")
    result = render_script("The door.", engine="fake", align=True)
    assert result.words == [] and "1 of 1" in result.warnings[0] and "bad file" in result.warnings[0]


def test_cancelled_while_aligning(fake, whisper, tmp_path):
    fake.words = False
    error = failing(align=True, cancel=lambda: bool(whisper.calls))
    assert error.code == "interrupted" and len(whisper.calls) == 1
    assert_nothing_left(tmp_path, working_folder=True)


# -- no writes elsewhere -----------------------------------------------------------------------------

_WATCH = []  # the running test's list of paths, when one is watching
_HOOKED = []


def _audit(event, args):
    if not _WATCH:
        return
    try:
        if event == "open" and isinstance(args[0], (str, bytes)):
            mode, flags = args[1], args[2] or 0
            writes = any(c in mode for c in "wax+") if isinstance(mode, str) else flags & (os.O_WRONLY | os.O_RDWR)
            if writes:
                _WATCH[0].append(os.fsdecode(args[0]))
        elif event in ("os.mkdir", "os.remove", "os.rmdir", "os.rename", "shutil.rmtree"):
            _WATCH[0].extend(os.fsdecode(a) for a in args[:2] if isinstance(a, (str, bytes)))
    except Exception:  # noqa: BLE001 - an audit hook must not break the process
        pass


def test_nothing_is_written_outside_tmp_path_or_into_the_segment_cache(fake, tmp_path):
    if not _HOOKED:  # a hook can't be removed, so one for the process
        sys.addaudithook(_audit)
        _HOOKED.append(True)

    def stamps():
        return {name: os.stat(os.path.join(ROOT, name)).st_mtime_ns for name in os.listdir(ROOT)
                if name != ".pytest_cache"}

    touched = []
    before = stamps()
    _WATCH.append(touched)
    try:
        render_script(SCRIPT, engine="fake", cast={"Sam": "v2"})
        failing(cast={"Sam": "nope"})
        fake.speak = lambda text: 1 / 0
        failing("Something new.")
    finally:
        _WATCH.clear()
    inside = os.path.normcase(os.path.realpath(tmp_path)) + os.sep
    outside = [p for p in touched if "__pycache__" not in p and os.path.basename(p).lower() != os.devnull
               and not os.path.normcase(os.path.realpath(p)).startswith(inside)]
    assert touched and outside == []
    assert stamps() == before
    assert [f for f in os.listdir(runtime.CACHE_DIR) if f != "projects"] == []
    assert not os.path.exists(runtime.STATS_FILE)
