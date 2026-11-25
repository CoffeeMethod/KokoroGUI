"""headless/speak.py, and playback from `speak` and `render_script`: after, and
stream on engines with and without `supports_jit_streaming`."""
import io
import json
import os
import threading

import numpy as np
import pytest
import soundfile as sf

from kokoro_gui.audio import post
from kokoro_gui.engine import runtime
from kokoro_gui.engines.base import EngineCapabilities
from kokoro_gui.headless import render as render_module
from kokoro_gui.headless import speak as speak_module
from kokoro_gui.headless.errors import HeadlessError
from kokoro_gui.headless.host import Host
from kokoro_gui.headless.player import PLAYBACK_ENV, Player
from kokoro_gui.headless.render import render_script
from kokoro_gui.headless.scratch import speak_dir
from kokoro_gui.headless.speak import speak
from kokoro_gui.qt import project as project_io

from .support import SCRIPT, failing, files_under, working_dirs

pytestmark = pytest.mark.usefixtures("offline")

LONG = " ".join(f"Sentence number {n} has quite a few words in it to speak." for n in range(6))


@pytest.fixture(autouse=True)
def playable(monkeypatch):
    monkeypatch.delenv(PLAYBACK_ENV, raising=False)


@pytest.fixture
def player():
    made = Player()
    yield made
    made.close()


@pytest.fixture
def heard(headless_env):
    """The files played, with their samples at that moment."""
    played = []

    def play(path, blocking=False):
        played.append((path, sf.read(path)[0]))

    headless_env.play.side_effect = play
    return played


def wrong(text="Hello.", **kwargs):
    kwargs.setdefault("engine", "fake")
    with pytest.raises(HeadlessError) as caught:
        speak(text, **kwargs)
    return caught.value


# -- the file -------------------------------------------------------------------------


def test_a_phrase_becomes_a_wav_with_words(fake, tmp_path):
    result = speak("Hello there, you.", engine="fake", voice="v2")
    assert (result.engine, result.voice, result.sample_rate, result.cached) == ("fake", "v2", 24000, False)
    assert (result.played, result.reason, result.warnings) == (False, None, [])
    info = sf.info(result.path)
    assert (info.frames, info.channels) == (7200, 1) and result.duration_s == pytest.approx(0.3)
    assert [w["word"] for w in result.words] == ["Hello", "there,", "you."]
    assert result.words[2] == {"word": "you.", "start": 0.2, "end": 0.3}
    name = os.path.relpath(result.path, tmp_path).replace("\\", "/")
    assert name.startswith("audio_output/headless/") and name.endswith("_Hello-there-you/Hello-there-you.wav")
    json.dumps(result.to_json())


def test_a_long_text_sums_its_pieces(fake):
    result = speak(LONG, engine="fake")
    words = LONG.split()
    assert result.duration_s == pytest.approx(0.1 * len(words)) and len(fake.spoken) > 1
    assert [w["word"] for w in result.words] == words
    assert [w["start"] for w in result.words] == pytest.approx([0.1 * n for n in range(len(words))], abs=1e-3)


def test_speed_and_pitch(fake):
    result = speak("Hi you.", engine="fake", speed=2, pitch=12)
    assert fake.spoken == [("Hi you.", "v1", 1.0, "a")]
    assert result.duration_s == pytest.approx(0.1, abs=0.001) and result.words[1]["end"] == pytest.approx(0.1, abs=0.001)


def test_markdown_is_removed_unless_told_not_to(fake):
    speak("# Title\n\nSome **bold** text.\n\n```\ncode\n```", engine="fake")
    assert [s[0] for s in fake.spoken] == ["Title. Some bold text."]
    fake.spoken.clear()
    speak("a **b**", engine="fake", markdown=False)
    assert fake.spoken[0][0] == "a **b**"


def test_tags_and_pauses_are_not_read_out(fake):
    speak("[Sam]: Hello. [pause:2] Bye.", engine="fake", voice="v1")
    assert " ".join(s[0] for s in fake.spoken) == "Hello. Bye."


def test_settings_give_the_lexicon_and_the_engine_values(fake):
    speak("The door.", engine="fake", settings={"lexicon": {"door": "gate"}, "speed": 3,
                                               "engines": {"fake": {"voice": "w1", "lang_code": "b"}}})
    assert fake.spoken == [("The gate.", "w1", 1.0, "b")]


def test_out_is_a_new_wav(fake, tmp_path):
    result = speak("Hi.", engine="fake", out="a/hi.wav")
    assert result.path == os.path.realpath(tmp_path / "a" / "hi.wav") and files_under(tmp_path / "a") == ["hi.wav"]
    before = (tmp_path / "a" / "hi.wav").read_bytes()
    assert wrong("Other.", out="a/hi.wav").code == "bad_arguments"
    assert (tmp_path / "a" / "hi.wav").read_bytes() == before


@pytest.mark.parametrize("out, code", [("x.mp3", "bad_arguments"), ("x", "bad_arguments"), ("../x.wav", "out_not_allowed"),
                                       ("//host/x.wav", "out_not_allowed"), ("nul.wav", "out_not_allowed")])
def test_out_elsewhere_or_not_a_wav(fake, out, code):
    assert wrong(out=out).code == code
    assert fake.spoken == []


def test_save_false_writes_nothing_under_audio_output(fake, tmp_path):
    result = speak("Hello.", engine="fake", save=False)
    assert result.path is None and result.duration_s == pytest.approx(0.1) and len(result.words) == 1
    assert not (tmp_path / "audio_output").exists()


def test_the_bytes_go_to_a_file_object(fake, tmp_path):
    buffer = io.BytesIO()
    result = speak("Hello you.", engine="fake", save=False, wav_file=buffer)
    buffer.seek(0)
    samples, rate = sf.read(buffer)
    assert rate == 24000 and len(samples) == 4800 and result.path is None


def test_only_the_speak_cache_is_written(fake, tmp_path):
    speak("Hello.", engine="fake", save=False)
    assert files_under(tmp_path) == sorted(
        f for f in files_under(tmp_path) if f.startswith("cache/headless_speak/"))
    assert len(os.listdir(speak_dir())) == 2 and os.listdir(runtime.CACHE_DIR) == ["headless_speak"]
    assert post._RENDER_CACHE == {}


# -- the cache ----------------------------------------------------------------------------


@pytest.mark.parametrize("content", ["", "{", "[1]", '[["a"]]', '{"a": 1}', '[["a", "x", 2]]', "null", "[" * 5000])
def test_damaged_words_are_no_words(fake, content):
    first = speak("Hello you.", engine="fake", save=False)
    (sidecar,) = [f for f in os.listdir(speak_dir()) if f.endswith(".words.json")]
    with open(os.path.join(speak_dir(), sidecar), "w") as f:
        f.write(content)
    again = speak("Hello you.", engine="fake", save=False)
    assert again.cached and again.words == [] and again.duration_s == first.duration_s


def test_words_that_cannot_be_stored_do_not_fail_the_call(fake, monkeypatch):
    def full(*_args):
        raise OSError("disk full")

    monkeypatch.setattr(speak_module.os, "replace", full)
    with monkeypatch.context():
        result = speak("Hello.", engine="fake", save=False)
    assert len(result.words) == 1 and not any(f.endswith(".tmp") for f in os.listdir(speak_dir()))


# -- arguments and failures -----------------------------------------------------------------


@pytest.mark.parametrize("text", ["", "  \n", None, 4, "[pause:3]", "```\ncode\n```", "[Sam]:"])
def test_nothing_to_speak(fake, tmp_path, text):
    assert wrong(text).code == "bad_arguments"
    assert fake.spoken == [] and not (tmp_path / "audio_output").exists()


@pytest.mark.parametrize("kwargs", [{"speed": 0}, {"speed": "fast"}, {"pitch": 99}, {"pitch": float("nan")},
                                    {"speed": True}, {"play": "now"}, {"out": "a.wav", "save": False}, {"settings": []}])
def test_bad_arguments(fake, kwargs):
    error = wrong(**kwargs)
    assert error.code == "bad_arguments" and "cast" not in error.message


@pytest.mark.parametrize("voice", ["nobody", "../v1", "", 4])
def test_unknown_voice(fake, tmp_path, voice):
    error = wrong(voice=voice)
    assert (error.code, error.exit_code, error.valid) == ("unknown_voice", 4, ["v1", "v2", "w1"])
    assert fake.spoken == [] and not (tmp_path / "audio_output").exists()


def test_an_engine_that_is_not_there(isolated_dirs):
    assert wrong(engine="nope").code == "engine_not_installed"


def not_downloaded(*_args):
    from huggingface_hub.errors import LocalEntryNotFoundError

    raise LocalEntryNotFoundError("weights.pth")


@pytest.mark.parametrize("where", ["load", "speak"])
def test_a_missing_model(fake, tmp_path, where):
    setattr(fake, where, not_downloaded)
    assert wrong().code == "model_not_downloaded"
    assert not (tmp_path / "audio_output").exists()


def test_a_failed_generate_and_the_next_one(fake, tmp_path):
    fake.speak = lambda text: 1 / 0
    with Host("fake") as host:
        assert wrong(host=host, engine=None).code == "generation_failed"
        assert not (tmp_path / "audio_output").exists()
        fake.speak = None
        assert speak("Hello.", host=host).path


def test_a_failed_write_leaves_nothing(fake, tmp_path, monkeypatch):
    def full(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(project_io, "_replace_with_retries", full)
    assert wrong().code == "generation_failed"
    assert not (tmp_path / "audio_output").exists()


def test_cancelled_while_speaking(fake, tmp_path):
    assert wrong(LONG, cancel=lambda: bool(fake.spoken)).code == "interrupted"
    assert not (tmp_path / "audio_output").exists()


def test_the_host_is_closed_only_when_it_is_its_own(fake):
    speak("Hi.", engine="fake")
    assert fake.closed == 1
    with Host("fake") as host:
        speak("Hi.", host=host)
        assert fake.closed == 1


# -- playback: speak ------------------------------------------------------------------------------


def test_off_plays_nothing(fake, headless_env):
    speak("Hello.", engine="fake")
    assert headless_env.play.call_count == 0


def test_after_plays_the_file(fake, heard, player):
    result = speak("Hello.", engine="fake", play="after", player=player, wait_for_playback=True)
    assert (result.played, result.reason) == (True, None) and [p for p, _s in heard] == [result.path]


def test_after_without_a_file_plays_a_scratch_one_and_deletes_it(fake, heard):
    result = speak("Hello.", engine="fake", play="after", save=False)
    assert result.played and len(heard[0][1]) == 2400
    assert not os.path.exists(heard[0][0]) and os.path.dirname(heard[0][0]) == speak_dir()


@pytest.mark.parametrize("jit", [True, False])
def test_stream_plays_each_section_before_the_next_is_made(fake, headless_env, monkeypatch, jit):
    """On an engine without `supports_jit_streaming` too (grill PG25)."""
    with Host("fake") as host:
        monkeypatch.setattr(type(host.backend()), "capabilities", EngineCapabilities(supports_jit_streaming=jit))
        played = threading.Event()
        order = []

        def play(path, blocking):
            order.append("play")
            played.set()

        def say(text):
            order.append("speak")
            if len(fake.spoken) == 2:
                assert played.wait(10), "the second section started before the first was played"

        headless_env.play.side_effect = play
        fake.speak = say
        result = speak(LONG, host=host, play="stream")
    assert result.played and len(fake.spoken) > 3
    assert order[:2] == ["speak", "play"] or order[:3] == ["speak", "speak", "play"]
    assert order.count("play") == len(fake.spoken)
    assert len(sf.read(result.path)[0]) == round(0.1 * len(LONG.split()) * 24000)


def test_stream_plays_what_the_file_holds_in_order(fake, heard):
    result = speak(LONG, engine="fake", play="stream", pitch=3)
    joined = np.concatenate([samples for _path, samples in heard])
    assert np.allclose(joined, sf.read(result.path)[0], atol=1e-3)
    assert not any(os.path.exists(path) for path, _s in heard)
    assert not any(f.startswith(".play-") for f in os.listdir(speak_dir()))


def test_stream_without_processing_plays_the_segments_themselves(fake, heard):
    speak(LONG, engine="fake", play="stream", save=False)
    assert all(os.path.exists(path) for path, _s in heard)


def test_a_stream_that_fails_stops_playing(fake, headless_env, player):
    ended = threading.Event()
    headless_env.play.side_effect = lambda path, blocking: ended.wait(10)
    headless_env.stop.side_effect = ended.set

    def say(text):
        if len(fake.spoken) == 3:
            raise RuntimeError("no")

    fake.speak = say
    assert wrong(LONG, play="stream", player=player).code == "generation_failed"
    assert player.status()["playing"] is None and player.status()["queued"] == 0


@pytest.mark.parametrize("reason", ["disabled", "no_audio_device"])
@pytest.mark.parametrize("mode", ["after", "stream"])
def test_the_reason_nothing_played(fake, headless_env, monkeypatch, reason, mode):
    if reason == "disabled":
        monkeypatch.setenv(PLAYBACK_ENV, "off")
    headless_env.AVAILABLE = reason == "disabled"
    result = speak("Hello.", engine="fake", play=mode)
    assert (result.played, result.reason) == (False, reason) and os.path.isfile(result.path)
    assert headless_env.play.call_count == 0
    rendered = render_script(SCRIPT, engine="fake", play=mode)
    assert (rendered.played, rendered.reason) == (False, reason)


def test_interrupt_replaces_what_plays(fake, headless_env, player):
    ended = threading.Event()
    started = threading.Event()
    played = []

    def play(path, blocking):
        played.append(path)
        if len(played) == 1:
            started.set()
            ended.wait(10)

    headless_env.play.side_effect = play
    headless_env.stop.side_effect = ended.set
    player.enqueue("old")
    player.enqueue("older")
    assert started.wait(5)
    result = speak("New.", engine="fake", play="after", player=player, interrupt=True, wait_for_playback=True)
    assert played == ["old", result.path]


def test_a_cancel_while_playing_keeps_the_file(fake, headless_env):
    ended = threading.Event()
    headless_env.play.side_effect = lambda path, blocking: ended.wait(10)
    headless_env.stop.side_effect = ended.set
    result = speak("Hello.", engine="fake", play="after", cancel=lambda: headless_env.play.call_count > 0)
    assert (result.played, result.reason) == (False, "interrupted") and os.path.isfile(result.path)


def test_ctrl_c_while_a_phrase_plays_keeps_its_file(fake, monkeypatch, tmp_path):
    def stop(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(Player, "wait", stop)
    assert wrong(play="after").code == "interrupted"
    assert len(files_under(tmp_path / "audio_output")) == 1


# -- playback: render_script --------------------------------------------------------------------------


def test_after_plays_the_mixdown(fake, heard):
    result = render_script(SCRIPT, engine="fake", play="after")
    assert result.played and [p for p, _s in heard] == [result.mixdown_path]


def test_stream_plays_clips_in_order_with_their_gaps(fake, heard, monkeypatch):
    gaps = []
    real = render_module.Playback.add

    def add(self, path, *, gap_s=0.0, delete=False):
        gaps.append(round(gap_s, 3))
        real(self, path, delete=delete)

    monkeypatch.setattr(render_module.Playback, "add", add)
    result = render_script("One two.\n\n[pause:1.5]\n\nThree.\n\n[Sam]: Four five.", engine="fake", cast={"Sam": "v1"},
                           play="stream")
    assert gaps == [0.0, 1.5, 0.0] and [len(s) for _p, s in heard] == [4800, 2400, 4800]
    assert result.played and os.path.isfile(result.mixdown_path) and result.duration_s == pytest.approx(2.0)


def test_stream_generates_one_clip_at_a_time(fake, headless_env):
    played = threading.Event()
    headless_env.play.side_effect = lambda path, blocking: played.set()

    def say(text):
        if len(fake.spoken) == 2:
            assert played.wait(10), "the second clip started before the first was played"

    fake.speak = say
    assert render_script(SCRIPT, engine="fake", play="stream").played
    assert headless_env.play.call_count == 3


def test_stream_plays_a_clip_with_fx_as_it_sounds(fake, heard, tmp_path):
    folder = tmp_path / "presets" / "fx"
    folder.mkdir(parents=True)
    (folder / "Radio.json").write_text(json.dumps({"highpass_enabled": True, "highpass_freq": 900}))
    result = render_script("[Sam]: One two.", engine="fake", cast={"Sam": {"voice": "v1", "fx_preset": "Radio"}},
                           play="stream")
    assert np.allclose(heard[0][1], sf.read(result.mixdown_path)[0], atol=1e-3)
    assert not os.path.exists(heard[0][0])


def test_stream_of_an_update_plays_the_kept_clips_too(fake, heard):
    first = render_script(SCRIPT, engine="fake", cast={"Sam": "v1", "Ann": "v1"})
    result = render_script(SCRIPT.replace("Who", "And who"), engine="fake", update=first.project_path, play="stream")
    assert result.regenerated == 1 and [len(s) // 2400 for _p, s in heard] == [3, 4, 5]


def test_a_failed_stream_render_stops_playing_and_leaves_nothing(fake, headless_env, player, tmp_path):
    ended = threading.Event()
    headless_env.play.side_effect = lambda path, blocking: ended.wait(10)
    headless_env.stop.side_effect = ended.set

    def say(text):
        if len(fake.spoken) == 3:
            raise RuntimeError("no")

    fake.speak = say
    assert failing(play="stream", player=player).code == "generation_failed"
    assert player.status()["playing"] is None and not (tmp_path / "audio_output").exists()
    assert not project_io.is_locked(working_dirs()[0])


def test_ctrl_c_while_playing_keeps_the_files(fake, headless_env, monkeypatch, tmp_path):
    def stop(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(Player, "wait", stop)
    error = failing(play="after")
    assert error.code == "interrupted" and len(files_under(tmp_path / "audio_output")) == 2
    assert not project_io.is_locked(working_dirs()[0])


def test_bad_play_mode(fake):
    assert failing(play="later").code == "bad_arguments"
