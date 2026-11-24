"""headless/host.py: engines without Qt, error mapping, the download gate and
close(). Nothing here may reach the network: `network` records every attempt
and fails it, and a test that doesn't ask for `expect_network` then fails."""
import asyncio
import contextlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import requests

from kokoro_gui.engine.runner import EngineRunner, ModelBase
from kokoro_gui.engines import registry
from kokoro_gui.engines.base import BackendHooksMixin, EngineCapabilities, Synthesis, common_fields
from kokoro_gui.headless import host as host_module
from kokoro_gui.headless.errors import HeadlessError
from kokoro_gui.headless.host import Host, download_gate, downloads_allowed

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHA = "0123456789abcdef0123456789abcdef01234567"


class NetworkTouched(AssertionError):
    pass


class Attempts(list):
    expected = False


@pytest.fixture(autouse=True)
def network(monkeypatch):
    """Loopback passes: asyncio's self-pipe uses it on Windows."""
    attempts = Attempts()
    real_connect = socket.socket.connect

    def touched(what):
        attempts.append(what)
        raise NetworkTouched(what)

    def connect(self, address):
        if isinstance(address, tuple) and address and address[0] in ("127.0.0.1", "::1", "localhost"):
            return real_connect(self, address)
        return touched(f"connect {address}")

    monkeypatch.setattr(socket.socket, "connect", connect, raising=False)
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, *a, **k: touched(f"getaddrinfo {host}"))
    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send",
                        lambda self, request, *a, **k: touched(f"send {request.url}"))
    yield attempts
    assert attempts == [] or attempts.expected


@pytest.fixture
def expect_network(network):
    network.expected = True
    return network


@pytest.fixture
def state():
    """Everything the gate may change, as a comparable value."""
    from huggingface_hub import constants
    from huggingface_hub.utils import _http
    from transformers.utils import hub

    import spacy.cli

    return lambda: (dict(os.environ), constants.HF_HUB_OFFLINE, hub._is_offline_mode, hub.hf_hub_download,
                    _http._GLOBAL_BACKEND_FACTORY, spacy.cli.download)


@pytest.fixture(autouse=True)
def clean(monkeypatch, state):
    """No test starts with the gate's variables set or leaves one changed."""
    from huggingface_hub import constants
    from transformers.utils import hub

    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", host_module.ALLOW_DOWNLOAD_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(hub, "_is_offline_mode", False)
    yield
    assert host_module._GATE._holders == 0 and host_module._GATE._undo == []


@pytest.fixture
def hub_cache(monkeypatch, tmp_path):
    """An empty Hugging Face cache, and a function that puts files in it."""
    from huggingface_hub import constants
    from transformers.utils import hub

    root = tmp_path / "hf"
    root.mkdir()
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(root))
    monkeypatch.setattr(hub, "TRANSFORMERS_CACHE", str(root))

    def put(repo, files):
        base = root / ("models--" + repo.replace("/", "--"))
        (base / "refs").mkdir(parents=True, exist_ok=True)
        (base / "refs" / "main").write_text(SHA)
        for name, content in files.items():
            path = base / "snapshots" / SHA / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)

    return put


class FakeModel(ModelBase):
    engine_id = "fake"
    display_name = "Fake"

    def __init__(self, load=None, speak=None):
        self._load, self._speak = load, speak
        self.loads = 0

    def load(self, lang_code, device):
        self.loads += 1
        return (self._load() or True) if self._load else True

    def synthesize(self, text, voice, speed, lang_code, params):
        if self._speak:
            self._speak(text, voice, params)
        return Synthesis(np.full(800, 0.1, dtype=np.float32), [])

    def engine_version(self):
        return "1"


@pytest.fixture
def fake(isolated_dirs):
    """Registers an engine "fake" whose load and synthesize run given functions."""
    def register(load=None, speak=None, build=None):
        class Adapter(BackendHooksMixin):
            id = "fake"
            display_name = "Fake"
            capabilities = EngineCapabilities()

            def __init__(self):
                if build is not None:
                    build()
                self.model = FakeModel(load, speak)
                self._engine = EngineRunner(self.model)

            @property
            def engine(self):
                return self._engine

            @classmethod
            def get_config_schema(cls):
                return common_fields(caching_default=False)

            def cancel(self):
                self._engine.cancel()

        registry.register_engine("fake", Adapter, display_name="Fake")

    yield register
    registry.unregister_engine("fake")


@pytest.fixture
def make_host(clean):
    hosts = []

    def make(*args, **kwargs):
        hosts.append(Host(*args, **kwargs))
        return hosts[-1]

    yield make
    for host in hosts:
        host.close()


@pytest.fixture
def chunk(make_config, tmp_path):
    def make(index=0, text="Hello there.", **overrides):
        return (index, text, make_config(**{"voice": "v", "out_dir": str(tmp_path / f"c{index}"), **overrides}))

    return make


@pytest.fixture
def batch(chunk):
    return lambda *texts: [(f"c{i}", text, chunk(i, text)[2]) for i, text in enumerate(texts)]


def hub_file(filename="weights.pth", repo="acme/model"):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo, filename=filename)


def load_error(host, engine_id="fake", **kwargs) -> HeadlessError:
    with pytest.raises(HeadlessError) as caught:
        host.load(engine_id, **kwargs)
    return caught.value


def generate_error(host, engine_id, chunk) -> HeadlessError:
    with pytest.raises(HeadlessError) as caught:
        host.generate_clip(engine_id, chunk)
    return caught.value


def cause_names(error) -> list:
    return [type(e).__name__ for e in host_module._chain(error.__cause__)]


# -- loading -------------------------------------------------------------------

def test_dummy_loads_and_reports_status(make_host):
    seen = []
    host = make_host(on_status=lambda *a: seen.append(a))
    backend = host.load("dummy")
    assert backend is host.engines.resident["dummy"]
    assert backend.is_ready()
    assert seen == [("dummy", "Dummy ready (a).", False)]
    assert host.load("dummy") is backend
    assert len(seen) == 1


def test_the_default_engine_is_the_hosts(make_host):
    host = make_host("dummy")
    assert host.load().id == "dummy"
    assert host.engines.primary_engine_id == registry.DEFAULT_ENGINE_ID


def test_a_call_can_take_the_status(make_host):
    seen, own = [], []
    host = make_host(on_status=lambda *a: seen.append(a))
    host.load("dummy", on_status=lambda *a: own.append(a))
    assert seen == [] and len(own) == 1


def test_a_hook_that_raises_does_not_fail_the_load(make_host):
    def broken(*_args):
        raise ValueError("hook")

    assert make_host(on_status=broken).load("dummy").is_ready()


def test_another_language_loads_again(fake, make_host):
    fake()
    host = make_host()
    model = host.load("fake", lang_code="a").model
    host.load("fake", lang_code="a")
    assert model.loads == 1
    host.load("fake", lang_code="b")
    assert model.loads == 2


def test_loads_of_one_engine_run_one_at_a_time(fake, make_host):
    def load():
        time.sleep(0.1)

    fake(load=load)
    host = make_host()
    threads = [threading.Thread(target=host.load, args=("fake",)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)
    assert host.engines.resident["fake"].model.loads == 1


def test_generate_clip(make_host, chunk):
    host = make_host()
    host.load("dummy")
    results = host.generate_clip("dummy", chunk(voice="dummy"))
    assert results and all(os.path.isfile(r["path"]) for r in results)


# -- error mapping ---------------------------------------------------------------

def test_an_unknown_engine(make_host):
    error = load_error(make_host(), "no-such-engine")
    assert (error.code, error.exit_code) == ("engine_not_installed", 3)
    assert error.valid == registry.list_engines() and "dummy" in error.valid


def test_an_uninstalled_engine(make_host):
    registry.mark_unavailable("gone", "Gone Engine", "ImportError: no module named gone")
    try:
        error = load_error(make_host(), "gone")
    finally:
        registry.unregister_engine("gone")
    assert error.code == "engine_not_installed"
    assert "Gone Engine" in error.message and "no module named gone" in error.message
    assert "gone" not in error.valid


def test_an_engine_whose_constructor_fails(fake, make_host):
    boom = OSError("disk")

    def build():
        raise boom

    fake(build=build)
    error = load_error(make_host())
    assert (error.code, error.__cause__) == ("generation_failed", boom)


def test_a_failing_load_keeps_the_exception(fake, make_host):
    boom = RuntimeError("CUDA requested but not available")

    def load():
        raise boom

    fake(load=load)
    error = load_error(make_host())
    assert (error.code, error.exit_code) == ("generation_failed", 7)
    assert error.__cause__ is boom
    assert error.message == "Fake failed to load: CUDA requested but not available"


def test_a_load_can_be_retried_after_a_failure(fake, make_host):
    failing = [True]

    def load():
        if failing[0]:
            raise RuntimeError("first")

    fake(load=load)
    host = make_host()
    load_error(host)
    failing[0] = False
    assert host.load("fake").is_ready()


def test_a_system_exit_does_not_kill_the_worker(fake, make_host):
    """spaCy's downloader calls sys.exit."""
    fake(load=lambda: sys.exit(1))
    host = make_host()
    error = load_error(host)
    assert error.code == "generation_failed" and "SystemExit" in cause_names(error)
    assert host.engines.resident["fake"].engine.worker.is_alive()


def test_a_system_exit_in_a_batch_does_not_kill_the_worker(fake, make_host, batch):
    fake(speak=lambda *_a: sys.exit(1))
    host = make_host()
    with pytest.raises(HeadlessError) as caught:
        host.generate_clips("fake", batch("One."))
    assert caught.value.code == "generation_failed"
    assert host.engines.resident["fake"].engine.worker.is_alive()


@pytest.mark.parametrize("wrap", [False, True])
def test_uncached_weights(fake, make_host, hub_cache, wrap):
    def load():
        if not wrap:
            return hub_file()
        try:
            return hub_file()
        except Exception as e:
            raise RuntimeError(f"Failed to load: {e}") from e

    fake(load=load)
    error = load_error(make_host())
    assert (error.code, error.exit_code) == ("model_not_downloaded", 5)
    assert "weights.pth" in error.message and host_module.ALLOW_DOWNLOAD_ENV in error.message
    assert "LocalEntryNotFoundError" in cause_names(error)


def test_a_swallowed_optional_miss_does_not_hide_another_failure(fake, make_host, hub_cache):
    from transformers.utils.hub import cached_file

    def load():
        assert cached_file("acme/model", "model.safetensors", _raise_exceptions_for_missing_entries=False) is None
        raise RuntimeError("CUDA out of memory")

    hub_cache("acme/model", {"config.json": "{}"})
    fake(load=load)
    assert load_error(make_host()).code == "generation_failed"


def test_transformers_with_an_uncached_config(fake, make_host, hub_cache):
    from transformers import AutoConfig

    fake(load=lambda: AutoConfig.from_pretrained("acme/model"))
    error = load_error(make_host())
    assert error.code == "model_not_downloaded"
    assert isinstance(error.__cause__, OSError)


def test_transformers_with_a_cached_config_and_no_weights(fake, make_host, hub_cache):
    """transformers 4.57 gated ends in `AttributeError: 'NoneType' object has
    no attribute 'endswith'`, with no trace of the missing file."""
    from transformers import AutoModel

    hub_cache("acme/model", {"config.json": json.dumps({
        "model_type": "bert", "hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 1,
        "intermediate_size": 8, "vocab_size": 8})})
    fake(load=lambda: AutoModel.from_pretrained("acme/model"))
    error = load_error(make_host())
    assert error.code == "model_not_downloaded" and "model.safetensors" in error.message
    assert cause_names(error) == ["AttributeError"]


def test_an_empty_snapshot_download(fake, make_host, hub_cache):
    """What LuxTTS calls."""
    from huggingface_hub import snapshot_download

    fake(load=lambda: snapshot_download("acme/model"))
    assert load_error(make_host()).code == "model_not_downloaded"


def test_luxtts_through_its_own_adapter(make_host, hub_cache, monkeypatch):
    from huggingface_hub import snapshot_download

    from kokoro_gui.engines import luxtts

    monkeypatch.setattr(luxtts, "_load_lux", lambda device: snapshot_download(luxtts.MODEL_ID))
    assert load_error(make_host(), "luxtts").code == "model_not_downloaded"


def test_a_half_downloaded_snapshot(fake, make_host, hub_cache):
    from huggingface_hub import snapshot_download

    hub_cache("acme/model", {"config.json": "{}"})

    def load():
        with open(os.path.join(snapshot_download("acme/model"), "tokens.txt")) as f:
            return f.read()

    fake(load=load)
    assert load_error(make_host()).code == "model_not_downloaded"


def test_a_missing_file_outside_the_cache_is_not_a_download(fake, make_host, hub_cache, tmp_path):
    fake(load=lambda: open(tmp_path / "nothing.txt"))
    assert load_error(make_host()).code == "generation_failed"


def test_spacys_downloader_is_refused(fake, make_host, monkeypatch):
    import spacy.cli

    def download(*_args, **_kwargs):
        raise NetworkTouched("pip")

    monkeypatch.setattr(spacy.cli, "download", download)
    fake(load=lambda: spacy.cli.download("en_core_web_sm"))
    host = make_host()
    error = load_error(host)
    assert error.code == "model_not_downloaded" and "en_core_web_sm" in error.message
    assert isinstance(load_error(host, allow_download=True).__cause__, NetworkTouched)
    assert spacy.cli.download is download


# -- the gate --------------------------------------------------------------------

@pytest.mark.parametrize("value, allowed", [(None, False), ("", False), ("0", False), ("true", False), ("1", True)])
def test_the_variable(monkeypatch, value, allowed):
    if value is not None:
        monkeypatch.setenv(host_module.ALLOW_DOWNLOAD_ENV, value)
    assert downloads_allowed() is allowed
    assert downloads_allowed(True) is True
    assert downloads_allowed(False) is False


def test_the_hubs_variable_alone_does_nothing_once_it_is_imported(monkeypatch, expect_network, hub_cache):
    """Why the gate does more than set it."""
    from huggingface_hub.utils import _http

    _http.reset_sessions()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(NetworkTouched):
        hub_file()
    assert expect_network


def test_gated_with_the_hub_already_imported(hub_cache, state):
    from huggingface_hub.errors import LocalEntryNotFoundError

    before = state()
    with download_gate(False):
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        with pytest.raises(LocalEntryNotFoundError):
            hub_file()
    assert state() == before


GATED_FIRST = """
import os, socket, sys, tempfile
os.environ["HF_HUB_CACHE"] = tempfile.mkdtemp()
def touched(*a, **k):
    raise SystemExit("network touched")
socket.getaddrinfo = touched
from kokoro_gui.headless.host import download_gate
assert "huggingface_hub" not in sys.modules and "transformers" not in sys.modules
with download_gate(False):
    from transformers.utils import hub
    from huggingface_hub import constants, hf_hub_download, get_session
    assert hub.is_offline_mode() and constants.HF_HUB_OFFLINE
    try:
        hf_hub_download("acme/model", "weights.pth")
    except FileNotFoundError:
        pass
    else:
        raise SystemExit("not refused")
assert not hub.is_offline_mode() and not constants.HF_HUB_OFFLINE
assert "HF_HUB_OFFLINE" not in os.environ
assert all(type(a).__name__ == "UniqueRequestIdAdapter" for a in get_session().adapters.values())
print("ok")
"""


def test_gated_before_the_hub_and_transformers_are_imported():
    """A fresh interpreter. Nothing of the gate stays in either library."""
    result = subprocess.run([sys.executable, "-c", GATED_FIRST], cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split()[-1] == "ok"


def test_allowed_after_gated_and_gated_after_allowed(fake, make_host, expect_network, hub_cache):
    """The hub keeps one session per thread, so a kept one would carry the
    gate, or its absence, into the next load. These run on one pool thread."""
    thread_ids = set()

    def load():
        thread_ids.add(threading.get_ident())
        hub_file()

    fake(load=load)
    host = make_host()
    assert load_error(host).code == "model_not_downloaded"
    assert expect_network == []
    error = load_error(host, allow_download=True)
    assert len(expect_network) == 1 and isinstance(error.__cause__, NetworkTouched)
    assert load_error(host).code == "model_not_downloaded"
    assert len(expect_network) == 1
    assert len(thread_ids) == 1


def test_the_hosts_setting_the_calls_and_the_variable(fake, make_host, monkeypatch, expect_network, hub_cache):
    fake(load=hub_file)
    load_error(make_host(allow_download=True))
    assert len(expect_network) == 1
    monkeypatch.setenv(host_module.ALLOW_DOWNLOAD_ENV, "1")
    load_error(make_host())
    assert len(expect_network) == 2
    load_error(make_host(allow_download=False))
    load_error(make_host(), allow_download=False)
    assert len(expect_network) == 2


def test_a_cached_file_loads_while_gated(fake, make_host, hub_cache):
    hub_cache("acme/model", {"weights.pth": "weights"})
    fake(load=hub_file)
    assert make_host().load("fake").engine.pipeline.endswith("weights.pth")


def test_a_cached_tokenizer_loads_while_gated(fake, make_host, hub_cache):
    """transformers asks the hub about a model with over 100,000 tokens, as
    Audio8's has, unless its own flag is set."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import AutoTokenizer

    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, **{f"t{i}": i for i in range(1, 100_002)}}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    hub_cache("acme/model", {
        "config.json": json.dumps({"model_type": "bert"}),
        "tokenizer.json": tokenizer.to_str(),
        "tokenizer_config.json": json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"}),
    })
    loaded = []
    fake(load=lambda: loaded.append(AutoTokenizer.from_pretrained("acme/model")))
    make_host().load("fake")
    assert loaded[0].vocab_size == 100_002


def test_the_users_own_offline_setting_is_kept(monkeypatch):
    from huggingface_hub import constants

    monkeypatch.setenv("HF_HUB_OFFLINE", "yes")
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    for allow in (False, True, False):
        with download_gate(allow):
            assert os.environ["HF_HUB_OFFLINE"] == ("yes" if allow else "1")
            assert constants.HF_HUB_OFFLINE is True
        assert os.environ["HF_HUB_OFFLINE"] == "yes" and constants.HF_HUB_OFFLINE is True


def test_a_custom_http_backend_is_gated_and_put_back(hub_cache):
    """The hub's own offline flag doesn't apply to one."""
    from huggingface_hub import configure_http_backend, get_session
    from huggingface_hub.errors import LocalEntryNotFoundError
    from huggingface_hub.utils import _http

    original, own = _http._GLOBAL_BACKEND_FACTORY, requests.Session()
    configure_http_backend(lambda: own)
    try:
        with download_gate(False):
            with pytest.raises(LocalEntryNotFoundError):
                hub_file()
        assert get_session() is own
    finally:
        configure_http_backend(original)


def test_the_gate_nests_and_refuses_the_other_kind():
    with download_gate(False):
        with download_gate(False):
            pass
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        with pytest.raises(RuntimeError):
            with download_gate(True):
                pass
    assert "HF_HUB_OFFLINE" not in os.environ


def test_an_allowed_job_waits_for_a_gated_one(fake, make_host):
    started, go = threading.Event(), threading.Event()

    def load():
        started.set()
        assert go.wait(20)

    fake(load=load)
    host = make_host()
    first = threading.Thread(target=host.load, args=("fake",))
    first.start()
    assert started.wait(10)
    second = threading.Thread(target=host.load, args=("dummy",), kwargs={"allow_download": True})
    second.start()
    time.sleep(0.2)
    assert second.is_alive() and os.environ["HF_HUB_OFFLINE"] == "1"
    go.set()
    first.join(20)
    second.join(20)
    assert not second.is_alive() and "HF_HUB_OFFLINE" not in os.environ


def test_waiting_for_the_gate_can_be_cancelled(make_host):
    host = make_host()
    ready, go = threading.Event(), threading.Event()

    def hold():
        with download_gate(False):
            ready.set()
            go.wait(20)

    holder = threading.Thread(target=hold)
    holder.start()
    assert ready.wait(10)
    try:
        assert load_error(host, "dummy", allow_download=True, cancel=lambda: True).code == "interrupted"
    finally:
        go.set()
        holder.join(20)
    assert host.load("dummy", allow_download=True).is_ready()


def test_a_job_started_on_the_adapter_itself_is_gated(fake, make_host, hub_cache, chunk):
    fake(speak=lambda *_a: hub_file("voices/af_bella.pt"))
    future = make_host().backend("fake").generate_clip(chunk())
    assert type(future.exception(20)).__name__ == "LocalEntryNotFoundError"


# -- Kokoro and Audio8 -------------------------------------------------------------

def test_kokoro_with_an_empty_cache(make_host, hub_cache, monkeypatch):
    import kokoro_engine

    monkeypatch.setattr(kokoro_engine, "PIPELINE_DEVICE", None)  # the load sets it
    error = load_error(make_host(), "kokoro", device="cpu")
    assert error.code == "model_not_downloaded" and "config.json" in error.message


class VoiceOnlyPipeline:
    """A loaded Kokoro whose voices come from the real `KPipeline` code."""
    lang_code = "a"
    repo_id = "hexgrad/Kokoro-82M"

    def __init__(self, *_args, **_kwargs):
        self.voices = {}

    def __call__(self, text, voice=None, speed=1.0, split_pattern=None):
        from kokoro import KPipeline

        KPipeline.load_single_voice(self, voice)
        yield from ()


@pytest.fixture
def kokoro_pipeline(monkeypatch):
    """Replaces `KPipeline` in `kokoro_engine`, with no pipeline kept per thread."""
    import kokoro_engine

    def use(pipeline):
        monkeypatch.setattr(kokoro_engine, "KPipeline", pipeline)
        monkeypatch.setattr(kokoro_engine, "thread_local", threading.local())

    return use


def test_kokoro_with_an_uncached_voice(make_host, hub_cache, kokoro_pipeline, chunk, batch):
    kokoro_pipeline(VoiceOnlyPipeline)
    host = make_host()
    host.load("kokoro")
    error = generate_error(host, "kokoro", chunk(voice="af_bella"))
    assert error.code == "model_not_downloaded" and "voices/af_bella.pt" in error.message
    (clip_id, text, config), = batch("One.")
    with pytest.raises(HeadlessError) as caught:
        host.generate_clips("kokoro", [(clip_id, text, {**config, "voice": "af_bella"})])
    assert caught.value.code == "model_not_downloaded" and "voices/af_bella.pt" in caught.value.message
    assert "LocalEntryNotFoundError" in cause_names(caught.value)


def test_kokoros_per_thread_pipeline_swallows_the_miss(make_host, hub_cache, kokoro_pipeline, chunk):
    """`get_thread_pipeline` prints the error and returns None."""
    kokoro_pipeline(lambda **_kwargs: hub_file("kokoro-v1_0.pth", "hexgrad/Kokoro-82M"))
    error = generate_error(make_host(), "kokoro", chunk(voice="af_bella"))
    assert error.code == "model_not_downloaded" and "kokoro-v1_0.pth" in error.message


def test_kokoro_failing_for_another_reason(make_host, hub_cache, kokoro_pipeline, chunk):
    def pipeline(**_kwargs):
        raise RuntimeError("espeak is broken")

    kokoro_pipeline(pipeline)
    assert generate_error(make_host(), "kokoro", chunk(voice="af_bella")).code == "generation_failed"


def test_audio8_with_an_empty_cache(make_host, hub_cache, monkeypatch, tmp_path):
    from kokoro_gui.engines import audio8_tts

    monkeypatch.setattr(audio8_tts, "AUDIO8_REFS_DIR", str(tmp_path / "refs"))
    monkeypatch.setattr(audio8_tts, "_model", None)
    monkeypatch.setattr(audio8_tts, "_processor", None)
    error = load_error(make_host(), "audio8")
    assert error.code == "model_not_downloaded" and isinstance(error.__cause__, RuntimeError)


# -- batches ----------------------------------------------------------------------

def test_a_batch(make_host, batch):
    seen = []
    outcomes = make_host().generate_clips("dummy", batch("One.", "Two."), progress=lambda *a: seen.append(a))
    assert [o["success"] for o in outcomes] == [True, True]
    assert seen == [("c0", True), ("c1", True)]
    assert all("failure" not in o for o in outcomes)


def test_a_batchs_failures_are_mapped_per_clip(fake, make_host, hub_cache, batch):
    boom = ValueError("bad text")

    def speak(text, *_args):
        if "bad" in text:
            raise boom
        if "voice" in text:
            hub_file("voices/af_bella.pt")

    fake(speak=speak)
    host = make_host()
    outcomes = host.generate_clips("fake", batch("good", "bad", "voice"), partial=True)
    assert [o["success"] for o in outcomes] == [True, False, False]
    assert [o["failure"].code for o in outcomes[1:]] == ["generation_failed", "model_not_downloaded"]
    assert outcomes[1]["failure"].__cause__ is boom
    with pytest.raises(HeadlessError) as caught:
        host.generate_clips("fake", batch("good", "bad", "voice"))
    assert caught.value.code == "model_not_downloaded"


def test_one_clips_failure_keeps_its_exception(fake, make_host, chunk):
    boom = ValueError("bad text")

    def speak(*_args):
        raise boom

    fake(speak=speak)
    error = generate_error(make_host(), "fake", chunk())
    assert error.code == "generation_failed" and error.__cause__ is boom


def test_a_progress_hook_that_raises_does_not_fail_the_clip(make_host, batch):
    def broken(*_args):
        raise ValueError("hook")

    assert make_host().generate_clips("dummy", batch("One."), progress=broken)[0]["success"]


# -- cancelling --------------------------------------------------------------------

@pytest.fixture
def blocking(fake):
    """An engine that blocks in synthesize until it is cancelled. `started` is
    set once it does, and clearing `on` lets later clips through."""
    blocking = SimpleNamespace(started=threading.Event(), on=True)

    def speak(text, voice, params):
        if blocking.on:
            blocking.started.set()
            assert params["cancel_event"].wait(20)

    fake(speak=speak)
    return blocking


def test_cancelling_a_batch_then_running_another(blocking, make_host, batch):
    host = make_host()
    with pytest.raises(HeadlessError) as caught:
        host.generate_clips("fake", batch("One.", "Two."), cancel=blocking.started.is_set)
    assert (caught.value.code, caught.value.exit_code) == ("interrupted", 130)
    blocking.on = False
    assert host.generate_clips("fake", batch("Three."))[0]["success"]


def test_cancelling_from_another_thread(blocking, make_host, chunk):
    host = make_host()
    codes = []
    thread = threading.Thread(target=lambda: codes.append(generate_error(host, "fake", chunk()).code))
    thread.start()
    assert blocking.started.wait(10)
    host.cancel("dummy")
    assert thread.is_alive()
    host.cancel("fake")
    thread.join(20)
    assert codes == ["interrupted"]


def test_ctrl_c_cancels_and_propagates(blocking, make_host, chunk):
    def cancel():
        if blocking.started.is_set():
            raise KeyboardInterrupt
        return False

    host = make_host()
    with pytest.raises(KeyboardInterrupt):
        host.generate_clip("fake", chunk(), cancel=cancel)


def test_a_cancelled_load_keeps_the_gate_until_it_ends(fake, make_host, hub_cache):
    """The caller is gone, but the thread still must not download."""
    started, go = threading.Event(), threading.Event()
    errors = []

    def load():
        started.set()
        assert go.wait(20)
        try:
            hub_file()
        except Exception as e:
            errors.append(type(e).__name__)
            raise

    fake(load=load)
    host = make_host()
    assert load_error(host, cancel=started.is_set).code == "interrupted"
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert load_error(host, cancel=lambda: True).code == "interrupted"
    go.set()
    host.close()
    assert errors == ["LocalEntryNotFoundError"]
    assert "HF_HUB_OFFLINE" not in os.environ


# -- close() -----------------------------------------------------------------------

def threads_of(host):
    """Every thread a host's engines own: each worker and its loop's pool."""
    threads = []
    for backend in host.engines.resident.values():
        threads.append(backend.engine.worker)
        executor = getattr(backend.engine.worker.loop, "_default_executor", None)
        threads += list(executor._threads) if executor else []
    return threads


def alive(threads):
    return [t for t in threads if t.is_alive()]


def test_close_joins_every_thread(make_host, chunk):
    host = make_host()
    host.load("dummy")
    host.generate_clip("dummy", chunk(voice="dummy"))
    threads = threads_of(host)
    assert len(threads) >= 2 and alive(threads) == threads
    host.close()
    assert alive(threads) == []
    assert threads[0].loop.is_closed() and host.engines.resident == {}


def test_close_after_a_failed_load(fake, make_host):
    def load():
        raise RuntimeError("no")

    fake(load=load)
    host = make_host()
    load_error(host)
    threads = threads_of(host)
    assert len(threads) >= 2
    host.close()
    assert alive(threads) == []


def test_close_twice_and_without_engines(make_host):
    host = make_host()
    host.close()
    host.close()
    host = make_host()
    host.load("dummy")
    host.close()
    host.close()


def test_close_from_several_threads(make_host):
    host = make_host()
    host.load("dummy")
    threads = [threading.Thread(target=host.close) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)
    assert alive(threads) == [] and host.engines.resident == {}


def test_close_stops_a_running_generate(blocking, make_host, batch):
    host = make_host()
    codes = []

    def work():
        with pytest.raises(HeadlessError) as caught:
            host.generate_clips("fake", batch("One.", "Two."))
        codes.append(caught.value.code)

    thread = threading.Thread(target=work)
    thread.start()
    assert blocking.started.wait(10)
    threads = threads_of(host)
    host.close()
    thread.join(20)
    assert codes == ["interrupted"] and alive(threads) == []


def test_a_closed_host_refuses(make_host):
    host = make_host()
    backend = host.load("dummy")
    host.close()
    assert load_error(host, "dummy").code == "interrupted"
    with pytest.raises(HeadlessError):
        backend.run(asyncio.sleep(0))


def test_close_goes_on_when_an_engine_fails_to_stop(fake, make_host):
    fake()
    host = make_host()
    broken = host.load("fake")
    host.load("dummy")
    threads = threads_of(host)
    real_stop = broken.stop

    def stop():
        real_stop()
        raise OSError("stop")

    broken.stop = stop
    host.close()
    assert alive(threads) == []


def test_a_dead_worker_does_not_hang_the_caller(fake, make_host, chunk):
    fake(speak=lambda *_a: time.sleep(0.2))
    host = make_host()
    worker = host.load("fake").engine.worker
    ended = []

    def work():
        with contextlib.suppress(HeadlessError):
            host.generate_clip("fake", chunk())
        ended.append(1)

    thread = threading.Thread(target=work)
    thread.start()
    worker.loop.call_soon_threadsafe(worker.loop.stop)
    thread.join(20)
    assert ended == [1]
    host.close()


def test_the_context_manager_closes_and_leaves_no_change(state, fake, hub_cache):
    before = state()
    fake(load=hub_file)
    with Host() as host:
        load_error(host)
        host.load("dummy")
    assert host.engines.resident == {}
    assert state() == before
