"""Tests for kokoro_gui/daw/share.py: the `.kgshare` format (plan 40). Export,
the validation of an untrusted file, and the import with its name clashes. No
test touches a real store: each builds its own library, FX folder and voice
stores under tmp_path, and the zips are made in memory."""
import io
import json
import os
import wave
import zipfile

import pytest

from kokoro_gui.daw import share
from kokoro_gui.daw.library import CharacterLibrary
from kokoro_gui.daw.models import Character
from kokoro_gui.engines.voice_store import EmbeddingStore, ReferenceStore


def wav_bytes(frames: int = 800, rate: int = 8000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * frames)
    return buf.getvalue()


class Machine:
    """One machine's stores under `root`."""

    def __init__(self, root):
        self.root = root
        self.fx_dir = str(root / "presets" / "fx")
        self.library = CharacterLibrary(str(root / "characters"))
        self.mixes = EmbeddingStore("kokoro", ".pt", global_dir=lambda: str(root / "custom_voices"))
        self.refs = ReferenceStore("fakeref", global_dir=lambda: str(root / "refs"))
        self.stores = {"kokoro": self.mixes, "fakeref": self.refs}

    def store_for(self, engine_id):
        return self.stores.get(engine_id)

    def add_fx(self, name, values):
        os.makedirs(self.fx_dir, exist_ok=True)
        with open(os.path.join(self.fx_dir, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(values, f)

    def add_ir(self, name, data=None):
        os.makedirs(os.path.join(self.fx_dir, "ir"), exist_ok=True)
        with open(os.path.join(self.fx_dir, "ir", f"{name}.wav"), "wb") as f:
            f.write(data or wav_bytes())

    def add_mix(self, name, data=b"pt-bytes"):
        os.makedirs(self.mixes.global_dir, exist_ok=True)
        with open(os.path.join(self.mixes.global_dir, f"{name}.pt"), "wb") as f:
            f.write(data)

    def add_ref(self, name, transcript="hello there", data=None):
        os.makedirs(self.refs.global_dir, exist_ok=True)
        with open(os.path.join(self.refs.global_dir, f"{name}.wav"), "wb") as f:
            f.write(data or wav_bytes())
        with open(os.path.join(self.refs.global_dir, f"{name}.txt"), "w", encoding="utf-8") as f:
            f.write(transcript)

    def export(self, path, characters=(), **kwargs):
        return share.export_share(str(path), characters, fx_dir=self.fx_dir, store_for=self.store_for, **kwargs)

    def plan(self, info):
        return share.plan_import(info, library=self.library, fx_dir=self.fx_dir, store_for=self.store_for)

    def import_(self, info, choices):
        return share.import_share(info, choices, library=self.library, fx_dir=self.fx_dir, store_for=self.store_for)


@pytest.fixture
def source(tmp_path):
    m = Machine(tmp_path / "a")
    m.add_fx("Radio", {"reverb_enabled": True, "convolution_ir": "Hall", "convolution_mix": 0.4, "gain_db": 3})
    m.add_ir("Hall")
    m.add_mix("mymix")
    return m


@pytest.fixture
def target(tmp_path):
    return Machine(tmp_path / "b")


def alice(**preset):
    data = {"voice": "mymix", "speed": 1.2, "fx_preset": "Radio"}
    data.update(preset)
    return Character(name="Alice", preset_data=data, highlight_color="#112233")


def make_zip(path, entries: dict, manifest="default"):
    """Writes a zip at `path` from `{name: bytes | ZipInfo-writing callable}`."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        if manifest == "default":
            manifest = {"format": share.FORMAT, "version": share.VERSION, "characters": []}
        if manifest is not None:
            zf.writestr("share.json", json.dumps(manifest))
        for name, data in entries.items():
            zf.writestr(name, data)
    return str(path)


def manifest_with(*characters):
    return {"format": share.FORMAT, "version": share.VERSION, "characters": list(characters)}


# -- round trip --------------------------------------------------------------------


def test_a_character_with_its_fx_ir_and_mix_round_trips(tmp_path, source, target):
    path = tmp_path / "alice.kgshare"
    report = source.export(path, [alice()])
    assert report.fx == ["Radio"] and report.irs == ["Hall"] and report.voices == ["kokoro/mymix"]
    assert report.warnings == []

    info = share.inspect_share(str(path))
    assert [c.name for c in info.characters] == ["Alice"]
    assert set(info.fx) == {"Radio"} and info.irs == ["Hall"]
    assert [(v.engine_id, v.name) for v in info.voices] == [("kokoro", "mymix")]

    choices = {item.key: share.KEEP for item in target.plan(info)}
    result = target.import_(info, choices)
    assert sorted(result.imported) == ["Character Alice", "FX Radio", "Impulse response Hall", "Voice mymix (kokoro)"]

    (entry,) = target.library.list()
    assert entry.name == "Alice" and entry.highlight_color == "#112233"
    assert entry.preset_data == {"voice": "mymix", "speed": 1.2, "fx_preset": "Radio"}
    assert entry.library_id and entry.library_id != "Alice"
    with open(os.path.join(target.fx_dir, "Radio.json"), encoding="utf-8") as f:
        assert json.load(f)["convolution_ir"] == "Hall"
    with open(os.path.join(target.fx_dir, "ir", "Hall.wav"), "rb") as f:
        assert f.read() == wav_bytes()
    with open(os.path.join(target.mixes.global_dir, "mymix.pt"), "rb") as f:
        assert f.read() == b"pt-bytes"


def test_a_real_torch_file_round_trips(tmp_path, source, target):
    torch = pytest.importorskip("torch")
    pt = os.path.join(source.mixes.global_dir, "mymix.pt")
    torch.save({"w": torch.zeros(3)}, pt)
    path = tmp_path / "t.kgshare"
    source.export(path, [alice()])
    info = share.inspect_share(str(path))
    target.import_(info, {item.key: share.KEEP for item in target.plan(info)})
    loaded = torch.load(os.path.join(target.mixes.global_dir, "mymix.pt"), weights_only=True)
    assert loaded["w"].tolist() == [0.0, 0.0, 0.0]


def test_a_reference_voice_takes_its_wav_and_transcript(tmp_path, source, target):
    source.add_ref("narrator", "once upon a time")
    cloner = Character(name="Narrator", backend_id="fakeref", preset_data={"voice": "narrator"},
                       variants={"angry": "narrator"})
    path = tmp_path / "n.kgshare"
    assert source.export(path, [cloner]).voices == ["fakeref/narrator"]
    info = share.inspect_share(str(path))
    target.import_(info, {item.key: share.KEEP for item in target.plan(info)})
    assert target.refs.get_transcript("narrator") == "once upon a time"
    assert target.refs.find_wav("narrator")


def test_export_can_leave_fx_and_voices_out(tmp_path, source):
    path = tmp_path / "plain.kgshare"
    report = source.export(path, [alice()], include_fx=False, include_voices=False)
    assert report.fx == [] and report.irs == [] and report.voices == []
    with zipfile.ZipFile(path) as zf:
        assert zf.namelist() == ["share.json"]


def test_fx_only_export_carries_the_fx_and_its_ir(tmp_path, source):
    path = tmp_path / "fx.kgshare"
    report = source.export(path, fx_names=["Radio"])
    assert report.characters == 0 and report.fx == ["Radio"] and report.irs == ["Hall"]
    info = share.inspect_share(str(path))
    assert info.characters == [] and set(info.fx) == {"Radio"}


def test_the_project_copy_of_an_fx_wins(tmp_path, source):
    project = tmp_path / "proj"
    (project / "fx").mkdir(parents=True)
    (project / "fx" / "Radio.json").write_text(json.dumps({"gain_db": -6}), encoding="utf-8")
    path = tmp_path / "p.kgshare"
    source.export(path, [alice()], project_dir=str(project))
    assert share.inspect_share(str(path)).fx["Radio"] == {"gain_db": -6}


def test_something_missing_is_a_warning_not_a_failure(tmp_path, source):
    path = tmp_path / "w.kgshare"
    report = source.export(path, [alice(fx_preset="Gone", voice="nope")])
    assert any("Gone" in w for w in report.warnings)
    assert report.fx == [] and report.voices == []


def test_a_failed_export_leaves_the_old_file_alone(tmp_path, source, monkeypatch):
    path = tmp_path / "keep.kgshare"
    path.write_bytes(b"old")
    monkeypatch.setattr(share, "MAX_TOTAL_BYTES", 10)
    with pytest.raises(share.ShareError):
        source.export(path, [alice()])
    assert path.read_bytes() == b"old"
    assert not os.path.exists(str(path) + ".tmp")


def test_export_carries_no_id_and_no_unknown_keys():
    c = Character(name="Bob", preset_data={"voice": "x", "mystery": 1}, library_id="lib1", extra={"k": 1})
    data = share.character_to_share(c)
    assert set(data) == {"name", "backend_id", "highlight_color", "preset_data", "variants"}
    assert data["preset_data"] == {"voice": "x"}


# -- an untrusted file ---------------------------------------------------------------


@pytest.mark.parametrize("name", ["../evil.json", "/abs/evil.json", "C:evil.json", "fx/../../evil.json",
                                  "..\\evil.json", "C:\\evil.json"])
def test_unsafe_entry_names_are_refused(tmp_path, name):
    path = make_zip(tmp_path / "bad.kgshare", {name: b"{}"})
    with pytest.raises(share.ShareError, match="unsafe entry name"):
        share.inspect_share(path)


def test_a_symlink_entry_is_refused(tmp_path):
    path = tmp_path / "link.kgshare"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("share.json", json.dumps(manifest_with()))
        link = zipfile.ZipInfo("fx/Radio.json")
        link.external_attr = (0o120777 << 16)
        zf.writestr(link, "/etc/passwd")
    with pytest.raises(share.ShareError, match="symlink"):
        share.inspect_share(str(path))


def test_a_repeated_entry_is_refused(tmp_path):
    path = tmp_path / "dup.kgshare"
    with pytest.warns(UserWarning):
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("share.json", json.dumps(manifest_with()))
            zf.writestr("share.json", json.dumps(manifest_with()))
    with pytest.raises(share.ShareError, match="twice"):
        share.inspect_share(str(path))


def test_too_many_entries_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(share, "MAX_ENTRIES", 5)
    path = make_zip(tmp_path / "many.kgshare", {f"fx/f{i}.json": b"{}" for i in range(6)})
    with pytest.raises(share.ShareError, match="entries"):
        share.inspect_share(path)


def test_a_total_over_the_limit_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(share, "MAX_TOTAL_BYTES", 1000)
    path = make_zip(tmp_path / "big.kgshare", {"fx/a.json": b" " * 600, "fx/b.json": b" " * 600})
    with pytest.raises(share.ShareError, match="expands to"):
        share.inspect_share(path)


def test_an_entry_over_its_own_limit_is_refused(tmp_path):
    # 60 MB of zeros compresses to a few dozen KB; the 50 MB impulse-response limit still applies.
    path = make_zip(tmp_path / "ir.kgshare", {"fx/ir/Hall.wav": b"RIFF\0\0\0\0WAVE" + b"\0" * (60 * 1024 * 1024)})
    with pytest.raises(share.ShareError, match="limit"):
        share.inspect_share(path)


def test_a_pt_needs_torch_2_6(tmp_path, monkeypatch):
    path = make_zip(tmp_path / "pt.kgshare", {"voices/kokoro/mix.pt": b"x"})
    monkeypatch.setattr(share, "_torch_weights_only_available", lambda: False)
    with pytest.raises(share.ShareError, match="torch"):
        share.inspect_share(path)
    monkeypatch.setattr(share, "_torch_weights_only_available", lambda: True)
    assert [v.name for v in share.inspect_share(path).voices] == ["mix"]


def test_a_file_without_a_manifest_or_with_the_wrong_one_is_refused(tmp_path):
    with pytest.raises(share.ShareError, match="share.json"):
        share.inspect_share(make_zip(tmp_path / "a.kgshare", {}, manifest=None))
    with pytest.raises(share.ShareError, match="isn't a KokoroGUI share file"):
        share.inspect_share(make_zip(tmp_path / "b.kgshare", {}, manifest={"format": "other", "version": 1}))
    with pytest.raises(share.ShareError, match="newer"):
        share.inspect_share(make_zip(tmp_path / "c.kgshare", {}, manifest={"format": share.FORMAT, "version": 9}))
    (tmp_path / "d.kgshare").write_bytes(b"not a zip")
    with pytest.raises(share.ShareError, match="readable"):
        share.inspect_share(str(tmp_path / "d.kgshare"))
    with pytest.raises(share.ShareError, match="doesn't exist"):
        share.inspect_share(str(tmp_path / "missing.kgshare"))


def test_a_manifest_that_is_not_json_is_refused(tmp_path):
    path = tmp_path / "junk.kgshare"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("share.json", "{not json")
    with pytest.raises(share.ShareError, match="isn't a KokoroGUI share file"):
        share.inspect_share(str(path))


def test_a_character_dict_loses_unknown_keys_and_bad_values(tmp_path):
    bad = {"name": "Eve", "highlight_color": "red", "backend_id": "../x",
           "preset_data": {"voice": "ok", "speed": "fast", "volume": 1e9, "pitch": float("nan"), "trim": "yes",
                           "format": "wav", "out_dir": "/etc", "fx_preset": "../../evil", "normalize": True,
                           "apply_fx": False},
           "variants": {"a": "b", "c": 5, "": "x"}, "library_id": "steal", "id": "steal", "extra": {"x": 1}}
    path = make_zip(tmp_path / "c.kgshare", {}, manifest=manifest_with(bad, "nope", {"name": ""}, {"name": 3}))
    info = share.inspect_share(path)
    (eve,) = info.characters
    assert eve.preset_data == {"voice": "ok", "volume": 4.0, "format": "wav", "normalize": True, "apply_fx": False}
    assert eve.backend_id == "kokoro" and eve.highlight_color != "red"
    assert eve.variants == {"a": "b"}
    assert eve.library_id is None and eve.extra == {}
    assert any("out_dir" in w for w in info.warnings)


def test_fx_values_go_through_the_preset_coercion(tmp_path):
    fx = {"gain_db": 1e9, "comp_ratio": "loud", "reverb_enabled": "yes", "convolution_ir": "Hall", "evil": 1}
    info = share.inspect_share(make_zip(tmp_path / "f.kgshare", {"fx/Odd.json": json.dumps(fx)}))
    assert info.fx["Odd"] == {"gain_db": 20, "convolution_ir": "Hall"}


def test_an_fx_that_is_not_an_object_is_ignored(tmp_path):
    info = share.inspect_share(make_zip(tmp_path / "f.kgshare", {"fx/List.json": "[1]", "fx/Junk.json": "zzz"}))
    assert info.fx == {} and len(info.warnings) == 2


def test_entries_this_version_has_no_use_for_are_ignored(tmp_path):
    path = make_zip(tmp_path / "x.kgshare", {
        "fx/.hidden.json": b"{}", "voices/kokoro/.pt": b"x", "voices/b ad/x.pt": b"x",
        "readme.txt": b"hi", "fx/deep/er.json": b"{}", "voices/kokoro/x.exe": b"x", "fx/CON.json": b"{}"})
    info = share.inspect_share(path)
    assert info.is_empty() and info.ignored == 7


def test_a_wav_entry_that_is_not_a_wav_is_dropped(tmp_path):
    path = make_zip(tmp_path / "w.kgshare", {"fx/ir/Fake.wav": b"MZ not a wav at all!",
                                             "voices/fakeref/v.wav": b"nope", "fx/ir/Real.wav": wav_bytes()})
    info = share.inspect_share(path)
    assert info.irs == ["Real"] and info.voices == []
    assert len(info.warnings) == 2


def test_classify_entry():
    assert share.classify_entry("share.json") == ("manifest",)
    assert share.classify_entry("fx/Radio.json") == ("fx", "Radio")
    assert share.classify_entry("fx/ir/Hall.wav") == ("ir", "Hall")
    assert share.classify_entry("voices/kokoro/mix.pt") == ("voice", "kokoro", "mix", ".pt")
    assert share.classify_entry("fx/a:b.json") is None
    assert share.classify_entry("voices/kokoro/") is None


def test_clean_stem():
    assert share.clean_stem("Hall (2)") == "Hall (2)"
    for bad in ("", " x", "x ", ".", "..", ".x", "x.", "a/b", "a\\b", "a:b", "nul", "COM1.txt", "a\x00b", "x" * 101, 5):
        assert share.clean_stem(bad) is None, bad


# -- import --------------------------------------------------------------------------


def exported(tmp_path, source, characters=None):
    path = tmp_path / "s.kgshare"
    source.export(path, characters if characters is not None else [alice()])
    return share.inspect_share(str(path))


def test_nothing_is_written_until_import(tmp_path, source, target):
    info = exported(tmp_path, source)
    assert target.plan(info)
    assert not os.path.exists(target.fx_dir) and target.library.list() == []


def test_the_plan_marks_clashes_and_identical_items(tmp_path, source, target):
    info = exported(tmp_path, source)
    target.add_fx("Radio", {"gain_db": -3})                 # differs
    target.add_ir("Hall")                                    # identical
    target.add_mix("mymix", b"other bytes")                  # differs
    target.library.save(Character(name="alice"))             # same name, any case
    plan = {item.key: item for item in target.plan(info)}
    assert plan["character:0"].clash
    assert plan["fx:Radio"].clash and not plan["fx:Radio"].same
    assert plan["ir:Hall"].same and not plan["ir:Hall"].clash
    assert plan["voice:kokoro:mymix"].clash


def test_keep_both_numbers_the_name_and_the_references_follow(tmp_path, source, target):
    info = exported(tmp_path, source)
    target.add_fx("Radio", {"gain_db": -3})
    target.add_ir("Hall", wav_bytes(frames=1600))
    target.add_mix("mymix", b"other bytes")
    target.library.save(Character(name="Alice"))
    result = target.import_(info, {item.key: share.KEEP for item in target.plan(info)})

    assert result.renamed == {"ir:Hall": "Hall (2)", "fx:Radio": "Radio (2)", "voice:kokoro:mymix": "mymix (2)",
                              "character:0": "Alice (2)"}
    # The old ones are untouched.
    with open(os.path.join(target.fx_dir, "Radio.json"), encoding="utf-8") as f:
        assert json.load(f) == {"gain_db": -3}
    with open(os.path.join(target.mixes.global_dir, "mymix.pt"), "rb") as f:
        assert f.read() == b"other bytes"
    # The new FX points at the new IR, the character at the new FX and Mix.
    with open(os.path.join(target.fx_dir, "Radio (2).json"), encoding="utf-8") as f:
        assert json.load(f)["convolution_ir"] == "Hall (2)"
    assert os.path.isfile(os.path.join(target.fx_dir, "ir", "Hall (2).wav"))
    by_name = {c.name: c for c in target.library.list()}
    assert by_name["Alice"].preset_data == {}
    assert by_name["Alice (2)"].preset_data["fx_preset"] == "Radio (2)"
    assert by_name["Alice (2)"].preset_data["voice"] == "mymix (2)"


def test_replace_overwrites_and_keeps_the_library_id(tmp_path, source, target):
    info = exported(tmp_path, source)
    target.add_fx("Radio", {"gain_db": -3})
    target.add_mix("mymix", b"other bytes")
    old_id = target.library.save(Character(name="alice", preset_data={"voice": "old"}))
    result = target.import_(info, {item.key: share.REPLACE for item in target.plan(info)})
    assert result.renamed == {}
    (entry,) = target.library.list()
    assert entry.library_id == old_id and entry.name == "Alice" and entry.preset_data["voice"] == "mymix"
    with open(os.path.join(target.fx_dir, "Radio.json"), encoding="utf-8") as f:
        assert json.load(f)["gain_db"] == 3
    with open(os.path.join(target.mixes.global_dir, "mymix.pt"), "rb") as f:
        assert f.read() == b"pt-bytes"


def test_skip_and_unlisted_items_write_nothing(tmp_path, source, target):
    info = exported(tmp_path, source)
    target.import_(info, {"character:0": share.SKIP})
    assert target.library.list() == [] and not os.path.exists(target.fx_dir)


def test_an_identical_item_is_not_written_again(tmp_path, source, target):
    info = exported(tmp_path, source)
    target.add_ir("Hall")
    result = target.import_(info, {item.key: share.KEEP for item in target.plan(info)})
    assert "Impulse response Hall" not in result.imported
    assert sorted(os.listdir(os.path.join(target.fx_dir, "ir"))) == ["Hall.wav"]


def test_replace_on_an_item_with_no_clash_just_adds_it(tmp_path, source, target):
    info = exported(tmp_path, source)
    result = target.import_(info, {item.key: share.REPLACE for item in target.plan(info)})
    assert len(result.imported) == 4 and result.renamed == {}


def test_a_voice_for_an_engine_not_installed_is_skipped(tmp_path, source, target):
    path = make_zip(tmp_path / "e.kgshare", {"voices/ghost/v.wav": wav_bytes(), "voices/ghost/v.txt": b"hi"})
    info = share.inspect_share(path)
    (item,) = target.plan(info)
    assert "isn't installed" in item.unavailable
    result = target.import_(info, {item.key: share.KEEP})
    assert result.imported == [] and len(result.skipped) == 1


def test_a_reference_without_its_transcript_is_not_imported(tmp_path, target):
    path = make_zip(tmp_path / "r.kgshare", {"voices/fakeref/v.wav": wav_bytes()})
    (item,) = target.plan(share.inspect_share(path))
    assert item.unavailable == "it has no transcript"


def test_a_pt_for_a_reference_engine_is_unavailable(tmp_path, target):
    path = make_zip(tmp_path / "r.kgshare", {"voices/fakeref/v.pt": b"x"})
    (item,) = target.plan(share.inspect_share(path))
    assert item.unavailable


def test_extraction_counts_bytes_and_stops_past_the_cap(tmp_path):
    path = make_zip(tmp_path / "i.kgshare", {"fx/ir/Hall.wav": wav_bytes(frames=4000)})
    out = tmp_path / "out.wav"
    with zipfile.ZipFile(path) as zf:
        with pytest.raises(share.ShareError, match="bigger than it says"):
            share._extract(zf, "fx/ir/Hall.wav", str(out), 100)
        assert not out.exists() and not os.path.exists(str(out) + ".tmp")
        share._extract(zf, "fx/ir/Hall.wav", str(out), 10_000)
    assert out.read_bytes() == wav_bytes(frames=4000)


def test_an_ir_that_is_not_audio_is_removed_after_the_write(tmp_path, target):
    path = make_zip(tmp_path / "i.kgshare", {"fx/ir/Bad.wav": b"RIFF\0\0\0\0WAVE not really a wav"})
    result = target.import_(share.inspect_share(path), {"ir:Bad": share.KEEP})
    assert result.imported == [] and any("Bad" in w for w in result.warnings)
    assert os.listdir(os.path.join(target.fx_dir, "ir")) == []


def test_import_rereads_the_file(tmp_path, source, target):
    info = exported(tmp_path, source)
    # The file is swapped for something unsafe between inspect and import.
    make_zip(info.path, {"../evil.json": b"{}"})
    with pytest.raises(share.ShareError):
        target.import_(info, {"character:0": share.KEEP})
    assert target.library.list() == []


def test_a_keep_name_stays_inside_the_stores(tmp_path, target):
    # Names are clean stems before they reach a path, so nothing here can leave the FX folder.
    path = make_zip(tmp_path / "n.kgshare", {"fx/ok.json": b'{"gain_db": 1}'})
    target.import_(share.inspect_share(path), {"fx:ok": share.KEEP})
    assert os.listdir(target.fx_dir) == ["ok.json"]
