"""Tests for kokoro_gui/daw/library_copy.py: Save to library (plan 41). Each
test builds a project dir and a library under tmp_path; nothing touches a real
store."""
import json
import os

import pytest

from kokoro_gui.daw import library_copy as lc
from kokoro_gui.daw import share
from kokoro_gui.daw.models import Character
from kokoro_gui.engines.voice_store import EmbeddingStore, ReferenceStore


class Setup:
    def __init__(self, root):
        self.project = str(root / "project")
        self.fx_dir = str(root / "presets" / "fx")
        self.mixes = EmbeddingStore("kokoro", ".pt", global_dir=lambda: str(root / "custom_voices"))
        self.refs = ReferenceStore("fakeref", global_dir=lambda: str(root / "refs"))
        self.stores = {"kokoro": self.mixes, "fakeref": self.refs}

    def store_for(self, engine_id):
        return self.stores.get(engine_id)

    @staticmethod
    def put(path, data):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data if isinstance(data, bytes) else data.encode("utf-8"))

    def project_mix(self, name, data=b"mix"):
        self.put(os.path.join(self.project, "engines", "kokoro", "voices", f"{name}.pt"), data)

    def library_mix(self, name, data=b"mix"):
        self.put(os.path.join(self.mixes.global_dir, f"{name}.pt"), data)

    def project_ref(self, name, text="hello", wav=b"wav"):
        base = os.path.join(self.project, "engines", "fakeref", "refs", name)
        self.put(base + ".wav", wav)
        self.put(base + ".txt", text)

    def library_ref(self, name, text="hello", wav=b"wav"):
        base = os.path.join(self.refs.global_dir, name)
        self.put(base + ".wav", wav)
        self.put(base + ".txt", text)

    def project_fx(self, name, values, ir=None):
        self.put(os.path.join(self.project, "fx", f"{name}.json"), json.dumps(values))
        if ir:
            self.put(os.path.join(self.project, "fx", "ir", f"{ir}.wav"), b"ir-bytes")

    def library_fx(self, name, values):
        self.put(os.path.join(self.fx_dir, f"{name}.json"), json.dumps(values))

    def library_ir(self, name, data=b"ir-bytes"):
        self.put(os.path.join(self.fx_dir, "ir", f"{name}.wav"), data)

    def save(self, items, choices=None):
        return lc.save_to_library(items, choices, fx_dir=self.fx_dir)


@pytest.fixture
def s(tmp_path):
    return Setup(tmp_path)


def read(path):
    with open(path, "rb") as f:
        return f.read()


# -- voices -----------------------------------------------------------------------------


def test_a_voice_the_library_lacks_is_copied(s):
    s.project_mix("deep", b"tensor")
    item = lc.voice_item(s.mixes, "kokoro", "deep", s.project)
    assert item.status == lc.ABSENT and item.label == "Mix"

    result = s.save([item])

    assert result.saved == ["Mix deep (kokoro)"] and result.renamed == {}
    assert read(os.path.join(s.mixes.global_dir, "deep.pt")) == b"tensor"
    assert os.path.isfile(os.path.join(s.project, "engines", "kokoro", "voices", "deep.pt"))  # the project is only read


def test_an_identical_voice_is_a_no_op(s):
    s.project_mix("deep", b"same")
    s.library_mix("deep", b"same")
    item = lc.voice_item(s.mixes, "kokoro", "deep", s.project)
    assert item.status == lc.IDENTICAL and item.same
    result = s.save([item], {item.key: share.REPLACE})
    assert result.saved == [] and "already has it" in result.skipped[0]


def test_a_voice_that_differs_offers_three_choices(s):
    s.project_mix("deep", b"project")
    s.library_mix("deep", b"library")
    item = lc.voice_item(s.mixes, "kokoro", "deep", s.project)
    assert item.status == lc.DIFFERENT and item.clash

    assert s.save([item]).saved == []  # no choice is a skip
    assert read(os.path.join(s.mixes.global_dir, "deep.pt")) == b"library"

    kept = s.save([item], {item.key: share.KEEP})
    assert kept.saved == ["Mix deep (2) (kokoro)"] and kept.renamed == {item.key: "deep (2)"}
    assert read(os.path.join(s.mixes.global_dir, "deep (2).pt")) == b"project"
    assert read(os.path.join(s.mixes.global_dir, "deep.pt")) == b"library"

    replaced = s.save([item], {item.key: share.REPLACE})
    assert replaced.saved == ["Mix deep (kokoro)"]
    assert read(os.path.join(s.mixes.global_dir, "deep.pt")) == b"project"


def test_a_reference_copies_both_files(s):
    s.project_ref("ann", "the words", b"audio")
    item = lc.voice_item(s.refs, "fakeref", "ann", s.project)
    assert item.label == "Reference" and item.status == lc.ABSENT
    s.save([item])
    assert read(os.path.join(s.refs.global_dir, "ann.wav")) == b"audio"
    assert read(os.path.join(s.refs.global_dir, "ann.txt")) == b"the words"


def test_a_reference_with_a_different_transcript_differs(s):
    s.project_ref("ann", "new words")
    s.library_ref("ann", "old words")
    assert lc.voice_item(s.refs, "fakeref", "ann", s.project).status == lc.DIFFERENT


def test_a_reference_missing_its_transcript_in_the_library_is_copied(s):
    s.project_ref("ann")
    s.library_ref("ann")
    os.remove(os.path.join(s.refs.global_dir, "ann.txt"))
    item = lc.voice_item(s.refs, "fakeref", "ann", s.project)
    assert item.status == lc.ABSENT
    s.save([item])
    assert read(os.path.join(s.refs.global_dir, "ann.txt")) == b"hello"


def test_a_voice_the_project_does_not_carry_is_not_local(s):
    s.library_mix("deep")
    assert lc.voice_item(s.mixes, "kokoro", "deep", s.project).status == lc.NOT_LOCAL
    assert lc.voice_item(s.mixes, "kokoro", "af_heart", s.project).status == lc.NOT_LOCAL
    assert lc.voice_item(None, "dummy", "x", s.project).status == lc.NOT_LOCAL
    assert lc.voice_item(s.mixes, "kokoro", "deep", None).status == lc.NOT_LOCAL


@pytest.mark.parametrize("name", ["../evil", "a/b", "..", "con", "a\\b", ""])
def test_an_unsafe_voice_name_is_never_looked_up(s, tmp_path, name):
    s.put(str(tmp_path / "evil.pt"), b"x")
    s.project_mix("ok")
    item = lc.voice_item(s.mixes, "kokoro", name, s.project)
    assert item.status == lc.NOT_LOCAL and item.files == []
    assert s.save([item]).saved == []


def test_a_project_file_that_is_a_link_out_of_the_project_is_refused(s, tmp_path):
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"secret")
    folder = os.path.join(s.project, "engines", "kokoro", "voices")
    os.makedirs(folder)
    try:
        os.symlink(str(outside), os.path.join(folder, "sneaky.pt"))
    except (OSError, NotImplementedError):
        pytest.skip("no symlinks here")
    # `_inside` resolves links, and the result is outside the folder.
    assert lc.voice_item(s.mixes, "kokoro", "sneaky", s.project).status == lc.NOT_LOCAL


# -- FX ---------------------------------------------------------------------------------


def test_an_fx_the_library_lacks_is_copied_with_its_impulse_response(s):
    s.project_fx("Radio", {"gain_db": 3, "convolution_ir": "Hall"}, ir="Hall")
    items = lc.fx_items("Radio", s.project, s.fx_dir)
    assert [(i.kind, i.status) for i in items] == [("fx", lc.ABSENT), ("ir", lc.ABSENT)]
    result = s.save(items)
    assert sorted(result.saved) == ["FX Radio", "Impulse response Hall"]
    with open(os.path.join(s.fx_dir, "Radio.json"), encoding="utf-8") as f:
        assert json.load(f)["gain_db"] == 3
    assert read(os.path.join(s.fx_dir, "ir", "Hall.wav")) == b"ir-bytes"


def test_an_identical_fx_is_a_no_op_even_if_the_formatting_differs(s):
    s.project_fx("Radio", {"gain_db": 3})
    s.put(os.path.join(s.fx_dir, "Radio.json"), json.dumps({"gain_db": 3}, indent=8))
    (item,) = lc.fx_items("Radio", s.project, s.fx_dir)
    assert item.status == lc.IDENTICAL


def test_keeping_both_renames_the_fx_and_its_impulse_response_together(s):
    s.project_fx("Radio", {"gain_db": 3, "convolution_ir": "Hall"}, ir="Hall")
    s.library_fx("Radio", {"gain_db": -1})
    s.library_ir("Hall", b"another")
    items = lc.fx_items("Radio", s.project, s.fx_dir)
    assert [i.status for i in items] == [lc.DIFFERENT, lc.DIFFERENT]
    result = s.save(items, {i.key: share.KEEP for i in items})
    assert sorted(result.saved) == ["FX Radio (2)", "Impulse response Hall (2)"]
    with open(os.path.join(s.fx_dir, "Radio (2).json"), encoding="utf-8") as f:
        assert json.load(f)["convolution_ir"] == "Hall (2)"
    assert read(os.path.join(s.fx_dir, "ir", "Hall.wav")) == b"another"


def test_replace_overwrites_the_library_fx(s):
    s.project_fx("Radio", {"gain_db": 3})
    s.library_fx("Radio", {"gain_db": -1})
    (item,) = lc.fx_items("Radio", s.project, s.fx_dir)
    s.save([item], {item.key: share.REPLACE})
    with open(os.path.join(s.fx_dir, "Radio.json"), encoding="utf-8") as f:
        assert json.load(f)["gain_db"] == 3


def test_a_library_only_fx_has_nothing_to_save_but_its_project_ir_does(s):
    s.library_fx("Radio", {"gain_db": 3, "convolution_ir": "Hall"})
    s.put(os.path.join(s.project, "fx", "ir", "Hall.wav"), b"ir-bytes")
    items = lc.fx_items("Radio", s.project, s.fx_dir)
    assert [i.kind for i in items] == ["ir"]
    assert lc.fx_items("Nope", s.project, s.fx_dir) == []
    assert lc.fx_items("Select FX Preset...", s.project, s.fx_dir) == []


def test_an_fx_names_the_save_writes_are_clean_stems(s, tmp_path):
    s.project_fx("ok", {"gain_db": 1})
    assert lc.fx_items("../ok", s.project, s.fx_dir) == []
    assert not os.path.exists(os.path.join(os.path.dirname(s.fx_dir), "ok.json"))


def test_values_are_filtered_on_the_way_out(s):
    s.project_fx("Loud", {"gain_db": 999, "not_a_key": "x"})
    (item,) = lc.fx_items("Loud", s.project, s.fx_dir)
    s.save([item])
    with open(os.path.join(s.fx_dir, "Loud.json"), encoding="utf-8") as f:
        saved = json.load(f)
    assert "not_a_key" not in saved and saved == share.filter_fx_preset_values({"gain_db": 999})


# -- characters -------------------------------------------------------------------------


def test_a_character_gathers_its_voice_variants_fx_and_ir(s):
    s.project_ref("ann")
    s.project_ref("ann_whisper")
    s.library_ref("elsewhere")
    s.project_fx("Radio", {"convolution_ir": "Hall"}, ir="Hall")
    character = Character(name="Ann", backend_id="fakeref", preset_data={"voice": "ann", "fx_preset": "Radio"},
                          variants={"whisper": "ann_whisper", "far": "elsewhere"})
    items = lc.character_items(character, s.project, s.fx_dir, s.store_for)
    assert sorted((i.kind, i.name) for i in items) == [
        ("fx", "Radio"), ("ir", "Hall"), ("voice", "ann"), ("voice", "ann_whisper")]
    assert len({i.key for i in items}) == 4
    s.save(items)
    assert os.path.isfile(os.path.join(s.refs.global_dir, "ann_whisper.wav"))
    assert os.path.isfile(os.path.join(s.fx_dir, "ir", "Hall.wav"))


def test_a_character_with_nothing_local_gathers_nothing(s):
    s.library_mix("deep")
    character = Character(name="Bo", backend_id="kokoro", preset_data={"voice": "deep", "fx_preset": "Radio"})
    assert lc.character_items(character, s.project, s.fx_dir, s.store_for) == []
