import json

from kokoro_gui.headless.results import RenderResult, SpeakResult


def test_speak_result_json_round_trips():
    r = SpeakResult(path="a.wav", duration_s=1.5, sample_rate=24000, engine="dummy", voice="v",
                    words=[{"word": "hi", "start": 0.0, "end": 0.2}], warnings=["w"])
    data = json.loads(json.dumps(r.to_json()))
    assert data["path"] == "a.wav"
    assert data["words"][0]["word"] == "hi"
    assert data["played"] is False and data["reason"] is None


def test_speak_result_save_false_has_no_path():
    assert SpeakResult().to_json()["path"] is None


def test_render_result_fields():
    r = RenderResult(project_path="p.tbaw", mixdown_path="m.wav", clips=3, regenerated=1, channels=1)
    data = json.loads(json.dumps(r.to_json()))
    assert {"project_path", "mixdown_path", "out_dir", "duration_s", "clips", "regenerated", "speakers",
            "words", "played", "reason", "warnings", "channels"} <= set(data)
    assert data["clips"] == 3 and data["regenerated"] == 1


def test_lists_are_not_shared_between_instances():
    a, b = SpeakResult(), SpeakResult()
    a.warnings.append("x")
    assert b.warnings == []


def test_no_audio_bytes_in_a_result():
    def has_bytes(value):
        if isinstance(value, (bytes, bytearray)):
            return True
        if isinstance(value, dict):
            return any(has_bytes(v) for v in value.values())
        if isinstance(value, list):
            return any(has_bytes(v) for v in value)
        return False

    assert not has_bytes(SpeakResult().to_json())
    assert not has_bytes(RenderResult().to_json())
