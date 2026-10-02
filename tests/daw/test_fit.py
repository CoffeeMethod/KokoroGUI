"""kokoro_gui/daw/fit.py: the speaking rates learned from generated clips."""
import numpy as np
import pytest
import soundfile as sf

from kokoro_gui.daw import fit
from kokoro_gui.daw.models import Character, Clip, Document, Run, Segment
from kokoro_gui.daw.undo import ImportBedCommand


def _document():
    doc = Document()
    doc.characters.append(Character.from_preset_dict("Ann", {"voice": "af_heart"}))
    return doc


def _add(doc, text, seconds=None, **clip_kwargs):
    clip = Clip(**clip_kwargs)
    if seconds is not None:
        clip.segments = [Segment(text=text, audio_path="a.wav", duration=seconds)]
    doc.clips.append(clip)
    doc.runs.append(Run(text=text, clip_id=clip.id, kind=clip.run_kind))
    doc.runs.append(Run(text=" "))
    return clip


def test_a_music_bed_does_not_count_toward_the_speaking_rate(tmp_path):
    doc = _document()
    ann = doc.characters[0].id
    _add(doc, "x" * 40, 2.0, character_id=ann)
    wav = str(tmp_path / "theme.wav")
    sf.write(wav, np.zeros(8000 * 180, dtype=np.float32), 8000)
    doc.undo_stack.push(ImportBedCommand(wav, "a very long music bed file name", 0.0))
    assert any(c.is_bed for c in doc.clips)

    rates = fit.speaking_rates(doc)

    assert rates == {ann: pytest.approx(20.0), None: pytest.approx(20.0)}


def test_nested_and_imported_recording_clips_are_skipped():
    doc = _document()
    ann = doc.characters[0].id
    _add(doc, "x" * 40, 2.0, character_id=ann)
    _add(doc, "chapter one", 600.0, source="nested", child={"kind": "embedded", "id": "c1"})
    _add(doc, "y" * 90, 1.0, source="imported", character_id=ann)

    assert fit.speaking_rates(doc) == {ann: pytest.approx(20.0), None: pytest.approx(20.0)}


def test_a_clip_without_a_character_counts_once_toward_the_document_rate():
    doc = _document()
    _add(doc, "x" * 40, 2.0)

    assert fit.speaking_rates(doc) == {None: pytest.approx(20.0)}

    ann = doc.characters[0].id
    _add(doc, "y" * 10, 2.0, character_id=ann)
    rates = fit.speaking_rates(doc)
    assert rates[ann] == pytest.approx(5.0)
    assert rates[None] == pytest.approx(50 / 4.0)
