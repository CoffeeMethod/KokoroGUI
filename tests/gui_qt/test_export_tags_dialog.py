"""The Export dialog's Tags tab and what a tagged export writes (plan 17):
kokoro_gui/qt/docks/export_dialog.py with kokoro_gui/daw/tagging.py."""
import os

import pytest

mutagen = pytest.importorskip("mutagen")  # a pinned requirement; CI installs it

from mutagen.flac import FLAC  # noqa: E402
from mutagen.id3 import ID3  # noqa: E402
from PySide6.QtGui import QColor, QImage  # noqa: E402

import kokoro_gui.qt.app  # noqa: F401,E402 - app.py must load before any docks module (circular import)
from kokoro_gui.daw import markers, tagging  # noqa: E402
from kokoro_gui.qt.docks.export_dialog import ExportDialog, export_defaults, run_export, tag_options  # noqa: E402
from tests.gui_qt.test_export_and_characters import _generated_clip, _type  # noqa: E402
from tests.gui_qt.test_export_presets_dialog import _scheduled_result, _two_subprojects  # noqa: E402


def _png(path, size=200):
    image = QImage(size, size // 2, QImage.Format.Format_RGB32)
    image.fill(QColor("red"))
    assert image.save(str(path))
    return str(path)


def _tone_project(qt_app, tmp_path):
    qt_app.document.settings["gap_s"] = 0.0
    _type(qt_app.editor, "hello world")
    return _generated_clip(qt_app, tmp_path, 0, 5, seconds=2.0, name="tone")


def _tab_of(dialog, widget):
    page = widget
    while page is not None and dialog.tabs.indexOf(page) < 0:
        page = page.parentWidget()
    return dialog.tabs.tabText(dialog.tabs.indexOf(page))


def test_the_tags_tab_sits_between_extras_and_the_project_file(qt_app):
    dialog = ExportDialog(qt_app)

    assert [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())] == ["Audio", "Extras", "Tags", "Project file"]
    for name in ("tags_check", "tag_title_edit", "tag_artist_edit", "tag_album_edit", "tag_track_edit",
                 "tag_year_edit", "tag_description_edit", "cover_edit", "cover_preview", "tag_chapters_check"):
        assert _tab_of(dialog, getattr(dialog, name)) == "Tags", name


def test_the_defaults_write_tags_with_the_project_title_and_chapters(qt_app):
    qt_app.project_settings["title"] = "My Podcast"
    dialog = ExportDialog(qt_app)

    assert dialog.tags_check.isChecked() and dialog.tag_chapters_check.isChecked()
    assert dialog.tag_title_edit.text() == "" and dialog.tag_title_edit.placeholderText() == "My Podcast"
    assert dialog.values()["tags"] == dict(tagging.DEFAULT_SETTINGS)


def test_the_tag_fields_persist_per_project(qt_app, tmp_path):
    cover = _png(tmp_path / "cover.png")
    dialog = ExportDialog(qt_app)
    dialog.tag_title_edit.setText("Episode One")
    dialog.tag_artist_edit.setText("The Cast")
    dialog.tag_album_edit.setText("The Show")
    dialog.tag_track_edit.setText("7")
    dialog.tag_year_edit.setText("2026")
    dialog.tag_description_edit.setPlainText("Line one.\nLine two.")
    dialog.cover_edit.setText(cover)
    dialog.tag_chapters_check.setChecked(False)
    values = dialog.values()
    qt_app.project_settings["export"] = values

    assert values["tags"] == {"enabled": True, "title": "Episode One", "artist": "The Cast", "album": "The Show",
                              "track": "7", "year": "2026", "description": "Line one.\nLine two.",
                              "cover": os.path.abspath(cover), "chapters": False}
    again = ExportDialog(qt_app)
    assert again.tag_title_edit.text() == "Episode One" and again.tag_track_edit.text() == "7"
    assert again.tag_description_edit.toPlainText() == "Line one.\nLine two."
    assert again.cover_edit.text() == os.path.abspath(cover) and not again.tag_chapters_check.isChecked()
    assert export_defaults(qt_app)["tags"] == values["tags"]


def test_the_number_and_year_fields_take_digits_only(qt_app):
    dialog = ExportDialog(qt_app)

    for edit, typed in ((dialog.tag_track_edit, "3a/b"), (dialog.tag_year_edit, "20x26")):
        edit.clear()
        for char in typed:
            if edit.validator().validate(edit.text() + char, 0)[0] != edit.validator().State.Invalid:
                edit.setText(edit.text() + char)

    assert dialog.tag_track_edit.text() == "3" and dialog.tag_year_edit.text() == "2026"


def test_export_defaults_ignore_bad_stored_tags(qt_app):
    qt_app.project_settings["export"] = {"tags": {"enabled": "yes", "title": 5, "artist": ["x"], "chapters": None,
                                                  "bogus": 1}}

    tags = export_defaults(qt_app)["tags"]

    assert tags["enabled"] is True and tags["chapters"] is True and tags["artist"] == "" and tags["title"] == "5"
    assert set(tags) == set(tagging.DEFAULT_SETTINGS)
    qt_app.project_settings["export"] = {"tags": "nonsense"}
    assert export_defaults(qt_app)["tags"] == dict(tagging.DEFAULT_SETTINGS)


def test_the_cover_gets_a_preview_and_a_bad_one_gets_a_reason(qt_app, tmp_path):
    dialog = ExportDialog(qt_app)
    assert dialog.cover_preview.pixmap().isNull() and dialog.cover_note_label.text() == ""

    dialog.cover_edit.setText(_png(tmp_path / "cover.png"))
    pixmap = dialog.cover_preview.pixmap()
    assert not pixmap.isNull() and max(pixmap.width(), pixmap.height()) == 64
    assert dialog.cover_note_label.text() == ""

    (tmp_path / "cover.gif").write_bytes(b"GIF89a....")
    dialog.cover_edit.setText(str(tmp_path / "cover.gif"))
    assert dialog.cover_preview.pixmap().isNull()
    assert dialog.cover_note_label.text() == "The cover image must be a JPEG or PNG; it will be left out."

    dialog.cover_edit.setText(str(tmp_path / "missing.png"))
    assert "doesn't exist" in dialog.cover_note_label.text()


def test_the_tab_explains_what_wav_split_and_mp3_do(qt_app):
    dialog = ExportDialog(qt_app)
    assert dialog.format_combo.currentText() == "wav"
    assert "WAV files can't carry tags" in dialog.tags_note_label.text()
    assert not dialog.tag_chapters_check.isEnabled()

    dialog.format_combo.setCurrentText("mp3")
    assert dialog.tags_note_label.text() == "" and dialog.tag_chapters_check.isEnabled()

    dialog.format_combo.setCurrentText("flac")
    assert not dialog.tag_chapters_check.isEnabled() and dialog.tag_chapters_check.isChecked()

    dialog.tags_check.setChecked(False)
    assert not dialog.tag_artist_edit.isEnabled() and not dialog.cover_edit.isEnabled()


def test_a_split_export_greys_out_the_title_and_number(qt_app):
    _two_subprojects(qt_app)
    dialog = ExportDialog(qt_app)
    assert dialog.tag_title_edit.isEnabled() and dialog.tag_track_edit.isEnabled()

    dialog.format_combo.setCurrentText("mp3")
    dialog.split_combo.setCurrentIndex(dialog.split_combo.findData("subprojects"))

    assert not dialog.tag_title_edit.isEnabled() and not dialog.tag_track_edit.isEnabled()
    assert dialog.tag_artist_edit.isEnabled()
    assert "titled with its chapter" in dialog.tags_note_label.text()


def test_without_mutagen_the_tab_is_disabled_and_nothing_is_written(qt_app, monkeypatch):
    qt_app.project_settings["export"] = {"tags": {"enabled": True, "title": "Kept"}}
    monkeypatch.setattr(tagging, "available", lambda: False)

    dialog = ExportDialog(qt_app)

    index = dialog.tabs.indexOf(dialog.tags_form.parentWidget())
    assert not dialog.tabs.isTabEnabled(index) and "mutagen" in dialog.tabs.tabToolTip(index)
    assert not dialog.tags_check.isEnabled()
    assert dialog.values()["tags"]["enabled"] is True and dialog.values()["tags"]["title"] == "Kept"  # not lost
    assert tag_options(qt_app, dialog.values()) is None


def test_tag_options_fall_back_to_the_project_title_and_take_the_genre_from_the_preset(qt_app):
    qt_app.project_settings["title"] = "My Podcast"
    values = export_defaults(qt_app)

    options = tag_options(qt_app, values)
    assert options["title"] == "My Podcast" and options["genre"] == "" and options["chapters"] is True

    assert tag_options(qt_app, {**values, "preset": "acx"})["genre"] == "Audiobook"
    assert tag_options(qt_app, {**values, "preset": "spotify"})["genre"] == "Podcast"
    assert tag_options(qt_app, {**values, "preset": "youtube"})["genre"] == ""

    named = {**values, "tags": {**values["tags"], "title": "Chosen", "track": "4"}}
    assert (tag_options(qt_app, named)["title"], tag_options(qt_app, named)["track"]) == ("Chosen", "4")
    assert tag_options(qt_app, {**values, "tags": {**values["tags"], "enabled": False}}) is None


def test_a_tagged_export_carries_the_title_and_the_cover(qt_app, tmp_path):
    _tone_project(qt_app, tmp_path)
    cover = _png(tmp_path / "cover.png")
    dialog = ExportDialog(qt_app)
    dialog.out_dir_edit.setText(str(tmp_path / "out"))
    dialog.filename_edit.setText("ep1")
    dialog.format_combo.setCurrentText("flac")
    dialog.tag_title_edit.setText("Episode One")
    dialog.tag_artist_edit.setText("The Cast")
    dialog.cover_edit.setText(cover)

    assert run_export(qt_app, dialog.values()) is True
    result = _scheduled_result(qt_app)

    audio = FLAC(result.audio_path)
    assert audio["TITLE"] == ["Episode One"] and audio["ARTIST"] == ["The Cast"]
    assert len(audio.pictures) == 1 and audio.pictures[0].mime == "image/png"
    assert qt_app.project_settings["export"]["tags"]["cover"] == os.path.abspath(cover)


def test_an_untitled_tagged_export_uses_the_project_title(qt_app, tmp_path):
    qt_app.project_settings["title"] = "My Podcast"
    _tone_project(qt_app, tmp_path)
    dialog = ExportDialog(qt_app)
    dialog.out_dir_edit.setText(str(tmp_path / "out"))
    dialog.format_combo.setCurrentText("flac")

    assert run_export(qt_app, dialog.values()) is True
    result = _scheduled_result(qt_app)

    assert FLAC(result.audio_path)["TITLE"] == ["My Podcast"]


def test_tags_off_writes_a_plain_file(qt_app, tmp_path):
    _tone_project(qt_app, tmp_path)
    dialog = ExportDialog(qt_app)
    dialog.out_dir_edit.setText(str(tmp_path / "out"))
    dialog.format_combo.setCurrentText("flac")
    dialog.tags_check.setChecked(False)

    assert run_export(qt_app, dialog.values()) is True
    result = _scheduled_result(qt_app)

    assert not FLAC(result.audio_path).tags and result.files[0].tagged is False


def test_a_tagged_mp3_export_carries_the_markers_as_chapters(qt_app, tmp_path):
    _tone_project(qt_app, tmp_path)
    for seconds, name in ((0.0, "Opening"), (1.0, "Second half")):
        qt_app.document.settings["markers"], _m = markers.add_marker(qt_app.document.settings, seconds, name)
    dialog = ExportDialog(qt_app)
    dialog.out_dir_edit.setText(str(tmp_path / "out"))
    dialog.format_combo.setCurrentText("mp3")

    assert run_export(qt_app, dialog.values()) is True
    result = _scheduled_result(qt_app)

    chapters = ID3(result.audio_path).getall("CHAP")
    assert [(c.start_time, str(c.sub_frames["TIT2"])) for c in chapters] == [(0, "Opening"), (1000, "Second half")]


def test_a_split_export_tags_each_file_with_its_chapter(qt_app, tmp_path):
    _two_subprojects(qt_app)
    dialog = ExportDialog(qt_app)
    dialog.out_dir_edit.setText(str(tmp_path / "out"))
    dialog.format_combo.setCurrentText("flac")
    dialog.split_combo.setCurrentIndex(dialog.split_combo.findData("subprojects"))
    dialog.tag_album_edit.setText("The Book")

    assert run_export(qt_app, dialog.values()) is True
    result = _scheduled_result(qt_app)

    titles = [(FLAC(f.path)["TITLE"], FLAC(f.path)["TRACKNUMBER"], FLAC(f.path)["ALBUM"]) for f in result.files]
    assert titles == [(["Alpha"], ["1"], ["The Book"]), (["Beta"], ["2"], ["The Book"])]
