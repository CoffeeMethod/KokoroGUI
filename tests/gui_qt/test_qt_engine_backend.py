"""Engine follows the character (grill V3): each character picks its engine,
every engine in use stays resident, and the Settings tab's schema form and
the Voices tab follow the active character (the selected clip's, else the
first). See app.py's `backend` property and `set_character_engine`."""
from kokoro_gui.daw.models import Character
from kokoro_gui.engines import registry as engine_registry


def _use_engine(qt_app, engine_id, character=None):
    character = character or qt_app.document.characters[0]
    assert qt_app.set_character_engine(character, engine_id)
    return character


def test_dummy_and_kokoro_both_registered():
    assert {"kokoro", "dummy"} <= set(engine_registry.list_engines())


def test_kokoro_backend_shows_mixing_dock(qt_app):
    assert qt_app.backend.id == "kokoro"
    assert qt_app.mixing_dock is not None


def test_a_dummy_character_hides_mixing_dock(qt_app):
    _use_engine(qt_app, "dummy")
    assert qt_app.backend.id == "dummy"
    assert qt_app.mixing_dock is None


def test_back_to_kokoro_shows_mixing_dock_again(qt_app):
    _use_engine(qt_app, "dummy")
    _use_engine(qt_app, "kokoro")
    assert qt_app.backend.id == "kokoro"
    assert qt_app.mixing_dock is not None


def test_engine_change_rebuilds_schema_form_for_the_characters_backend(qt_app):
    """The Settings tab's schema-driven fields reflect the active
    character's backend, not whatever the first backend built."""
    original_form = qt_app.settings_dock.schema_form
    _use_engine(qt_app, "dummy")
    assert qt_app.settings_dock.schema_form is not original_form

    dummy_schema_keys = {f.key for f in qt_app.backend.get_config_schema()}
    assert "lexicon" not in dummy_schema_keys  # dummy backend has no lexicon field
    rendered_keys = set(qt_app.settings_dock.schema_form.values().keys())
    # "pitch" is skip_keyed too - SettingsDock renders it via its own
    # hand-built pitch_spin (Audio Control), not the schema form, to avoid
    # two independent widgets fighting over the same override slot.
    assert rendered_keys == dummy_schema_keys - {"lexicon", "pitch"}


def test_engine_change_refused_while_job_running(qt_app):
    qt_app.transport_dock.set_busy(True)  # simulate a job in flight
    character = qt_app.document.characters[0]
    assert qt_app.set_character_engine(character, "dummy") is False
    assert (character.backend_id or "kokoro") == "kokoro"


def test_no_global_engine_menu(qt_app):
    titles = [a.text() for a in qt_app.options_menu.actions()]
    assert "Engine" not in titles
    assert not hasattr(qt_app, "switch_engine")


def test_two_characters_two_engines_stay_resident(qt_app):
    doc = qt_app.document
    narrator = doc.characters[0]
    robot = Character.from_preset_dict("Robot", {"voice": "dummy"}, backend_id="kokoro")
    doc.characters.append(robot)
    _use_engine(qt_app, "dummy", robot)

    assert {"kokoro", "dummy"} <= set(qt_app.backends)
    doc.text = "hello there"
    first = doc.assign_character_to_range(0, 5, narrator.id)
    second = doc.assign_character_to_range(6, 11, robot.id)
    assert qt_app.backend_for(first).id == "kokoro"
    assert qt_app.backend_for(second).id == "dummy"
    assert qt_app._assemble_generation_config(second)["engine_id"] == "dummy"

    # The Settings tab follows the selected clip's character.
    qt_app.selection.select_clip(second.id)
    assert qt_app.backend.id == "dummy"
    assert qt_app.mixing_dock is None
    qt_app.selection.select_clip(first.id)
    assert qt_app.backend.id == "kokoro"
    assert qt_app.mixing_dock is not None


def test_batch_generate_sends_each_clip_to_its_characters_engine(qt_app, monkeypatch):
    import concurrent.futures

    doc = qt_app.document
    narrator = doc.characters[0]
    robot = Character.from_preset_dict("Robot", {"voice": "dummy"})
    doc.characters.append(robot)
    _use_engine(qt_app, "dummy", robot)
    doc.text = "hello there"
    first = doc.assign_character_to_range(0, 5, narrator.id)
    second = doc.assign_character_to_range(6, 11, robot.id)

    dummy_engine = qt_app.backends["dummy"].engine
    sent = {}

    def fake_run(coro_owner):
        def _run(coro):
            coro.close()
            future = concurrent.futures.Future()
            sent.setdefault(coro_owner, []).append(future)
            return future
        return _run

    calls = {}
    monkeypatch.setattr(dummy_engine.worker, "run_coro", fake_run("dummy"))
    monkeypatch.setattr(dummy_engine, "generate_dirty_clips",
                        lambda group, progress_callback=None: calls.setdefault("dummy", group) and _noop())
    qt_app.engine.generate_dirty_clips.side_effect = lambda group, progress_callback=None: calls.setdefault(
        "kokoro", group)
    kokoro_future = concurrent.futures.Future()
    qt_app.engine.worker.run_coro.return_value = kokoro_future

    qt_app.timeline_dock.generate_dirty_clips_requested()

    assert [cid for cid, _t, _c in calls["kokoro"]] == [first.id]
    assert [cid for cid, _t, _c in calls["dummy"]] == [second.id]
    assert qt_app.is_busy()
    kokoro_future.set_result([{"clip_id": first.id, "success": False, "results": [], "error": "x",
                               "cancelled": False}])
    assert qt_app.is_busy()  # still waiting on the dummy batch
    sent["dummy"][0].set_result([{"clip_id": second.id, "success": False, "results": [], "error": "y",
                                  "cancelled": False}])
    qt_app.wait_for_project_io()
    assert not qt_app.is_busy()


async def _noop():
    return []


def test_characters_dialog_engine_picker_sets_the_characters_engine(qt_app):
    from kokoro_gui.qt.characters_dialog import CharactersDialog

    character = qt_app.document.characters[0]
    dialog = CharactersDialog(qt_app)
    assert dialog.engine_combo.currentData() == "kokoro"
    items = {dialog.engine_combo.itemData(i) for i in range(dialog.engine_combo.count())}
    assert items == set(engine_registry.list_engines())

    assert dialog.set_engine("dummy") is True
    assert character.backend_id == "dummy"
    assert dialog.engine_combo.currentData() == "dummy"
    assert "dummy" in [dialog.voice_combo.itemText(i) for i in range(dialog.voice_combo.count())]
    assert qt_app.backend.id == "dummy"


def test_a_linked_characters_engine_change_reaches_the_library(qt_app):
    from kokoro_gui.qt.characters_dialog import CharactersDialog

    dialog = CharactersDialog(qt_app)
    library_id = dialog.promote_current()
    dialog.set_engine("dummy")
    assert qt_app.character_library.get(library_id).backend_id == "dummy"


def test_sample_rate_is_the_highest_among_the_documents_engines(qt_app, monkeypatch):
    assert qt_app.project_sample_rate() == 24000
    _use_engine(qt_app, "dummy")
    monkeypatch.setattr(qt_app.backends["dummy"].engine, "SAMPLE_RATE", 44100, raising=False)
    assert qt_app.project_sample_rate() == 44100


def test_preview_uses_the_active_characters_engine_and_voice(qt_app):
    character = qt_app.document.characters[0]
    character.preset_data["voice"] = "bf_emma"
    qt_app.preview_conversion()
    (text, voice, speed, _path, _extra), kwargs = qt_app.engine.generate_preview.call_args
    assert voice == "bf_emma"
