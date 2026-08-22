"""Engine-picker switch behavior, including re-rendering the Generation
dock's schema-driven fields for the newly-active backend. See app.py's
`switch_engine` docstring."""
from kokoro_gui.engines import registry as engine_registry


def test_dummy_and_kokoro_both_registered():
    assert {"kokoro", "dummy"} <= set(engine_registry.list_engines())


def test_kokoro_backend_shows_mixing_dock(qt_app):
    assert qt_app.backend.id == "kokoro"
    assert qt_app.mixing_dock is not None


def test_switch_to_dummy_hides_mixing_dock(qt_app):
    qt_app.switch_engine("dummy")
    assert qt_app.backend.id == "dummy"
    assert qt_app.mixing_dock is None


def test_switch_back_to_kokoro_shows_mixing_dock_again(qt_app):
    qt_app.switch_engine("dummy")
    qt_app.switch_engine("kokoro")
    assert qt_app.backend.id == "kokoro"
    assert qt_app.mixing_dock is not None


def test_switch_engine_rebuilds_schema_form_for_new_backend(qt_app):
    """The Generation dock's schema-driven fields must reflect the
    newly-active backend's schema, not stay frozen at whatever the first
    backend built."""
    original_form = qt_app.generation_dock.schema_form
    qt_app.switch_engine("dummy")
    assert qt_app.generation_dock.schema_form is not original_form

    dummy_schema_keys = {f.key for f in qt_app.backend.get_config_schema()}
    assert "lexicon" not in dummy_schema_keys  # dummy backend has no lexicon field
    rendered_keys = set(qt_app.generation_dock.schema_form.values().keys())
    assert rendered_keys == dummy_schema_keys - {"lexicon"}


def test_switch_engine_refused_while_job_running(qt_app):
    qt_app.cancel_btn.setEnabled(True)  # simulate a job in flight
    original_backend_id = qt_app.backend.id
    qt_app.switch_engine("dummy")
    assert qt_app.backend.id == original_backend_id


def test_engine_picker_combo_lists_all_registered_engines(qt_app):
    items = [qt_app.engine_picker.itemText(i) for i in range(qt_app.engine_picker.count())]
    expected = {engine_registry.get_display_name(eid) for eid in engine_registry.list_engines()}
    assert set(items) == expected
