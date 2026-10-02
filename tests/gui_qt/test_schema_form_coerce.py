"""`schema_form.coerce`: a wrong-typed value from a settings file ends at the
field's default instead of raising in a Qt setter."""
import pytest

from kokoro_gui.engines.base import ConfigField, ConfigFieldType
from kokoro_gui.qt.schema_form import SchemaFormWidget, coerce

T = ConfigFieldType


def _field(type_, default, **kw):
    return ConfigField("k", "K", type_, default=default, **kw)


@pytest.mark.parametrize("value", [None, [], {}, "abc", "", True, float("nan"), float("inf"), 10**400])
def test_a_wrong_int_value_ends_at_the_default(value):
    # 10**400 does not fit a float: it is refused, not clamped.
    assert coerce(_field(T.INT, 3, min=0, max=10), value) == 3


@pytest.mark.parametrize("value", [None, [], {}, "abc", True, float("nan"), float("-inf")])
def test_a_wrong_float_value_ends_at_the_default(value):
    assert coerce(_field(T.FLOAT, 1.5, min=0, max=3), value) == 1.5
    assert coerce(_field(T.SLIDER, 1.5, min=0, max=3), value) == 1.5


@pytest.mark.parametrize("value", [None, [], {}, "maybe", 2, 0.5, "yes"])
def test_a_wrong_bool_value_ends_at_the_default(value):
    assert coerce(_field(T.BOOL, True), value) is True
    assert coerce(_field(T.BOOL, False), value) is False


@pytest.mark.parametrize("value, expected", [
    (True, True), (False, False), ("true", True), ("False", False), ("1", True), (" 0 ", False)])
def test_a_bool_takes_real_bools_and_the_four_strings(value, expected):
    assert coerce(_field(T.BOOL, not expected), value) is expected


@pytest.mark.parametrize("type_", [T.TEXT, T.FILE])
@pytest.mark.parametrize("value", [None, [], {}, True])
def test_a_wrong_text_value_ends_at_the_default(type_, value):
    assert coerce(_field(type_, "fallback"), value) == "fallback"


def test_text_keeps_a_string_and_stringifies_a_number():
    assert coerce(_field(T.TEXT, ""), "hello") == "hello"
    assert coerce(_field(T.TEXT, ""), 12) == "12"


def test_numbers_convert_and_clamp_to_the_range():
    f = _field(T.INT, 5, min=1, max=64)
    assert coerce(f, 10**9) == 64
    assert coerce(f, -5) == 1
    assert coerce(f, "7") == 7
    assert coerce(f, 7.9) == 7
    g = _field(T.FLOAT, 1.0, min=0.5, max=2.0)
    assert coerce(g, 99) == 2.0
    assert coerce(g, "0.75") == 0.75


def test_a_none_default_falls_back_to_the_types_zero():
    assert coerce(_field(T.INT, None, min=2, max=9), "x") == 2
    assert coerce(_field(T.BOOL, None), "x") is False
    assert coerce(_field(T.TEXT, None), None) == ""


def test_choice_is_returned_unchanged():
    assert coerce(_field(T.CHOICE, "a", choices=[("A", "a")]), "zzz") == "zzz"


def _schema():
    return [
        ConfigField("count", "Count", T.INT, default=2, min=1, max=8),
        ConfigField("gain", "Gain", T.FLOAT, default=1.0, min=0.0, max=4.0),
        ConfigField("on", "On", T.BOOL, default=True),
        ConfigField("name", "Name", T.TEXT, default="x"),
        ConfigField("path", "Path", T.FILE, default=""),
        ConfigField("mode", "Mode", T.CHOICE, default="a", choices=[("A", "a"), ("B", "b")]),
    ]


def test_set_values_survives_every_field_type_with_a_wrong_value(qtbot):
    form = SchemaFormWidget(_schema(), {"count": "many", "gain": [], "on": "maybe", "name": None,
                                        "path": {}, "mode": 99})
    qtbot.addWidget(form)
    assert form.values() == {"count": 2, "gain": 1.0, "on": True, "name": "x", "path": "", "mode": "a"}


def test_set_values_with_the_constructor_and_later_calls(qtbot):
    form = SchemaFormWidget(_schema(), {})
    qtbot.addWidget(form)
    form.set_values({"count": 10**9, "gain": "2.5", "on": "false"})
    assert form.values()["count"] == 8
    assert form.values()["gain"] == 2.5
    assert form.values()["on"] is False
