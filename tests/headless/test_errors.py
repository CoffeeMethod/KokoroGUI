import pytest

from kokoro_gui.headless import errors
from kokoro_gui.headless.errors import HeadlessError


@pytest.mark.parametrize("code,exit_code", [
    ("bad_arguments", 2),
    ("engine_not_installed", 3),
    ("unknown_voice", 4),
    ("model_not_downloaded", 5),
    ("project_locked", 6),
    ("generation_failed", 7),
    ("interrupted", 130),
])
def test_exit_codes_follow_the_plan(code, exit_code):
    assert HeadlessError(code, "m").exit_code == exit_code


def test_unlisted_code_exits_2():
    assert HeadlessError("something_new", "m").exit_code == 2


def test_explicit_exit_code_wins():
    assert HeadlessError("bad_arguments", "m", exit_code=9).exit_code == 9


def test_to_json_shape():
    err = HeadlessError("unknown_voice", "no voice 'zz'", valid=("af_heart", "am_adam"))
    assert err.to_json() == {"error": "unknown_voice", "message": "no voice 'zz'", "valid": ["af_heart", "am_adam"]}


def test_valid_defaults_to_none():
    assert HeadlessError("bad_arguments", "m").to_json()["valid"] is None


def test_is_an_exception_with_message():
    err = HeadlessError("project_locked", "locked")
    assert isinstance(err, Exception)
    assert err.message == "locked"
    assert "project_locked" in str(err)


def test_every_listed_code_is_a_known_exit():
    known = {errors.EXIT_BAD_ARGUMENTS, 3, 4, 5, 6, 7, 130}
    assert set(errors.EXIT_CODES.values()) <= known
