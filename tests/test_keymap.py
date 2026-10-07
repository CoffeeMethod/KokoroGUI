"""The keyboard map table (`kokoro_gui/qt/keymap.py`): no Qt needed."""
import pytest

from kokoro_gui.qt import keymap

PLAIN_KEYS = {"Left", "Right", "Home", "End", "Delete", "J", "K", "L", "S", "F", "G", "Shift+Left", "Shift+Right",
              "M", "N", "Shift+N", "[", "]"}


def test_no_two_bindings_share_a_sequence():
    sequences = [b.sequence.lower() for b in keymap.KEYS]
    assert len(sequences) == len(set(sequences))


def test_ids_are_unique_and_make_valid_attribute_names():
    ids = [b.id for b in keymap.KEYS]
    assert len(ids) == len(set(ids))
    assert all(i.isidentifier() for i in ids)


def test_every_scope_is_known_and_every_label_is_filled_in():
    for binding in keymap.KEYS:
        assert binding.scope in keymap.SCOPES
        assert binding.label.strip()
        assert binding.slot_name.strip()
        assert binding.sequence.strip()


def test_plain_keys_listen_only_while_the_timeline_has_the_focus():
    """Grill PG3: a bare arrow, Home, End, Delete or letter never reaches past the timeline."""
    for binding in keymap.KEYS:
        if binding.sequence in PLAIN_KEYS:
            assert binding.scope == keymap.TIMELINE, binding.id


def test_every_timeline_action_has_a_ctrl_twin_that_works_anywhere():
    anywhere = {b.slot_name for b in keymap.bindings(keymap.APP)}
    for slot in ("go_to_start", "go_to_end", "go_to_previous_clip", "go_to_next_clip",
                 "go_to_previous_marker", "go_to_next_marker"):
        assert slot in anywhere


def test_the_anywhere_keys_all_hold_ctrl():
    for binding in keymap.bindings(keymap.APP):
        assert "Ctrl" in binding.sequence.split("+"), binding.id


def test_only_esc_waits_for_a_generate_and_it_is_not_a_timeline_key():
    waiting = [b for b in keymap.KEYS if b.while_generating]
    assert [b.sequence for b in waiting] == ["Esc"]
    assert waiting[0].scope == keymap.WINDOW


def test_the_split_and_space_ids_keep_the_attribute_names_older_code_reads():
    ids = {b.id: b.sequence for b in keymap.KEYS}
    assert ids["space"] == "Space"
    assert ids["ctrl_space"] == "Ctrl+Space"
    assert ids["split"] == "S"


def test_bindings_filters_by_scope():
    assert keymap.bindings() == list(keymap.KEYS)
    assert all(b.scope == keymap.TIMELINE for b in keymap.bindings(keymap.TIMELINE))


@pytest.mark.parametrize("now,expected", [(0.0, None), (1.0, 0.0), (1.5, 1.0), (7.0, 5.0), (9.0, 5.0)])
def test_previous_time(now, expected):
    assert keymap.previous_time([0.0, 1.0, 5.0], now) == expected


@pytest.mark.parametrize("now,expected", [(0.0, 1.0), (1.0, 5.0), (4.99, 5.0), (5.0, None), (9.0, None)])
def test_next_time(now, expected):
    assert keymap.next_time([0.0, 1.0, 5.0], now) == expected


def test_a_jump_from_exactly_a_start_goes_to_the_neighbour():
    assert keymap.previous_time([0.0, 2.0, 4.0], 2.0) == 0.0
    assert keymap.next_time([0.0, 2.0, 4.0], 2.0) == 4.0


def test_jumps_over_nothing_return_none():
    assert keymap.previous_time([], 3.0) is None
    assert keymap.next_time([], 3.0) is None
