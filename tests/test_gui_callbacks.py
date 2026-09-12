"""Tests for worker-to-Tk callback dispatch."""


def test_post_to_ui_runs_callback_on_main_loop(tts_app):
    called = []

    tts_app._post_to_ui(lambda: called.append(True))
    assert called == []

    tts_app._process_ui_callbacks()
    assert called == [True]
