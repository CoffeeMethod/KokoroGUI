"""Document/Clip/Segment/Track/Character data model for the "DAW for text"
redesign (see Claude/PLAN_daw_ui_ux_redesign.md and Claude/Kokorogui grill
chat.md for the design this package implements).

This package deliberately has no Qt imports (same convention as
kokoro_gui/qt/spec.py) so it's testable as plain Python and importable from
non-GUI contexts. Nothing in kokoro_gui/qt wires this in yet - this is the
foundational data model only; the transcript editor, timeline widget, sync
layer, rescoped settings panel and consolidated action bar are later,
separately-planned passes that will consume it.
"""
