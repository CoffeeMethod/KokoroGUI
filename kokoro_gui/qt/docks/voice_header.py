"""The Engine row at the top of the Voices tab (grill EN3), shared by the
Mixing and Voice Reference docks: every engine with a voice editor, the
dock's own engine selected. A pick calls `app.set_voices_engine` through
`QTimer.singleShot(0, ...)`, since the switch can delete the dock the combo
sits in."""
from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel

from kokoro_gui.engines import registry as engine_registry


def engine_header(app, backend_id: str) -> tuple:
    """`(layout, combo)` for a voice dock editing `backend_id`'s voices."""
    row = QHBoxLayout()
    row.addWidget(QLabel("Engine:"))
    combo = QComboBox()
    combo.setToolTip("Whose voices to edit. Selecting a clip or character switches back to its engine.")
    for engine_id in app.voice_editor_engines():
        combo.addItem(engine_registry.get_display_name(engine_id), engine_id)
    index = combo.findData(backend_id)
    if index >= 0:
        combo.setCurrentIndex(index)

    def _picked(_index):
        engine_id = combo.currentData()
        QTimer.singleShot(0, lambda: app.set_voices_engine(engine_id))

    combo.activated.connect(_picked)
    row.addWidget(combo, 1)
    return row, combo
