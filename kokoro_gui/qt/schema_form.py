"""Generic renderer that walks a backend's `list[ConfigField]`
(kokoro_gui/engines/base.py) into Qt form rows.

This is the concrete payoff of workstream 1 for workstream 3a: the Qt
Generation dock's schema-covered fields (lang_code/voice/speed/split_pattern/
format/num_threads/caching) are built by walking whatever
`backend.get_config_schema()` returns, not hard-coded per engine. Rebuilding
this widget from a new backend's schema is what makes engine-switching
actually swap the visible fields (see docks/generation_dock.py and app.py's
`switch_engine`).
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
    QLineEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget, QFileDialog,
)

from kokoro_gui.engines.base import ConfigField, ConfigFieldType


class SchemaFormWidget(QWidget):
    """Renders `schema` (a `list[ConfigField]`) as `QFormLayout` rows grouped
    by `field.group` into `QGroupBox`es.

    `choices_overrides`: {key: [(label, value), ...]} for fields whose schema
    `choices` is `None` because the option set is GUI-resolved rather than
    engine data (today: "voice", "lang_code" - see
    kokoro_gui/engines/kokoro.py's `get_config_schema` docstring).

    `skip_keys`: field keys to omit entirely - the caller owns a dedicated
    widget for them instead (today: "lexicon", rendered as a CRUD list by
    docks/lexicon_dock.py rather than a single TEXT field).

    `on_change(key, value)`: called whenever a rendered field's value changes,
    so the owning dock can drive autosave the same way Tk's ctk var traces do.
    """

    def __init__(
        self,
        schema: list[ConfigField],
        values: dict[str, Any],
        choices_overrides: Optional[dict[str, list[tuple[str, Any]]]] = None,
        skip_keys: Optional[set[str]] = None,
        on_change: Optional[Callable[[str, Any], None]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._schema = schema
        self._choices_overrides = choices_overrides or {}
        self._skip_keys = skip_keys or set()
        self._on_change = on_change
        self._widgets: dict[str, QWidget] = {}
        self._getters: dict[str, Callable[[], Any]] = {}
        self._setters: dict[str, Callable[[Any], None]] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        groups: dict[str, QFormLayout] = {}
        group_order: list[str] = []
        for f in schema:
            if f.key in self._skip_keys:
                continue
            if f.group not in groups:
                box = QGroupBox(f.group)
                form = QFormLayout()
                box.setLayout(form)
                groups[f.group] = form
                group_order.append(f.group)
                outer.addWidget(box)
            self._add_field(groups[f.group], f)

        outer.addStretch(1)
        self.set_values(values)

    def _add_field(self, form: QFormLayout, f: ConfigField) -> None:
        choices = self._choices_overrides.get(f.key, f.choices)

        if f.type in (ConfigFieldType.CHOICE,) or choices is not None:
            combo = QComboBox()
            for label, value in (choices or []):
                combo.addItem(str(label), value)
            combo.currentIndexChanged.connect(lambda _i, k=f.key: self._emit_change(k))
            form.addRow(f.label, combo)
            self._widgets[f.key] = combo
            self._getters[f.key] = lambda c=combo: c.currentData()
            self._setters[f.key] = lambda v, c=combo: self._set_combo(c, v)

        elif f.type == ConfigFieldType.BOOL:
            box = QCheckBox()
            box.toggled.connect(lambda _v, k=f.key: self._emit_change(k))
            form.addRow(f.label, box)
            self._widgets[f.key] = box
            self._getters[f.key] = box.isChecked
            self._setters[f.key] = box.setChecked

        elif f.type == ConfigFieldType.INT:
            spin = QSpinBox()
            spin.setRange(int(f.min if f.min is not None else 0), int(f.max if f.max is not None else 100))
            spin.setSingleStep(int(f.step or 1))
            spin.valueChanged.connect(lambda _v, k=f.key: self._emit_change(k))
            form.addRow(f.label, spin)
            self._widgets[f.key] = spin
            self._getters[f.key] = spin.value
            self._setters[f.key] = spin.setValue

        elif f.type in (ConfigFieldType.FLOAT, ConfigFieldType.SLIDER):
            spin = QDoubleSpinBox()
            spin.setRange(float(f.min if f.min is not None else 0.0), float(f.max if f.max is not None else 1.0))
            spin.setSingleStep(float(f.step or 0.1))
            spin.setDecimals(3)
            spin.valueChanged.connect(lambda _v, k=f.key: self._emit_change(k))
            form.addRow(f.label, spin)
            self._widgets[f.key] = spin
            self._getters[f.key] = spin.value
            self._setters[f.key] = spin.setValue

        elif f.type == ConfigFieldType.FILE:
            row = QWidget()
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            edit = QLineEdit()
            browse = QPushButton("Browse...")

            def _browse(_checked=False, e=edit):
                path, _ = QFileDialog.getOpenFileName(self, "Select file")
                if path:
                    e.setText(path)

            browse.clicked.connect(_browse)
            layout.addWidget(edit)
            layout.addWidget(browse)
            edit.textChanged.connect(lambda _v, k=f.key: self._emit_change(k))
            form.addRow(f.label, row)
            self._widgets[f.key] = row
            self._getters[f.key] = edit.text
            self._setters[f.key] = edit.setText

        else:  # TEXT, or any future type - plain line edit fallback
            edit = QLineEdit()
            edit.textChanged.connect(lambda _v, k=f.key: self._emit_change(k))
            form.addRow(f.label, edit)
            self._widgets[f.key] = edit
            self._getters[f.key] = edit.text
            self._setters[f.key] = edit.setText

    @staticmethod
    def _set_combo(combo: QComboBox, value: Any) -> None:
        idx = combo.findData(value)
        if idx < 0 and combo.count() > 0:
            idx = 0
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _emit_change(self, key: str) -> None:
        if self._on_change is not None and key in self._getters:
            self._on_change(key, self._getters[key]())

    def widget_for(self, key: str) -> Optional[QWidget]:
        return self._widgets.get(key)

    def values(self) -> dict[str, Any]:
        return {k: getter() for k, getter in self._getters.items()}

    def set_values(self, values: dict[str, Any]) -> None:
        for k, setter in self._setters.items():
            if k in values:
                setter(values[k])

    def set_choices(self, key: str, choices: list[tuple[str, Any]], current: Any = None) -> None:
        """Repopulate a CHOICE field's options at runtime (used for "voice"
        when the active language or backend changes)."""
        combo = self._widgets.get(key)
        if not isinstance(combo, QComboBox):
            return
        combo.blockSignals(True)
        combo.clear()
        for label, value in choices:
            combo.addItem(str(label), value)
        combo.blockSignals(False)
        if current is not None:
            self._set_combo(combo, current)
        elif combo.count() > 0:
            combo.setCurrentIndex(0)
