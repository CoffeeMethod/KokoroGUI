"""Tooltips shared by a form row's label and its control."""
from __future__ import annotations

from PySide6.QtWidgets import QFormLayout, QWidget


def _control_tip(widget: QWidget | None) -> str:
    """The tooltip of a row's control, or of the first control inside it
    when the row holds several."""
    if widget is None:
        return ""
    if widget.toolTip():
        return widget.toolTip()
    return next((child.toolTip() for child in widget.findChildren(QWidget) if child.toolTip()), "")


def label_rows(root: QWidget) -> None:
    """Gives every form row's label under `root` its control's tooltip, so a
    person who points at the words gets it too. A label that has a tooltip
    keeps it."""
    for form in root.findChildren(QFormLayout):
        for row in range(form.rowCount()):
            label = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
            field = form.itemAt(row, QFormLayout.ItemRole.FieldRole)
            if label is None or field is None or label.widget() is None or label.widget().toolTip():
                continue
            tip = _control_tip(field.widget())
            if not tip and field.layout() is not None:
                widgets = (field.layout().itemAt(i).widget() for i in range(field.layout().count()))
                tip = next((t for t in map(_control_tip, widgets) if t), "")
            if tip:
                label.widget().setToolTip(tip)
