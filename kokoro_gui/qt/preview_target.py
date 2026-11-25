"""What the Transport dock's Preview speaks, and how the dock says so.

`preview_source` is the one rule (the selected text, else the transcript,
else a sample line). `QtTTSApp.preview_conversion` speaks `preview_text`'s
answer and `describe` puts the same rule into words, without copying a
book-size transcript on every caret move.
"""
from __future__ import annotations

from kokoro_gui.i18n import tr
from kokoro_gui.qt.scope_target import shorten

SELECTION = "selection"
TRANSCRIPT = "transcript"
SAMPLE = "sample"

# What is handed to the engine.
PREVIEW_LIMIT = 1000
# What the engine reads of it (`ConversionMixin.generate_preview`): this many
# characters of each of the first two tagged stretches, so of plain text only
# this many.
SPOKEN_LIMIT = 500

SAMPLE_TEXT = ("This is a sample audio preview using the Koh-koh-ro Tea-Tea-S engine. "
               "It demonstrates the voice quality and speed settings.")


def preview_source(editor) -> str:
    if editor.textCursor().hasSelection():
        return SELECTION
    block = editor.document().firstBlock()
    while block.isValid():
        if block.text().strip():
            return TRANSCRIPT
        block = block.next()
    return SAMPLE


def preview_text(editor) -> str:
    source = preview_source(editor)
    if source == SELECTION:
        text = editor.textCursor().selectedText().replace(" ", "\n")
    elif source == TRANSCRIPT:
        text = editor.toPlainText().strip()
    else:
        text = SAMPLE_TEXT
    return text[:PREVIEW_LIMIT]


def describe(editor, character) -> str:
    """"Preview speaks the selected text as Marta." """
    source = preview_source(editor)
    cursor = editor.textCursor()
    if source == SELECTION:
        length = cursor.selectionEnd() - cursor.selectionStart()
        what = tr("the selected text")
    elif source == TRANSCRIPT:
        length = editor.document().characterCount() - 1
        what = tr("the transcript")
    else:
        length, what = 0, tr("a sample line")
    if length > SPOKEN_LIMIT:
        what = tr("the start of {what}", what=what)
    if character is None:
        return tr("Preview speaks {what}.", what=what)
    return tr("Preview speaks {what} as {name}.", what=what, name=shorten(character.name))


def tooltip() -> str:
    return tr("Hear text in a character's voice without changing the project. It speaks the selected text, else the "
              "transcript, up to {limit} characters (the first two parts of tagged text, {limit} each), "
              "with the default FX. It does not play the timeline's audio; Play does.", limit=SPOKEN_LIMIT)
