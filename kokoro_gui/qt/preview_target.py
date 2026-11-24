"""What the Transport dock's Preview speaks, and how the dock says so.

`preview_source` is the one rule (the selected text, else the transcript,
else a sample line). `QtTTSApp.preview_conversion` speaks `preview_text`'s
answer and `describe` puts the same rule into words, without copying a
book-size transcript on every caret move.
"""
from __future__ import annotations

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
        what = "the selected text"
    elif source == TRANSCRIPT:
        length = editor.document().characterCount() - 1
        what = "the transcript"
    else:
        length, what = 0, "a sample line"
    if length > SPOKEN_LIMIT:
        what = f"the start of {what}"
    voice = f" as {shorten(character.name)}" if character is not None else ""
    return f"Preview speaks {what}{voice}."


TOOLTIP = ("Hear text in a character's voice without changing the project. It speaks the selected text, else the "
           f"transcript, up to {SPOKEN_LIMIT} characters (the first two parts of tagged text, {SPOKEN_LIMIT} each), "
           "with the default FX. It does not play the timeline's audio; Play does.")
