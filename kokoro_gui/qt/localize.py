"""Puts the interface in a language at launch. `main.py` calls `apply` once,
before the window exists; a change in Options > Language waits for the next
launch, because most text is set when a widget is built."""
from __future__ import annotations

from PySide6.QtCore import QLibraryInfo, QLocale, QTranslator

from kokoro_gui import i18n


def choices() -> list:
    """`(setting value, label)` for the language menus. A language is listed
    under its own name, so a person who can't read the current one finds it."""
    return [(i18n.SYSTEM_LANGUAGE, i18n.tr("System default")), *i18n.available_languages()]


def resolve(setting) -> str:
    """The language code `settings["language"]` asks for. A code with no
    file, or one that isn't text, is English."""
    if setting == i18n.SYSTEM_LANGUAGE:
        return i18n.match_language(QLocale.system().uiLanguages())
    codes = {code for code, _name in i18n.available_languages()}
    return setting if isinstance(setting, str) and setting in codes else i18n.SOURCE_LANGUAGE


def apply(app, setting) -> str:
    """Loads the language, Qt's own translation of its buttons and dialogs
    (OK, Cancel, the color picker) when Qt ships one, and the layout
    direction. Returns the language in use."""
    code = i18n.set_language(resolve(setting))
    if code == i18n.SOURCE_LANGUAGE:
        return code
    locale = QLocale(code)
    translator = QTranslator(app)
    if translator.load(locale, "qtbase", "_", QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)):
        app.installTranslator(translator)
    app.setLayoutDirection(locale.textDirection())
    return code
