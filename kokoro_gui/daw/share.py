"""Share characters and FX between machines (plan 40).

A share file is a zip named `<name>.kgshare`. It is not a `.tbaw`: nothing here
touches the bundle code, and the file never holds a project. Entries:

- `share.json`: `{"format": "kokorogui-share", "version": 1, "characters": [...], "fx": [names]}`
- `fx/<name>.json`: one FX (the values of a `presets/fx/<name>.json`)
- `fx/ir/<name>.wav`: an impulse response an FX names
- `voices/<engine id>/<file>`: the files of a character's voice, as the engine's
  voice store names them (`.pt` for a Mix, `.wav` and `.txt` for a Reference)

A share file is untrusted input, read the way a `.tbaw` is and a little
stricter. `inspect_share` checks every entry name the way
`qt/project.py::validate_entries` does (no absolute path, drive letter, `..` or
symlink), refuses an encrypted or duplicated entry, caps the entry count, each
entry's size and the total, and refuses a `.pt` when this machine's torch is
older than 2.6 (`torch.load` only defaults to `weights_only=True` from there).
Only the entry names this module knows are used, and every file it writes is
named from a validated stem, never from the entry name. Character dicts go
through `ALLOWED_PRESET_KEYS` and a type check, FX through
`filter_fx_preset_values` (the same clamping the Audio FX dock and the engine
use). Nothing is written until `import_share` runs.

Qt-free like the rest of `kokoro_gui/daw/`. Voice files go through each
engine's voice store (`kokoro_gui/engines/voice_store.py`), found by
`store_for(engine_id)`.
"""
from __future__ import annotations

import copy
import json
import logging
import math
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass, field

from kokoro_gui.daw.models import DEFAULT_HIGHLIGHT_PALETTE, Character
from kokoro_gui.engine import presets as engine_presets
from kokoro_gui.engine.paths import ensure_private_dir
from kokoro_gui.engine.presets import ALLOWED_PRESET_KEYS, filter_allowed_keys, filter_fx_preset_values
from kokoro_gui.engines.registry import DEFAULT_ENGINE_ID

FORMAT = "kokorogui-share"
VERSION = 1
EXTENSION = ".kgshare"
SHARE_JSON = "share.json"

MAX_ENTRIES = 500
MAX_TOTAL_BYTES = 200 * 1024 * 1024  # uncompressed, all entries
MAX_CHARACTERS = 200
MAX_NAME_CHARS = 100
MAX_VARIANTS = 64
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_FX_BYTES = 256 * 1024
MAX_IR_BYTES = 50 * 1024 * 1024
MAX_PT_BYTES = 64 * 1024 * 1024
MAX_WAV_BYTES = 100 * 1024 * 1024
MAX_TXT_BYTES = 1024 * 1024

KIND_CHARACTER = "character"
KIND_FX = "fx"
KIND_IR = "ir"
KIND_VOICE = "voice"

# What to do with an item the import is asked about. "keep" imports it, under a
# numbered name when one is taken ("Hall (2)"); "replace" overwrites the one
# that is there; "skip" leaves it out.
KEEP, REPLACE, SKIP = "keep", "replace", "skip"

_log = logging.getLogger(__name__)

_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f]')
_RESERVED = frozenset({"con", "prn", "aux", "nul", *(f"com{n}" for n in range(1, 10)),
                       *(f"lpt{n}" for n in range(1, 10))})
_ENGINE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")

_VOICE_EXTENSIONS = {".pt": MAX_PT_BYTES, ".wav": MAX_WAV_BYTES, ".txt": MAX_TXT_BYTES}

_PRESET_STRINGS = frozenset({"voice", "format", "fx_preset"})
_PRESET_BOOLS = frozenset({"normalize", "trim", "apply_fx"})
_PRESET_RANGES = {"speed": (0.1, 4.0), "volume": (0.0, 4.0), "pitch": (-12.0, 12.0)}
_PRESET_STRING_CHARS = 200
_PLACEHOLDER_FX = "Select FX Preset..."  # what older presets store for "no FX"


class ShareError(Exception):
    """A share file this version can't or won't read, or an export that can't be
    written; the message is for the user."""


# -- names ----------------------------------------------------------------------------


def clean_stem(name) -> str | None:
    """`name` when it is safe to use as a file stem, else None. Stricter than a
    sanitiser: an unsafe name is refused rather than rewritten, so the name a
    character stores is the name the file gets."""
    if not isinstance(name, str) or not name or name != name.strip() or len(name) > MAX_NAME_CHARS:
        return None
    if name in (".", "..") or name.startswith(".") or name.endswith(".") or _UNSAFE.search(name):
        return None
    if name.split(".")[0].casefold() in _RESERVED:
        return None
    return name


def _entry_name_is_safe(name: str) -> bool:
    """The rule of `qt/project.py::_entry_name_is_safe`, copied (this module
    stays Qt-free and does not import the project code)."""
    normalized = name.replace("\\", "/")
    if not normalized or normalized.endswith("/") and normalized.count("/") == 0:
        return True
    if os.path.isabs(normalized) or normalized.startswith("/"):
        return False
    if len(normalized) >= 2 and normalized[1] == ":":
        return False  # drive-relative on Windows (C:foo)
    return not any(part == ".." for part in normalized.split("/"))


def _is_symlink_entry(info: zipfile.ZipInfo) -> bool:
    return ((info.external_attr >> 16) & 0o170000) == 0o120000


def _torch_weights_only_available() -> bool:
    """Whether `torch.load` defaults to `weights_only=True` here (torch 2.6 or
    later). A test patches this name."""
    try:
        import torch

        major, minor = (int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
        return (major, minor) >= (2, 6)
    except Exception:  # noqa: BLE001 - no torch, or a version string we can't read
        return False


def _validate_entries(zf: zipfile.ZipFile) -> list:
    """Every entry, or `ShareError` for one that would write outside a folder
    (absolute on either OS, drive-relative, `..`, a symlink), an encrypted one
    or a name used twice. Mirrors `project.validate_entries`."""
    root = os.path.realpath(os.path.join(tempfile.gettempdir(), "kgshare-root"))
    entries = []
    seen = set()
    for info in zf.infolist():
        name = info.filename
        if not _entry_name_is_safe(name):
            raise ShareError(f"Refusing to open: the share file has an unsafe entry name ({name!r}).")
        if _is_symlink_entry(info):
            raise ShareError(f"Refusing to open: the share file contains a symlink ({name!r}).")
        target = os.path.realpath(os.path.join(root, *name.replace("\\", "/").split("/")))
        if target != root and not target.startswith(root + os.sep):
            raise ShareError(f"Refusing to open: an entry resolves outside its folder ({name!r}).")
        if info.flag_bits & 0x1:
            raise ShareError(f"Refusing to open: the share file has an encrypted entry ({name!r}).")
        if name in seen:
            raise ShareError(f"Refusing to open: the share file lists {name!r} twice.")
        seen.add(name)
        entries.append(info)
    return entries


def classify_entry(name: str):
    """What a safe entry name is: `("manifest",)`, `("fx", name)`, `("ir", name)`,
    `("voice", engine_id, name, ext)`, or None for a name this module does not
    use (a directory, a stray file, a name that is not a clean stem)."""
    parts = name.replace("\\", "/").split("/")
    if parts == [SHARE_JSON]:
        return ("manifest",)
    if len(parts) == 2 and parts[0] == "fx" and parts[1].endswith(".json"):
        stem = clean_stem(parts[1][: -len(".json")])
        return ("fx", stem) if stem else None
    if len(parts) == 3 and parts[:2] == ["fx", "ir"] and parts[2].endswith(".wav"):
        stem = clean_stem(parts[2][: -len(".wav")])
        return ("ir", stem) if stem else None
    if len(parts) == 3 and parts[0] == "voices" and _ENGINE_ID.match(parts[1]):
        stem, ext = os.path.splitext(parts[2])
        stem = clean_stem(stem)
        if stem and ext in _VOICE_EXTENSIONS:
            return ("voice", parts[1], stem, ext)
    return None


def _entry_cap(kind: tuple) -> int:
    if kind[0] == "manifest":
        return MAX_MANIFEST_BYTES
    if kind[0] == "fx":
        return MAX_FX_BYTES
    if kind[0] == "ir":
        return MAX_IR_BYTES
    return _VOICE_EXTENSIONS[kind[3]]


# -- characters -----------------------------------------------------------------------


def _is_number(value) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _clean_text(value, limit: int = _PRESET_STRING_CHARS) -> str | None:
    if isinstance(value, str) and value and len(value) <= limit and not _CONTROL.search(value):
        return value
    return None


def clean_preset_data(data, warnings: list | None = None, who: str = "") -> dict:
    """A character's `preset_data` fit to import: only `ALLOWED_PRESET_KEYS`, each
    of the type it has in the app (a string, a bool or a finite number clamped
    to its range). A value of the wrong type is dropped and the key falls back
    to its default; `warnings` gets one line per drop."""
    if not isinstance(data, dict):
        return {}
    out = {}
    for key, value in filter_allowed_keys(data, ALLOWED_PRESET_KEYS).items():
        cleaned = None
        if key in _PRESET_STRINGS:
            cleaned = _clean_text(value, 16 if key == "format" else _PRESET_STRING_CHARS)
            if key == "fx_preset" and cleaned in (None, _PLACEHOLDER_FX):
                cleaned = None
            elif key == "fx_preset" and clean_stem(cleaned) is None:
                cleaned = None
        elif key in _PRESET_BOOLS:
            cleaned = value if isinstance(value, bool) else None
        elif key in _PRESET_RANGES and _is_number(value):
            low, high = _PRESET_RANGES[key]
            cleaned = max(low, min(high, value))
        if cleaned is None:
            if warnings is not None and key != "fx_preset":
                warnings.append(f"{who or 'A character'}: ignored a bad value for {key!r}")
            continue
        out[key] = cleaned
    dropped = [k for k in data if k not in ALLOWED_PRESET_KEYS]
    if dropped and warnings is not None:
        warnings.append(f"{who or 'A character'}: ignored unknown settings ({', '.join(sorted(map(str, dropped))[:5])})")
    return out


def clean_character(data, warnings: list | None = None) -> Character | None:
    """A `Character` from one entry of `share.json`, or None when it has no
    usable name. Unknown keys are dropped, wrong-typed ones fall back to the
    default, and the character gets no id or library id (the library mints
    one on import)."""
    if not isinstance(data, dict):
        return None
    name = " ".join(str(data.get("name", "")).split()) if isinstance(data.get("name"), str) else ""
    if not name or _CONTROL.search(name):
        return None
    name = name[:MAX_NAME_CHARS]
    color = data.get("highlight_color")
    if not (isinstance(color, str) and _COLOR.match(color)):
        color = DEFAULT_HIGHLIGHT_PALETTE[0]
    backend_id = data.get("backend_id")
    if not (isinstance(backend_id, str) and _ENGINE_ID.match(backend_id)):
        backend_id = DEFAULT_ENGINE_ID
    variants = {}
    raw_variants = data.get("variants")
    if isinstance(raw_variants, dict):
        for key, value in raw_variants.items():
            key, value = _clean_text(key, MAX_NAME_CHARS), _clean_text(value)
            if key and value and len(variants) < MAX_VARIANTS:
                variants[key] = value
    return Character(name=name, preset_data=clean_preset_data(data.get("preset_data"), warnings, name),
                     highlight_color=color, backend_id=backend_id, variants=variants)


def character_to_share(character: Character) -> dict:
    """The dict `share.json` holds for `character`: the settings a character is
    made of and nothing that belongs to one machine or project (no id, library
    id or unknown keys)."""
    clean = clean_character({"name": character.name, "preset_data": character.to_preset_dict(),
                             "highlight_color": character.highlight_color, "backend_id": character.backend_id,
                             "variants": character.variants})
    if clean is None:
        clean = Character(name="Character")
    return {"name": clean.name, "backend_id": clean.backend_id, "highlight_color": clean.highlight_color,
            "preset_data": clean.preset_data, "variants": clean.variants}


def voice_names_of(character: Character) -> set:
    """The voice names a character speaks with: its voice and its variants'."""
    names = set()
    voice = (character.preset_data or {}).get("voice")
    if isinstance(voice, str) and voice:
        names.add(voice)
    for value in (character.variants or {}).values():
        if isinstance(value, str) and value:
            names.add(value)
    return names


def default_store_for(engine_id: str):
    """The voice store of the engine registered as `engine_id`, or None for an
    engine this install lacks or one with no voice files (a named-voice
    engine). Read off the factory, so no model is built."""
    from kokoro_gui.engines import registry

    return getattr(registry.get_factory(engine_id), "voice_store", None)


# -- export ---------------------------------------------------------------------------


@dataclass
class ExportReport:
    path: str
    characters: int = 0
    fx: list = field(default_factory=list)
    irs: list = field(default_factory=list)
    voices: list = field(default_factory=list)  # "engine/name"
    warnings: list = field(default_factory=list)


def _fx_source(name: str, project_dir, fx_dir: str) -> str | None:
    """The file FX `name` loads from: the project's `fx/` first, then `fx_dir`.
    The name is a clean stem and the result must stay inside its folder."""
    stem = clean_stem(name)
    if stem is None:
        return None
    directories = [os.path.join(project_dir, "fx")] if project_dir else []
    directories.append(fx_dir)
    for directory in directories:
        root = os.path.realpath(directory)
        path = os.path.realpath(os.path.join(root, f"{stem}.json"))
        if path.startswith(root + os.sep) and os.path.isfile(path):
            return path
    return None


def export_share(path: str, characters=(), *, fx_names=(), include_fx: bool = True, include_voices: bool = True,
                 project_dir: str | None = None, fx_dir: str | None = None, store_for=None) -> ExportReport:
    """Writes a share file at `path` (a temp file and `os.replace`, so a failed
    export leaves any older file alone).

    `characters` go in as `share.json` entries. With `include_fx`, the FX each
    one names go in too, with the impulse response an FX names; `fx_names` adds
    FX no character needs (the FX dock's export). With `include_voices`, the
    files of each character's voice and variants go in, found through
    `store_for(engine_id)` (default: the registered engine's voice store). The
    project's own `fx/` and voice folders are read first, the global stores
    second. Something a character names that can't be found is skipped and
    named in the report's warnings. Raises `ShareError` when an item is over
    its size limit or the file would be."""
    fx_dir = fx_dir or os.path.join("presets", "fx")
    ir_dir = os.path.join(fx_dir, "ir")
    store_for = store_for or default_store_for
    report = ExportReport(path=path)
    characters = list(characters)
    shared = [character_to_share(c) for c in characters]
    report.characters = len(shared)

    # name -> (arcname, payload bytes or source path)
    members: dict = {}
    wanted_fx = []
    for name in list(fx_names) + [(c.preset_data or {}).get("fx_preset") for c in characters if include_fx]:
        if isinstance(name, str) and name and name != _PLACEHOLDER_FX and name not in wanted_fx:
            wanted_fx.append(name)
    for name in wanted_fx:
        source = _fx_source(name, project_dir, fx_dir)
        if source is None:
            report.warnings.append(f"FX {name!r} wasn't found; left out.")
            continue
        try:
            with open(source, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, ValueError, RecursionError) as e:
            report.warnings.append(f"FX {name!r} couldn't be read ({e}); left out.")
            continue
        values = filter_fx_preset_values(raw)
        members[f"fx/{name}.json"] = ("bytes", json.dumps(values, indent=2).encode("utf-8"))
        report.fx.append(name)
        ir = values.get("convolution_ir")
        if ir:
            ir_stem = clean_stem(ir)
            ir_path = engine_presets.resolve_ir(ir, project_dir, ir_dir) if ir_stem else None
            if ir_path is None:
                report.warnings.append(f"Impulse response {ir!r} (used by FX {name!r}) wasn't found; left out.")
            elif f"fx/ir/{ir_stem}.wav" not in members:
                members[f"fx/ir/{ir_stem}.wav"] = ("file", ir_path)
                report.irs.append(ir_stem)

    if include_voices:
        by_engine: dict = {}
        for character in characters:
            by_engine.setdefault(character.backend_id or DEFAULT_ENGINE_ID, set()).update(voice_names_of(character))
        for engine_id, names in sorted(by_engine.items()):
            store = store_for(engine_id) if _ENGINE_ID.match(engine_id) else None
            if store is None:
                continue
            for name in sorted(names):
                assets = store.bundle_assets(name, project_dir)
                if not assets:
                    continue
                added = False
                for asset in assets:
                    file_name = os.path.basename(asset.bundle_path)
                    kind = classify_entry(f"voices/{engine_id}/{file_name}")
                    if kind is None:
                        report.warnings.append(f"Voice file {file_name!r} has a name a share file can't carry; left out.")
                        continue
                    members[f"voices/{engine_id}/{file_name}"] = ("file", asset.source_path)
                    added = True
                if added:
                    report.voices.append(f"{engine_id}/{name}")

    manifest = {"format": FORMAT, "version": VERSION, "characters": shared, "fx": sorted(report.fx)}
    manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
    total = len(manifest_bytes)
    for arcname, (how, payload) in members.items():
        size = len(payload) if how == "bytes" else os.path.getsize(payload)
        kind = classify_entry(arcname)
        if kind is None or size > _entry_cap(kind):
            raise ShareError(f"{arcname} is too big for a share file ({size // (1024 * 1024)} MB).")
        total += size
    if len(members) + 1 > MAX_ENTRIES:
        raise ShareError(f"A share file holds at most {MAX_ENTRIES} files.")
    if total > MAX_TOTAL_BYTES:
        raise ShareError(f"Everything selected is {total // (1024 * 1024)} MB; a share file holds at most "
                         f"{MAX_TOTAL_BYTES // (1024 * 1024)} MB.")

    tmp = path + ".tmp"
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(SHARE_JSON, manifest_bytes)
            for arcname in sorted(members):
                how, payload = members[arcname]
                if how == "bytes":
                    zf.writestr(arcname, payload)
                else:
                    zf.write(payload, arcname)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return report


# -- inspect --------------------------------------------------------------------------


@dataclass
class SharedVoice:
    """One voice in a share file: the files of `name` for `engine_id`, by
    extension (`.pt`, or `.wav` and `.txt`)."""
    engine_id: str
    name: str
    files: dict = field(default_factory=dict)  # ext -> entry name


@dataclass
class ShareInfo:
    path: str
    characters: list = field(default_factory=list)  # [Character], no ids
    fx: dict = field(default_factory=dict)  # name -> cleaned values
    irs: list = field(default_factory=list)  # names
    voices: list = field(default_factory=list)  # [SharedVoice]
    warnings: list = field(default_factory=list)
    ignored: int = 0  # entries this version has no use for

    def is_empty(self) -> bool:
        return not (self.characters or self.fx or self.irs or self.voices)


def _read_member(zf: zipfile.ZipFile, info: zipfile.ZipInfo, cap: int) -> bytes:
    with zf.open(info) as f:
        data = f.read(cap + 1)
    if len(data) > cap:
        raise ShareError(f"Refusing to open: {info.filename!r} is bigger than it says.")
    return data


def _read_json(zf: zipfile.ZipFile, info: zipfile.ZipInfo, cap: int):
    try:
        return json.loads(_read_member(zf, info, cap).decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None


def _looks_like_wav(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> bool:
    with zf.open(info) as f:
        head = f.read(12)
    return len(head) == 12 and head[:4] == b"RIFF" and head[8:12] == b"WAVE"


def inspect_share(path: str) -> ShareInfo:
    """Opens the share file at `path` and reads what it holds, writing nothing.
    Raises `ShareError` (with a message for the user) for a file that isn't a
    share file or that breaks a limit."""
    if not path or not os.path.isfile(path):
        raise ShareError(f"{path} doesn't exist.")
    try:
        zf = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile):
        raise ShareError(f"{os.path.basename(path)} isn't a readable share file.") from None
    with zf:
        entries = _validate_entries(zf)
        if len(entries) > MAX_ENTRIES:
            raise ShareError(f"Refusing to open: the share file has {len(entries)} entries (the limit is {MAX_ENTRIES}).")
        total = sum(e.file_size for e in entries)
        if total > MAX_TOTAL_BYTES:
            raise ShareError(f"Refusing to open: the share file expands to {total // (1024 * 1024)} MB "
                             f"(the limit is {MAX_TOTAL_BYTES // (1024 * 1024)} MB).")
        if any(e.filename.endswith(".pt") for e in entries) and not _torch_weights_only_available():
            raise ShareError("This share file carries a voice Mix (.pt) and this machine's torch is older than 2.6, "
                             "which can't load it safely. Upgrade torch to import it.")
        by_name = {}
        info = ShareInfo(path=os.path.abspath(path))
        for entry in entries:
            kind = classify_entry(entry.filename)
            if kind is None:
                if not entry.filename.replace("\\", "/").endswith("/"):
                    info.ignored += 1
                continue
            if entry.file_size > _entry_cap(kind):
                raise ShareError(f"Refusing to open: {entry.filename!r} is {entry.file_size // (1024 * 1024)} MB, "
                                 f"over the {_entry_cap(kind) // (1024 * 1024) or 1} MB limit for that kind of file.")
            by_name[entry.filename.replace("\\", "/")] = (kind, entry)
        if SHARE_JSON not in by_name:
            raise ShareError(f"{os.path.basename(path)} isn't a KokoroGUI share file (no {SHARE_JSON}).")
        manifest = _read_json(zf, by_name[SHARE_JSON][1], MAX_MANIFEST_BYTES)
        if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
            raise ShareError(f"{os.path.basename(path)} isn't a KokoroGUI share file.")
        version = manifest.get("version")
        if isinstance(version, bool) or not isinstance(version, int) or version > VERSION:
            raise ShareError("This share file was made by a newer KokoroGUI. Update to import it.")

        raw_characters = manifest.get("characters")
        for raw in raw_characters if isinstance(raw_characters, list) else []:
            if len(info.characters) >= MAX_CHARACTERS:
                info.warnings.append(f"Only the first {MAX_CHARACTERS} characters are read.")
                break
            character = clean_character(raw, info.warnings)
            if character is None:
                info.warnings.append("A character with no usable name was ignored.")
            else:
                info.characters.append(character)

        voices: dict = {}
        for entry_name, (kind, entry) in sorted(by_name.items()):
            if kind[0] == "fx":
                raw = _read_json(zf, entry, MAX_FX_BYTES)
                if not isinstance(raw, dict):
                    info.warnings.append(f"FX {kind[1]!r} isn't a readable FX file; ignored.")
                    continue
                info.fx[kind[1]] = filter_fx_preset_values(raw)
            elif kind[0] == "ir":
                if not _looks_like_wav(zf, entry):
                    info.warnings.append(f"Impulse response {kind[1]!r} isn't a wav file; ignored.")
                    continue
                info.irs.append(kind[1])
            elif kind[0] == "voice":
                _, engine_id, name, ext = kind
                if ext == ".wav" and not _looks_like_wav(zf, entry):
                    info.warnings.append(f"Voice file {name}.wav isn't a wav file; ignored.")
                    continue
                voices.setdefault((engine_id, name), SharedVoice(engine_id, name)).files[ext] = entry_name
        info.voices = [voices[key] for key in sorted(voices)]
    return info


# -- import ---------------------------------------------------------------------------


@dataclass
class ImportItem:
    """One thing a share file can add. `clash` is True when something of that
    name is already here and differs, `same` when it is already here and
    identical (nothing to do). `unavailable` says why it can't be imported
    (an engine this install lacks)."""
    kind: str
    key: str
    name: str
    detail: str = ""
    clash: bool = False
    same: bool = False
    unavailable: str = ""


@dataclass
class ImportResult:
    imported: list = field(default_factory=list)  # one line per item written
    renamed: dict = field(default_factory=dict)  # key -> the name it was stored under
    skipped: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def _free_name(base: str, taken) -> str:
    """`base`, or `base (2)`, `base (3)` ... the first `taken(name)` is False for."""
    if not taken(base):
        return base
    n = 2
    while True:
        suffix = f" ({n})"
        name = base[: MAX_NAME_CHARS - len(suffix)] + suffix
        if not taken(name):
            return name
        n += 1


def _inside(root: str, name: str, ext: str) -> str | None:
    """`<root>/<name><ext>` when `name` is a clean stem and the path stays in `root`."""
    stem = clean_stem(name)
    if stem is None:
        return None
    real_root = os.path.realpath(root)
    path = os.path.realpath(os.path.join(real_root, f"{stem}{ext}"))
    return path if path.startswith(real_root + os.sep) else None


def _taken(root: str, name: str, ext: str) -> bool:
    path = _inside(root, name, ext)
    return path is not None and os.path.exists(path)


def _zip_matches_file(zf: zipfile.ZipFile, entry_name: str, path: str) -> bool:
    try:
        if os.path.getsize(path) != zf.getinfo(entry_name).file_size:
            return False
        with zf.open(entry_name) as a, open(path, "rb") as b:
            while True:
                chunk = a.read(1 << 20)
                if chunk != b.read(len(chunk)):
                    return False
                if not chunk:
                    return True
    except OSError:
        return False


def _fx_values_on_disk(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return filter_fx_preset_values(json.load(f))
    except (OSError, ValueError, RecursionError):
        return None


def plan_import(info: ShareInfo, *, library, fx_dir: str | None = None, store_for=None) -> list:
    """The items `info` could add, each marked clash / same / unavailable against
    this machine's library, FX folder and voice stores. Reads, never writes."""
    fx_dir = fx_dir or os.path.join("presets", "fx")
    store_for = store_for or default_store_for
    items = []
    names_here = {c.name.casefold() for c in library.list()}
    for index, character in enumerate(info.characters):
        items.append(ImportItem(KIND_CHARACTER, f"{KIND_CHARACTER}:{index}", character.name, character.backend_id,
                                clash=character.name.casefold() in names_here))
    with zipfile.ZipFile(info.path) as zf:
        for name in sorted(info.fx):
            path = _inside(fx_dir, name, ".json")
            item = ImportItem(KIND_FX, f"{KIND_FX}:{name}", name)
            if path and os.path.isfile(path):
                item.same = _fx_values_on_disk(path) == info.fx[name]
                item.clash = not item.same
            items.append(item)
        for name in sorted(info.irs):
            path = _inside(os.path.join(fx_dir, "ir"), name, ".wav")
            item = ImportItem(KIND_IR, f"{KIND_IR}:{name}", name)
            if path and os.path.isfile(path):
                item.same = _zip_matches_file(zf, f"fx/ir/{name}.wav", path)
                item.clash = not item.same
            items.append(item)
        for voice in info.voices:
            item = ImportItem(KIND_VOICE, f"{KIND_VOICE}:{voice.engine_id}:{voice.name}", voice.name, voice.engine_id)
            store = store_for(voice.engine_id)
            plan = _voice_plan(voice, store)
            if isinstance(plan, str):
                item.unavailable = plan
            else:
                states = []
                for ext, entry_name in plan:
                    path = _inside(store.global_dir, voice.name, ext)
                    if path is None:
                        item.unavailable = "its name can't be used as a file name"
                        break
                    if not os.path.isfile(path):
                        states.append("new")
                    else:
                        states.append("same" if _zip_matches_file(zf, entry_name, path) else "differs")
                else:
                    item.same = all(state == "same" for state in states)
                    item.clash = not item.same and "differs" in states
            items.append(item)
    return items


def _voice_plan(voice: SharedVoice, store):
    """`[(ext, entry name), ...]` for the files of `voice` its engine's store
    takes, or a string saying why none can be imported."""
    if store is None:
        return f"{voice.engine_id} isn't installed here"
    if hasattr(store, "extension"):  # EmbeddingStore: one file, the engine's extension
        entry_name = voice.files.get(store.extension)
        return [(store.extension, entry_name)] if entry_name else "it has no file for this engine"
    if hasattr(store, "requires_transcript"):  # ReferenceStore: a wav, and its transcript
        if ".wav" not in voice.files:
            return "it has no wav"
        plan = [(".wav", voice.files[".wav"])]
        if ".txt" in voice.files:
            plan.append((".txt", voice.files[".txt"]))
        elif store.requires_transcript:
            return "it has no transcript"
        return plan
    return f"{voice.engine_id} keeps no voice files"


def _write_atomic(path: str, writer) -> None:
    tmp = path + ".tmp"
    try:
        with open(tmp, "wb") as f:
            writer(f)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _extract(zf: zipfile.ZipFile, entry_name: str, path: str, cap: int) -> None:
    """Copies the entry to `path` (a temp file and `os.replace`), counting the
    bytes read so an entry that inflates past its limit stops."""
    def write(out):
        written = 0
        with zf.open(entry_name) as src:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    return
                written += len(chunk)
                if written > cap:
                    raise ShareError(f"{entry_name!r} is bigger than it says.")
                out.write(chunk)

    _write_atomic(path, write)


def _ir_is_usable(path: str) -> bool:
    """Whether `path` reads as audio no longer than the FX chain would load."""
    try:
        import soundfile as sf

        from kokoro_gui.engine.audio_fx import MAX_IR_SECONDS

        info = sf.info(path)
        return info.samplerate > 0 and info.frames / info.samplerate <= MAX_IR_SECONDS
    except Exception:  # noqa: BLE001 - soundfile raises several types for a bad file
        return False


def import_share(info: ShareInfo, choices: dict, *, library, fx_dir: str | None = None,
                 store_for=None) -> ImportResult:
    """Writes the items `choices` names into this machine's stores.

    `choices` maps an `ImportItem.key` to `KEEP`, `REPLACE` or `SKIP`; an item
    missing from it is skipped. A taken name under `KEEP` gets a number
    ("Hall (2)"), and everything that names it follows: a character's FX and
    voice, an FX's impulse response. `REPLACE` overwrites what is there. An
    item that is already here and identical is left alone. The share file is
    read again here, so a file changed since `inspect_share` is checked afresh.

    Characters go into `library` (a `CharacterLibrary`), FX into `fx_dir`
    (`presets/fx`), impulse responses into `fx_dir/ir`, voice files into each
    engine's global voice store."""
    fx_dir = fx_dir or os.path.join("presets", "fx")
    ir_dir = os.path.join(fx_dir, "ir")
    store_for = store_for or default_store_for
    fresh = inspect_share(info.path)
    items = {item.key: item for item in plan_import(fresh, library=library, fx_dir=fx_dir, store_for=store_for)}
    result = ImportResult(warnings=list(fresh.warnings))
    ir_map: dict = {}
    fx_map: dict = {}
    voice_map: dict = {}

    def action_for(key: str) -> str:
        item = items.get(key)
        if item is None or item.unavailable:
            if item is not None and item.unavailable:
                result.skipped.append(f"{item.name}: {item.unavailable}")
            return SKIP
        if item.same:
            return SKIP
        choice = choices.get(key, SKIP)
        if choice == REPLACE and not item.clash:
            return KEEP
        return choice if choice in (KEEP, REPLACE) else SKIP

    with zipfile.ZipFile(fresh.path) as zf:
        for name in sorted(fresh.irs):
            key = f"{KIND_IR}:{name}"
            if items[key].same:
                continue
            action = action_for(key)
            if action == SKIP:
                continue
            stored = name if action == REPLACE else _free_name(name, lambda n: _taken(ir_dir, n, ".wav"))
            path = _inside(ir_dir, stored, ".wav")
            if path is None:
                result.skipped.append(f"{name}: its name can't be used as a file name")
                continue
            try:
                os.makedirs(ir_dir, exist_ok=True)
                _extract(zf, f"fx/ir/{name}.wav", path, MAX_IR_BYTES)
            except (ShareError, OSError) as e:
                result.warnings.append(f"Impulse response {name!r} wasn't imported ({e}).")
                continue
            if not _ir_is_usable(path):
                os.remove(path)
                result.warnings.append(f"Impulse response {name!r} isn't audio the FX chain can load; left out.")
                continue
            if stored != name:
                ir_map[name] = stored
                result.renamed[key] = stored
            result.imported.append(f"Impulse response {stored}")
        for name in sorted(fresh.fx):
            key = f"{KIND_FX}:{name}"
            if items[key].same:
                continue
            action = action_for(key)
            if action == SKIP:
                continue
            values = copy.deepcopy(fresh.fx[name])
            if values.get("convolution_ir") in ir_map:
                values["convolution_ir"] = ir_map[values["convolution_ir"]]
            stored = name if action == REPLACE else _free_name(name, lambda n: _taken(fx_dir, n, ".json"))
            path = _inside(fx_dir, stored, ".json")
            if path is None:
                result.skipped.append(f"{name}: its name can't be used as a file name")
                continue
            try:
                os.makedirs(fx_dir, exist_ok=True)
                _write_atomic(path, lambda f, v=values: f.write(json.dumps(v, indent=4).encode("utf-8")))
            except OSError as e:
                result.warnings.append(f"FX {name!r} wasn't imported ({e}).")
                continue
            if stored != name:
                fx_map[name] = stored
                result.renamed[key] = stored
            result.imported.append(f"FX {stored}")
        for voice in fresh.voices:
            key = f"{KIND_VOICE}:{voice.engine_id}:{voice.name}"
            if items[key].same:
                continue
            action = action_for(key)
            if action == SKIP:
                continue
            store = store_for(voice.engine_id)
            plan = _voice_plan(voice, store)
            main_ext = plan[0][0]
            stored = voice.name if action == REPLACE else _free_name(
                voice.name, lambda n: _taken(store.global_dir, n, main_ext))
            try:
                ensure_private_dir(store.global_dir, fallback=False)
                for ext, entry_name in plan:
                    path = _inside(store.global_dir, stored, ext)
                    if path is None:
                        raise ShareError("its name can't be used as a file name")
                    _extract(zf, entry_name, path, _VOICE_EXTENSIONS[ext])
            except (ShareError, OSError) as e:
                result.warnings.append(f"Voice {voice.name!r} wasn't imported ({e}).")
            else:
                if stored != voice.name:
                    voice_map[(voice.engine_id, voice.name)] = stored
                    result.renamed[key] = stored
                result.imported.append(f"Voice {stored} ({voice.engine_id})")

    existing = {c.name.casefold(): c for c in library.list()}
    for index, character in enumerate(fresh.characters):
        key = f"{KIND_CHARACTER}:{index}"
        action = action_for(key)
        if action == SKIP:
            continue
        new = copy.deepcopy(character)
        engine_id = new.backend_id or DEFAULT_ENGINE_ID
        preset = new.preset_data
        if preset.get("fx_preset") in fx_map:
            preset["fx_preset"] = fx_map[preset["fx_preset"]]
        if preset.get("voice") is not None:
            preset["voice"] = voice_map.get((engine_id, preset["voice"]), preset["voice"])
        new.variants = {k: voice_map.get((engine_id, v), v) for k, v in new.variants.items()}
        if action == REPLACE and new.name.casefold() in existing:
            new.library_id = existing[new.name.casefold()].library_id
        else:
            new.name = _free_name(new.name, lambda n: n.casefold() in existing)
        new.library_id = library.save(new)
        existing[new.name.casefold()] = new
        if new.name != character.name:
            result.renamed[key] = new.name
        result.imported.append(f"Character {new.name}")
    return result
