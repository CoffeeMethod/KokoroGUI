"""Save to library (plan 41): keep what a project carried in.

A bundle brings its own voices (`engines/<id>/voices|refs/`) and FX (`fx/`,
`fx/ir/`). The resolvers read them first and the listings show them first
(grill TB3), but nothing copies them out, so opening the next project loses
them. This module plans and does that copy into the global stores.

Nothing here writes into a project or touches the bundle code. It reads the
open project's working copy and writes to the stores `share.py` imports into
(`presets/fx`, `presets/fx/ir`, each engine's voice store), with the same
clash choices (Keep both, Replace, Skip), the same numbered names ("Hall (2)")
and the same `clean_stem` names. Names that come from a project are untrusted:
every path goes through `share._inside`, which wants a clean stem and a real
path inside its folder. Identical means identical bytes for voice and impulse
response files, and equal filtered values for an FX.

Qt-free like the rest of `kokoro_gui/daw/`.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field

from kokoro_gui.daw import share
from kokoro_gui.daw.share import KEEP, REPLACE, SKIP
from kokoro_gui.engine.paths import ensure_private_dir
from kokoro_gui.engine.presets import filter_fx_preset_values
from kokoro_gui.engines.registry import DEFAULT_ENGINE_ID
from kokoro_gui.i18n import tr

# What the library holds for a project-local item: nothing of that name
# (ABSENT), the same thing (IDENTICAL), or something else (DIFFERENT).
# NOT_LOCAL is "the project doesn't carry it", so there is nothing to save.
ABSENT, IDENTICAL, DIFFERENT, NOT_LOCAL = "absent", "identical", "different", "not_local"

_PLACEHOLDER_FX = share._PLACEHOLDER_FX


@dataclass
class LibraryItem:
    """One project-local thing. The fields `clash`, `same` and `unavailable`
    are what `qt/share_dialogs.ImportShareDialog` reads off an `ImportItem`, so
    its table shows these rows too."""
    kind: str  # share.KIND_VOICE, KIND_FX or KIND_IR
    name: str
    label: str  # Mix, Reference, FX, Impulse response
    engine_id: str = ""
    status: str = NOT_LOCAL
    files: list = field(default_factory=list)  # voice and IR: [(ext, absolute source path)]
    values: dict | None = None  # FX: the filtered values read from the project
    store: object = None  # voice: its engine's voice store

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.engine_id}:{self.name}" if self.engine_id else f"{self.kind}:{self.name}"

    @property
    def detail(self) -> str:
        return self.engine_id

    @property
    def clash(self) -> bool:
        return self.status == DIFFERENT

    @property
    def same(self) -> bool:
        return self.status == IDENTICAL

    unavailable = ""


@dataclass
class CopyResult:
    saved: list = field(default_factory=list)  # one line per item written
    renamed: dict = field(default_factory=dict)  # key -> the name it was stored under
    skipped: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def _sha256(path: str) -> str | None:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    return h.hexdigest()
                h.update(chunk)
    except OSError:
        return None


def _same_files(a: str, b: str) -> bool:
    try:
        if os.path.getsize(a) != os.path.getsize(b):
            return False
    except OSError:
        return False
    first = _sha256(a)
    return first is not None and first == _sha256(b)


def _state_of(pairs) -> str:
    """The status of copying `[(source, destination), ...]`: any difference
    wins, then a missing file, else the same."""
    states = []
    for source, destination in pairs:
        if not os.path.isfile(destination):
            states.append(ABSENT)
        else:
            states.append(IDENTICAL if _same_files(source, destination) else DIFFERENT)
    if DIFFERENT in states:
        return DIFFERENT
    return ABSENT if ABSENT in states else IDENTICAL


# -- planning ---------------------------------------------------------------------------


def voice_item(store, engine_id: str, name, project_dir: str | None) -> LibraryItem:
    """The project's copy of voice `name` (`engines/<id>/voices/<name>.pt`, or the
    `.wav` and `.txt` of a Reference) against `store`'s global folder. NOT_LOCAL
    for a built-in voice, a name the project has no file for, or no store."""
    item = LibraryItem(share.KIND_VOICE, name if isinstance(name, str) else "", tr("Voice"), engine_id, store=store)
    if store is None or not project_dir or share.clean_stem(name) is None:
        return item
    local_dir = os.path.join(project_dir, *store.project_subdir.split("/"))
    if hasattr(store, "extension"):  # an embedding store: one file
        exts, wanted, item.label = [store.extension], [store.extension], tr("Mix")
    elif hasattr(store, "requires_transcript"):  # a reference store: a wav, and its transcript
        exts, wanted, item.label = [".wav", ".txt"], [".wav"], tr("Reference")
    else:
        return item
    for ext in exts:
        path = share._inside(local_dir, name, ext)
        if path and os.path.isfile(path):
            item.files.append((ext, path))
    if [ext for ext, _path in item.files if ext in wanted] != wanted:
        item.files = []
        return item
    pairs = []
    for ext, source in item.files:
        destination = share._inside(store.global_dir, name, ext)
        if destination is None:
            item.files = []
            return item
        pairs.append((source, destination))
    item.status = _state_of(pairs)
    return item


def _fx_file(name, project_dir: str | None, fx_dir: str):
    """`(path, is_project_local)` of FX `name`, the project's `fx/` first."""
    for directory, local in ((os.path.join(project_dir, "fx") if project_dir else None, True), (fx_dir, False)):
        if directory is None:
            continue
        path = share._inside(directory, name, ".json")
        if path and os.path.isfile(path):
            return path, local
    return None, False


def fx_items(name, project_dir: str | None, fx_dir: str) -> list:
    """The project-local items behind FX `name`: the FX itself when it is in the
    project's `fx/`, and the impulse response it names when that is in the
    project's `fx/ir/`. Empty for an FX that lives only in the library."""
    if not isinstance(name, str) or not name or name == _PLACEHOLDER_FX:
        return []
    path, local = _fx_file(name, project_dir, fx_dir)
    if path is None:
        return []
    values = share._fx_values_on_disk(path)
    if values is None:
        return []
    items = []
    if local:
        item = LibraryItem(share.KIND_FX, name, tr("FX"), values=values)
        target = share._inside(fx_dir, name, ".json")
        if target is None:
            return []
        item.status = ABSENT if not os.path.isfile(target) else (
            IDENTICAL if share._fx_values_on_disk(target) == values else DIFFERENT)
        items.append(item)
    ir = values.get("convolution_ir")
    if isinstance(ir, str) and ir and project_dir and share.clean_stem(ir):
        source = share._inside(os.path.join(project_dir, "fx", "ir"), ir, ".wav")
        target = share._inside(os.path.join(fx_dir, "ir"), ir, ".wav")
        if source and target and os.path.isfile(source):
            items.append(LibraryItem(share.KIND_IR, ir, tr("Impulse response"), files=[(".wav", source)],
                                     status=_state_of([(source, target)])))
    return items


def character_items(character, project_dir: str | None, fx_dir: str, store_for=None) -> list:
    """Everything `character` uses that only the project carries: its voice,
    its variants' voices, its FX and that FX's impulse response."""
    store_for = store_for or share.default_store_for
    engine_id = character.backend_id or DEFAULT_ENGINE_ID
    store = store_for(engine_id) if share._ENGINE_ID.match(engine_id) else None
    items = []
    for name in sorted(share.voice_names_of(character)):
        item = voice_item(store, engine_id, name, project_dir)
        if item.status != NOT_LOCAL:
            items.append(item)
    items.extend(fx_items((character.preset_data or {}).get("fx_preset"), project_dir, fx_dir))
    return items


# -- saving -----------------------------------------------------------------------------


def _copy_into(source: str, destination: str) -> None:
    def write(out):
        with open(source, "rb") as src:
            shutil.copyfileobj(src, out, 1 << 20)

    share._write_atomic(destination, write)


def save_to_library(items, choices: dict | None = None, *, fx_dir: str) -> CopyResult:
    """Copies the items into the global stores. `choices` maps `LibraryItem.key`
    to `KEEP`, `REPLACE` or `SKIP`; what the library lacks is copied whatever
    `choices` says (an item missing from it counts as `KEEP`), what it already
    holds identically is left alone, and a different one is skipped unless
    `choices` says otherwise. `KEEP` on a taken name stores a numbered copy
    ("Hall (2)"), and an FX that names a renamed impulse response follows it.
    The project is only read."""
    choices = choices or {}
    result = CopyResult()
    ir_dir = os.path.join(fx_dir, "ir")
    ir_map: dict = {}

    def action_for(item) -> str:
        if item.status in (IDENTICAL, NOT_LOCAL):
            return SKIP
        if item.status == ABSENT:
            return KEEP
        return choices.get(item.key, SKIP) if choices.get(item.key, SKIP) in (KEEP, REPLACE) else SKIP

    def note_skip(item) -> None:
        if item.status == IDENTICAL:
            result.skipped.append(tr("{label} {name}: the library already has it", label=item.label, name=item.name))
        elif item.status == DIFFERENT:
            result.skipped.append(tr("{label} {name}: left out", label=item.label, name=item.name))

    ordered = sorted(items, key=lambda i: {share.KIND_IR: 0, share.KIND_FX: 1}.get(i.kind, 2))
    for item in ordered:
        action = action_for(item)
        if action == SKIP:
            note_skip(item)
            continue
        overwrite = action == REPLACE or item.status == ABSENT
        try:
            if item.kind == share.KIND_IR:
                stored = item.name if overwrite else share._free_name(
                    item.name, lambda n: share._taken(ir_dir, n, ".wav"))
                path = share._inside(ir_dir, stored, ".wav")
                if path is None:
                    raise OSError(tr("its name can't be used as a file name"))
                os.makedirs(ir_dir, exist_ok=True)
                _copy_into(item.files[0][1], path)
                if stored != item.name:
                    ir_map[item.name] = stored
                line = tr("Impulse response {stored}", stored=stored)
            elif item.kind == share.KIND_FX:
                values = json.loads(json.dumps(item.values))
                if values.get("convolution_ir") in ir_map:
                    values["convolution_ir"] = ir_map[values["convolution_ir"]]
                values = filter_fx_preset_values(values)
                stored = item.name if overwrite else share._free_name(
                    item.name, lambda n: share._taken(fx_dir, n, ".json"))
                path = share._inside(fx_dir, stored, ".json")
                if path is None:
                    raise OSError(tr("its name can't be used as a file name"))
                os.makedirs(fx_dir, exist_ok=True)
                share._write_atomic(path, lambda f, v=values: f.write(json.dumps(v, indent=4).encode("utf-8")))
                line = tr("FX {stored}", stored=stored)
            else:
                store = item.store
                main_ext = item.files[0][0]
                stored = item.name if overwrite else share._free_name(
                    item.name, lambda n: share._taken(store.global_dir, n, main_ext))
                ensure_private_dir(store.global_dir, fallback=False)
                for ext, source in item.files:
                    path = share._inside(store.global_dir, stored, ext)
                    if path is None:
                        raise OSError(tr("its name can't be used as a file name"))
                    _copy_into(source, path)
                line = f"{item.label} {stored} ({item.engine_id})"
        except (OSError, ValueError) as e:
            result.warnings.append(tr("{label} {name!r} wasn't saved ({error}).", label=item.label, name=item.name,
                                      error=e))
            continue
        if stored != item.name:
            result.renamed[item.key] = stored
        result.saved.append(line)
    return result
