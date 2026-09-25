"""Project files for the File menu. Pure functions, no Qt.

The project format is the `.tbaw` bundle (Claude/old/PLAN_tbaw_bundle.md, grill
TB1-TB15): one zip holding `manifest.json`, `document.json`
(`kokoro_gui.daw.serialization`'s shape), `project.json` (the
`project_settings` block: export defaults, workspace override, bundle
options), every generated segment under `audio/generated/` named by its
segment key, and every named asset a character or clip points at (`fx/`,
`engines/<id>/...`), and, when `include_video` is on, the reference
video under `video/` (phase 5, TB16). A `.json` project (the 4.0-preview
format, the document shape plus a top-level `"project_settings"`) still
opens and is migrated to `.tbaw` on open (`migrate_json_project`).

The live project is a directory, `cache/projects/<project_id>/` (the
"project dir"), extracted from the zip on Open and written by autosave
(`document.json`, `project.json`) and by clip generation
(`audio/generated/`). Save rewrites the whole zip from it. While a project
is open the app holds an OS lock on `<project dir>/lock`, and
`session.json` next to it records what the dir belongs to
(`source_path`, the zip's size and mtime at extraction), the digest of the
last saved document, whether the dir is ahead of the zip (`dirty`), and an
`asset_index` so Save can skip re-hashing an unchanged asset. Recovery after
a crash keys on `project_id`, not on the path.

`settings["last_project"]` is what launch reopens (WF2, revised: the
welcome dialog offers it as Resume); `settings["recent_projects"]` is the
File > Recent list and the welcome dialog's rows, most recent first, at
most `MAX_RECENT` entries. The app (kokoro_gui/qt/app.py) owns the
sequencing and the threads: the functions here are the steps.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import secrets
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass, field

import kokoro_engine
from kokoro_gui import APP_VERSION
from kokoro_gui.daw import serialization
from kokoro_gui.daw.models import Document
from kokoro_gui.engine.caching import RESERVED_SUFFIX, compute_cache_key, effective_speed

MAX_RECENT = 10
PROJECT_FILTER = "KokoroGUI project (*.tbaw *.json)"
VIDEO_FILTER = "Video (*.mp4 *.mov *.mkv *.webm *.avi *.m4v);;All files (*)"
AUDIO_FILTER = "Audio (*.wav *.flac *.ogg *.mp3 *.aif *.aiff);;All files (*)"
DEFAULT_EXTENSION = ".tbaw"

FORMAT = "tbaw"
SUPPORTED_VERSION = 1
# Content features this reader implements; a bundle whose `requires` names
# one that isn't here is refused by name (section 8 of the plan).
SUPPORTED_FEATURES: frozenset = frozenset({"takes", "nested", "imported"})

MANIFEST = "manifest.json"
DOCUMENT = "document.json"
PROJECT_JSON = "project.json"
SESSION = "session.json"
LOCK = "lock"
AUDIO_GENERATED = "audio/generated"
AUDIO_IMPORTED = "audio/imported"
FX_DIR = "fx"
ENGINES_DIR = "engines"
# Embedded subprojects (phase 4): `projects/<child project_id>.tbaw`, each a
# complete bundle, stored uncompressed.
PROJECTS_DIR = "projects"
# The reference video (phase 5, TB16), when the `include_video` bundle
# option is on: `video/<sha256[:16]>.<ext>`, stored uncompressed.
VIDEO_DIR = "video"
# A subproject's rendered mix, in its own project dir (phase 4, NP2):
# `mixdown.<fmt>` plus `mixdown.json` recording the document digest it was
# rendered from, its length and rate. Derived data: never bundled, always
# rebuildable.
MIXDOWN = "mixdown"
MIXDOWN_JSON = "mixdown.json"
# A child project dir's `session.json` names its source as
# `<parent source>#<child id>`, so `choose_project_dir` and
# `sweep_orphan_dirs` can tell an embedded child from a root.
CHILD_SOURCE_SEP = "#"

# Entry prefixes this version owns and rewrites on every Save. Anything else
# in a bundle (a directory a newer KokoroGUI or a fourth engine added) is
# copied through byte for byte so a file survives a round trip.
_OWNED_FILES = {MANIFEST, DOCUMENT, PROJECT_JSON}
_OWNED_DIRS = (FX_DIR + "/", AUDIO_GENERATED + "/", AUDIO_IMPORTED + "/", PROJECTS_DIR + "/", VIDEO_DIR + "/")
# Entries extracted with the audio on the worker thread, not before the
# first paint: the audio, embedded children (each carries its own) and the
# reference video.
_HEAVY_PREFIXES = ("audio/", PROJECTS_DIR + "/", VIDEO_DIR + "/")
# The project dir's own bookkeeping. A bundle carrying one of these names is
# never extracted over it: `lock` is held open while Open runs, and
# `session.json` is what the sweep and the recover prompt trust.
_DIR_PRIVATE = {SESSION, LOCK, SESSION + ".tmp", DOCUMENT + ".tmp", PROJECT_JSON + ".tmp"}

DEFAULT_BUNDLE_OPTIONS = {"include_generated_audio": True, "include_imported_audio": True, "include_video": False,
                          "audio_format": "wav"}

# `torch.load` defaults to `weights_only=True` from 2.6, which is what makes
# a `.pt` from someone else's bundle safe to load. Checked once at import.
_TORCH_VERSION_OK: bool | None = None


class ProjectError(Exception):
    """A bundle this version can't or won't open; the message is for the user."""


class ProjectLockedError(ProjectError):
    """Another KokoroGUI holds the project dir's lock."""


@dataclass
class LoadedProject:
    document: Document
    project_settings: dict = field(default_factory=dict)
    project_dir: str | None = None
    project_id: str | None = None
    manifest: dict = field(default_factory=dict)
    # One-line notices for the status bar (a missing audio file, a version
    # difference), not errors.
    notices: list = field(default_factory=list)
    # The reference video to play (`video_source`): the settings path when
    # that file exists, else the bundled copy in the project dir, else None.
    video_path: str | None = None


@dataclass
class BundleInfo:
    """What `inspect_bundle` learns from a zip's central directory without
    extracting a byte. `audio_bytes` counts everything extracted on the
    worker thread: `audio/` and embedded `projects/`."""
    path: str
    manifest: dict
    project_id: str
    entries: list  # ZipInfo, validated
    audio_bytes: int
    zip_size: int
    zip_mtime: float


@dataclass
class SaveResult:
    zip_size: int
    zip_mtime: float
    asset_index: dict
    saved_digest: str


# --- paths and ids ---------------------------------------------------------------


def format_for_path(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".tbaw":
        return "tbaw"
    return "json"


def projects_root() -> str:
    """`cache/projects/` under `kokoro_engine.CACHE_DIR`, read at call time
    so the `isolated_dirs` fixture redirects it. Absolute: every
    `audio_path` is built on it and Save tells project-dir files from
    outside ones by prefix."""
    return os.path.abspath(os.path.join(kokoro_engine.CACHE_DIR, "projects"))


def new_project_id() -> str:
    return secrets.token_hex(8)


def project_title(path: str | None) -> str:
    if not path:
        return "Untitled"
    return os.path.splitext(os.path.basename(path))[0] or "Untitled"


def display_title(project_settings: dict | None, path: str | None, fallback: str = "Untitled") -> str:
    """A project's name as the parent, the breadcrumb and a placeholder run
    show it: `project_settings["title"]` (phase 4; `project.json`, not the
    document, so 4.0 keeps it), else the file stem, else `fallback`."""
    title = (project_settings or {}).get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    if path:
        return project_title(path)
    return fallback


def bundle_path_for(path: str) -> str:
    """The `.tbaw` name for any project path: `.json` becomes `.tbaw`, a
    bare name gets the extension."""
    root, ext = os.path.splitext(path)
    if ext.lower() in (".tbaw", ".json"):
        return root + DEFAULT_EXTENSION
    return path + DEFAULT_EXTENSION


def torch_weights_only_available() -> bool:
    global _TORCH_VERSION_OK
    if _TORCH_VERSION_OK is None:
        try:
            import torch

            major, minor = (int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
            _TORCH_VERSION_OK = (major, minor) >= (2, 6)
        except Exception:
            _TORCH_VERSION_OK = False
    return _TORCH_VERSION_OK


# --- recent list -----------------------------------------------------------------


def remember_recent(settings: dict, path: str) -> list:
    """Moves `path` to the front of `settings["recent_projects"]`, dropping
    duplicates and trimming to `MAX_RECENT`. Returns the new list."""
    path = os.path.abspath(path)
    recent = [p for p in settings.get("recent_projects", []) if isinstance(p, str) and os.path.abspath(p) != path]
    recent.insert(0, path)
    settings["recent_projects"] = recent[:MAX_RECENT]
    settings["last_project"] = path
    return settings["recent_projects"]


def forget_recent(settings: dict, path: str) -> None:
    path = os.path.abspath(path)
    settings["recent_projects"] = [p for p in settings.get("recent_projects", []) if os.path.abspath(p) != path]


def clear_recent(settings: dict) -> None:
    """Empties the list. `last_project` stays: the open project is still the
    one launch resumes, it just isn't listed any more."""
    settings["recent_projects"] = []


# --- .json (legacy) --------------------------------------------------------------


def load_json_project(path: str) -> LoadedProject | None:
    """`None` when the file is missing or unreadable, same tolerance
    `serialization.load_document` has."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    document = serialization.document_from_dict(data)
    project_settings = data.get("project_settings", {})
    return LoadedProject(document=document,
                         project_settings=dict(project_settings) if isinstance(project_settings, dict) else {})


def save_json_project(document: Document, path: str, project_settings: dict | None = None) -> None:
    """The 4.0-preview format, kept for the migration tests and for anyone who
    wants a plain-JSON export; the app never writes it any more."""
    data = serialization.document_to_dict(document)
    data["project_settings"] = dict(project_settings or {})
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_project(path: str) -> LoadedProject | None:
    """Reads a project's document and settings without a project dir: the
    `.json` format directly, a `.tbaw` straight out of the zip (audio paths
    left bundle-relative, so segments read as missing). The app's Open goes
    through `inspect_bundle`/`extract_*`/`finish_open` instead; this is for
    callers that only need the document (tests, tooling)."""
    if not path or not os.path.exists(path):
        return None
    if format_for_path(path) != "tbaw":
        return load_json_project(path)
    try:
        with zipfile.ZipFile(path) as zf:
            manifest = read_manifest_from(zf)
            document = serialization.document_from_dict(json.loads(zf.read(DOCUMENT).decode("utf-8")))
            project_settings = {}
            if PROJECT_JSON in zf.namelist():
                project_settings = json.loads(zf.read(PROJECT_JSON).decode("utf-8"))
    except (OSError, zipfile.BadZipFile, KeyError, json.JSONDecodeError, ProjectError):
        return None
    return LoadedProject(document=document, project_settings=project_settings,
                         project_id=manifest.get("project_id"), manifest=manifest)


# --- manifest ---------------------------------------------------------------------


def read_manifest_from(zf: zipfile.ZipFile) -> dict:
    try:
        manifest = json.loads(zf.read(MANIFEST).decode("utf-8"))
    except KeyError:
        raise ProjectError("Not a KokoroGUI project: no manifest.json in the bundle.")
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ProjectError("The bundle's manifest.json is not valid JSON.")
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise ProjectError("Not a KokoroGUI project: the manifest's format isn't \"tbaw\".")
    version = manifest.get("version", 0)
    if not isinstance(version, int) or version > SUPPORTED_VERSION:
        raise ProjectError(f"This project was made with a newer KokoroGUI (bundle version {version}, "
                           f"this build reads up to {SUPPORTED_VERSION}).")
    requires = manifest.get("requires", []) or []
    unknown = [str(r) for r in requires if str(r) not in SUPPORTED_FEATURES]
    if unknown:
        raise ProjectError("This project needs a feature this KokoroGUI doesn't have: " + ", ".join(unknown))
    if not manifest.get("project_id"):
        manifest["project_id"] = new_project_id()
    return manifest


def _entry_name_is_safe(name: str) -> bool:
    normalized = name.replace("\\", "/")
    if not normalized or normalized.endswith("/") and normalized.count("/") == 0:
        return True
    if os.path.isabs(normalized) or normalized.startswith("/"):
        return False
    if len(normalized) >= 2 and normalized[1] == ":":
        return False  # drive-relative on Windows (C:foo)
    parts = normalized.split("/")
    if any(part == ".." for part in parts):
        return False
    return True


def _is_symlink_entry(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0o170000
    return mode == 0o120000


def validate_entries(zf: zipfile.ZipFile, project_dir: str) -> list:
    """Every entry, or `ProjectError` for one that would write outside the
    project dir (absolute on either OS, drive-relative, `..`, a symlink)."""
    root = os.path.realpath(project_dir)
    entries = []
    for info in zf.infolist():
        name = info.filename
        if not _entry_name_is_safe(name):
            raise ProjectError(f"Refusing to open: the bundle has an unsafe entry name ({name!r}).")
        if _is_symlink_entry(info):
            raise ProjectError(f"Refusing to open: the bundle contains a symlink ({name!r}).")
        target = os.path.realpath(os.path.join(root, *name.replace("\\", "/").split("/")))
        if target != root and not target.startswith(root + os.sep):
            raise ProjectError(f"Refusing to open: an entry resolves outside the project dir ({name!r}).")
        entries.append(info)
    return entries


def free_space(path: str) -> int:
    probe = path
    while probe and not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    try:
        return shutil.disk_usage(probe or ".").free
    except OSError:
        return sys.maxsize


def check_free_space(path: str, needed: int, what: str) -> None:
    """`ProjectError` before a byte is written when the volume holding
    `path` can't take `needed` bytes (plus a small margin)."""
    margin = 32 * 1024 * 1024
    available = free_space(path)
    if available < needed + margin:
        raise ProjectError(f"Not enough free space to {what}: needs about {needed // (1024 * 1024)} MB, "
                           f"{available // (1024 * 1024)} MB free on {os.path.dirname(os.path.abspath(path)) or path}.")


def inspect_bundle(path: str) -> BundleInfo:
    """Step 1 and 3 of Open: the manifest, validated entries and the audio
    byte count, all from the central directory."""
    if not path or not os.path.isfile(path):
        raise ProjectError(f"{path} doesn't exist.")
    try:
        zf = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile):
        raise ProjectError(f"{os.path.basename(path)} isn't a readable .tbaw bundle.")
    with zf:
        manifest = read_manifest_from(zf)
        project_id = manifest["project_id"]
        entries = validate_entries(zf, os.path.join(projects_root(), project_id))
        if DOCUMENT not in zf.namelist():
            raise ProjectError("The bundle has no document.json.")
        if any(e.filename.endswith(".pt") for e in entries) and not torch_weights_only_available():
            raise ProjectError("This bundle carries a voice mix (.pt) and this machine's torch is older than 2.6, "
                               "which can't load it safely. Upgrade torch to open it.")
        audio_bytes = sum(e.file_size for e in entries if e.filename.replace("\\", "/").startswith(_HEAVY_PREFIXES))
    stat = os.stat(path)
    return BundleInfo(path=os.path.abspath(path), manifest=manifest, project_id=project_id, entries=entries,
                      audio_bytes=audio_bytes, zip_size=stat.st_size, zip_mtime=stat.st_mtime)


# --- project dir, session, lock ---------------------------------------------------


def read_session(project_dir: str) -> dict | None:
    path = os.path.join(project_dir, SESSION)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def write_session(project_dir: str, data: dict) -> None:
    """The GUI thread is the one writer; a background Save hands its
    numbers back for the GUI side to record."""
    os.makedirs(project_dir, exist_ok=True)
    tmp = os.path.join(project_dir, SESSION + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, os.path.join(project_dir, SESSION))


def choose_project_dir(project_id: str, source_path: str | None) -> str:
    """`cache/projects/<project_id>/`, or `<project_id>-2`, `-3`, ... when
    that dir's session names a different `source_path` and is clean (a Save
    As left two files sharing one id). A dirty dir with another path is
    still this project's: the id matched, that's what recovery keys on."""
    root = projects_root()
    suffix = 1
    while True:
        candidate = os.path.join(root, project_id if suffix == 1 else f"{project_id}-{suffix}")
        session = read_session(candidate)
        if session is None:
            return candidate
        theirs = session.get("source_path")
        same_source = (not theirs and not source_path) or bool(
            theirs and source_path and os.path.abspath(theirs) == os.path.abspath(source_path))
        if same_source or session.get("dirty"):
            return candidate
        suffix += 1


class ProjectLock:
    """An OS advisory lock on `<project dir>/lock`, held open for as long as
    the project is. The OS releases it when the process dies, so there is
    no pid to check and no stale-lock heuristic; a second instance whose
    attempt fails is told the project is open elsewhere."""

    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        self.path = os.path.join(project_dir, LOCK)
        self._handle = None

    def acquire(self) -> "ProjectLock":
        os.makedirs(self.project_dir, exist_ok=True)
        handle = open(self.path, "a+b")
        try:
            if sys.platform == "win32":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            raise ProjectLockedError("This project is already open in another KokoroGUI window.")
        self._handle = handle
        return self

    def release(self) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        handle.close()

    @property
    def held(self) -> bool:
        return self._handle is not None


def is_locked(project_dir: str) -> bool:
    if not os.path.isfile(os.path.join(project_dir, LOCK)):
        return False
    try:
        ProjectLock(project_dir).acquire().release()
    except ProjectLockedError:
        return True
    except OSError:
        return False
    return False


def wipe_project_dir(project_dir: str) -> None:
    """Everything except `lock`, which the holder has open (on Windows
    `rmtree` over an open locked handle raises)."""
    if not os.path.isdir(project_dir):
        return
    for entry in os.listdir(project_dir):
        if entry == LOCK:
            continue
        full = os.path.join(project_dir, entry)
        if os.path.isdir(full) and not os.path.islink(full):
            shutil.rmtree(full, ignore_errors=True)
        else:
            try:
                os.remove(full)
            except OSError:
                pass


def delete_project_dir(project_dir: str) -> None:
    wipe_project_dir(project_dir)
    try:
        os.remove(os.path.join(project_dir, LOCK))
    except OSError:
        pass
    try:
        os.rmdir(project_dir)
    except OSError:
        pass


def create_project_dir(project_id: str | None = None) -> tuple:
    """New: a fresh dir with an empty `document.json`, so an Untitled
    project has somewhere to generate into before its first Save. Returns
    `(project_dir, project_id)`."""
    project_id = project_id or new_project_id()
    project_dir = os.path.join(projects_root(), project_id)
    os.makedirs(os.path.join(project_dir, AUDIO_GENERATED), exist_ok=True)
    return project_dir, project_id


def safe_child_id(child_id) -> str | None:
    """A child project id as a bare file stem, or None."""
    if not isinstance(child_id, str):
        return None
    stem = os.path.basename(child_id.strip())
    if not stem or stem.startswith(".") or CHILD_SOURCE_SEP in stem:
        return None
    return stem


def embedded_child_path(project_dir: str, child_id: str) -> str | None:
    """`<project_dir>/projects/<child_id>.tbaw`: where an embedded child's
    bundle sits in its parent's project dir."""
    stem = safe_child_id(child_id)
    if stem is None or not project_dir:
        return None
    return os.path.join(project_dir, PROJECTS_DIR, f"{stem}{DEFAULT_EXTENSION}")


def embedded_child_ids(document: Document) -> list:
    """The project ids of `document`'s embedded subprojects, in clip order."""
    out = []
    for clip in document.clips:
        child = clip.child if clip.source == "nested" else None
        if isinstance(child, dict) and child.get("kind") == "embedded":
            stem = safe_child_id(child.get("id"))
            if stem and stem not in out:
                out.append(stem)
    return out


def parent_source_of(source: str) -> str:
    """The root file of a `<parent>#<child id>[#<grandchild id>...]`
    source; `source` itself for a root's."""
    while CHILD_SOURCE_SEP in source:
        head, tail = source.rsplit(CHILD_SOURCE_SEP, 1)
        if safe_child_id(tail) != tail:
            break
        source = head
    return source


def child_dirs_of(source_path: str | None) -> list:
    """Every project dir under `cache/projects/` whose session names an
    embedded child (at any depth) of the project whose source is
    `source_path`."""
    if not source_path:
        return []
    root = projects_root()
    if not os.path.isdir(root):
        return []
    prefix = os.path.abspath(source_path) + CHILD_SOURCE_SEP
    out = []
    for name in sorted(os.listdir(root)):
        full = os.path.join(root, name)
        session = read_session(full) if os.path.isdir(full) else None
        source = (session or {}).get("source_path")
        if isinstance(source, str) and source.startswith(prefix):
            out.append(full)
    return out


def child_source_path(parent_source: str | None, child_id: str) -> str | None:
    """`session.json`'s `source_path` for an embedded child of a parent
    whose own source is `parent_source` (its `.tbaw`, or None while
    Untitled)."""
    if not parent_source:
        return None
    return f"{os.path.abspath(parent_source)}{CHILD_SOURCE_SEP}{child_id}"


def mixdown_file(project_dir: str, fmt: str = "wav") -> str:
    return os.path.join(project_dir, f"{MIXDOWN}.{fmt}")


def read_mixdown_info(project_dir: str | None) -> dict | None:
    """`mixdown.json` when it names a mixdown file that exists: `{"file",
    "digest", "duration_s", "sample_rate"}` with `file` absolute. None
    otherwise."""
    if not project_dir:
        return None
    path = os.path.join(project_dir, MIXDOWN_JSON)
    try:
        with open(path, "r", encoding="utf-8") as f:
            info = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(info, dict):
        return None
    name = os.path.basename(str(info.get("file") or ""))
    if not name.startswith(MIXDOWN + "."):
        return None
    full = os.path.join(project_dir, name)
    if not os.path.isfile(full):
        return None
    try:
        duration = float(info.get("duration_s") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    return {"file": full, "digest": info.get("digest"), "duration_s": duration,
            "sample_rate": info.get("sample_rate")}


def write_mixdown_info(project_dir: str, file_path: str, digest: str, duration_s: float, sample_rate: int) -> None:
    tmp = os.path.join(project_dir, MIXDOWN_JSON + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"file": os.path.basename(file_path), "digest": digest, "duration_s": round(float(duration_s), 6),
                   "sample_rate": int(sample_rate)}, f, indent=2)
    os.replace(tmp, os.path.join(project_dir, MIXDOWN_JSON))


def remove_mixdown(project_dir: str) -> None:
    info = read_mixdown_info(project_dir)
    for path in ((info or {}).get("file"), os.path.join(project_dir, MIXDOWN_JSON)):
        if path:
            try:
                os.remove(path)
            except OSError:
                pass


def project_digest(document: Document, project_settings: dict, project_dir: str) -> str:
    """The digest autosave would record for the project as it is now."""
    return document_digest(*serialize_for_dir(document, project_settings, project_dir))


def document_digest(document_bytes: bytes, project_bytes: bytes) -> str:
    return hashlib.sha256(document_bytes + b"\n" + project_bytes).hexdigest()


def serialize_for_dir(document: Document, project_settings: dict, project_dir: str) -> tuple:
    """`(document_bytes, project_bytes)` as autosave writes them: audio
    paths absolute inside the project dir stay absolute in the dir's copy
    (they're rewritten to bundle-relative only in the zip)."""
    data = serialization.document_to_dict(document)
    document_bytes = json.dumps(data, indent=2).encode("utf-8")
    project_bytes = json.dumps(dict(project_settings or {}), indent=2).encode("utf-8")
    return document_bytes, project_bytes


def autosave_to_dir(document: Document, project_settings: dict, project_dir: str) -> str:
    """Writes `document.json` and `project.json` into the project dir and
    returns the digest of what was written, for the app to compare with
    `session.json`'s `saved_digest`."""
    document_bytes, project_bytes = serialize_for_dir(document, project_settings, project_dir)
    os.makedirs(project_dir, exist_ok=True)
    for name, payload in ((DOCUMENT, document_bytes), (PROJECT_JSON, project_bytes)):
        tmp = os.path.join(project_dir, name + ".tmp")
        with open(tmp, "wb") as f:
            f.write(payload)
        os.replace(tmp, os.path.join(project_dir, name))
    return document_digest(document_bytes, project_bytes)


# --- open --------------------------------------------------------------------------


def _extract_entry(zf: zipfile.ZipFile, info: zipfile.ZipInfo, project_dir: str) -> None:
    name = info.filename.replace("\\", "/")
    target = os.path.join(project_dir, *name.split("/"))
    if name.endswith("/"):
        os.makedirs(target, exist_ok=True)
        return
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with zf.open(info) as src, open(target, "wb") as dst:
        shutil.copyfileobj(src, dst, 1024 * 1024)


def extract_small(info: BundleInfo, project_dir: str) -> None:
    """Step 4: everything but `audio/` and `projects/`. Small, needed
    before the first paint."""
    os.makedirs(project_dir, exist_ok=True)
    with zipfile.ZipFile(info.path) as zf:
        for entry in info.entries:
            name = entry.filename.replace("\\", "/")
            if name.startswith(_HEAVY_PREFIXES) or name in _DIR_PRIVATE:
                continue
            _extract_entry(zf, entry, project_dir)


def extract_audio(info: BundleInfo, project_dir: str, progress=None, cancelled=None) -> None:
    """Step 5: `audio/`, embedded `projects/` and `video/`, eagerly, meant
    for a worker thread. The video is skipped when the path in
    `project.json` is a file here (`bundled_video_needed`).
    `progress(done, total)` in bytes; `cancelled()` is polled between
    entries."""
    skip_video = not bundled_video_needed(info)
    total = max(1, heavy_bytes(info))
    done = 0
    with zipfile.ZipFile(info.path) as zf:
        for entry in info.entries:
            if cancelled is not None and cancelled():
                return
            if not entry.filename.replace("\\", "/").startswith(_HEAVY_PREFIXES):
                continue
            if skip_video and _is_video_entry(entry):
                continue
            _extract_entry(zf, entry, project_dir)
            done += entry.file_size
            if progress is not None:
                progress(done, total)


def _load_dir(project_dir: str) -> tuple:
    """`(document, project_settings, notices)` from a project dir's
    `document.json` and `project.json`, audio paths made absolute (a file
    that isn't inside the dir becomes `None`, so its clip reads as dirty)."""
    with open(os.path.join(project_dir, DOCUMENT), "r", encoding="utf-8") as f:
        data = json.load(f)
    project_settings = {}
    project_json = os.path.join(project_dir, PROJECT_JSON)
    if os.path.isfile(project_json):
        try:
            with open(project_json, "r", encoding="utf-8") as f:
                project_settings = json.load(f)
        except (OSError, json.JSONDecodeError):
            project_settings = {}
    if not isinstance(project_settings, dict):
        project_settings = {}

    notices = []
    missing = []
    project_root = os.path.realpath(project_dir)

    def to_absolute(rel):
        # A bundle names its audio relative to the bundle; the dir's own
        # autosave names it absolute. Either way the file has to sit inside
        # the project dir: `document.json` is untrusted input, and a path
        # pointing anywhere else would pull that file into the next Save.
        if os.path.isabs(rel):
            candidate = rel
        else:
            candidate = os.path.join(project_dir, *rel.replace("\\", "/").split("/"))
        real = os.path.realpath(candidate)
        if real.startswith(project_root + os.sep) and os.path.isfile(real):
            return os.path.abspath(candidate)
        missing.append(rel)
        return None

    serialization.rewrite_audio_paths(data, to_absolute)
    document = serialization.document_from_dict(data)
    if missing:
        notices.append(f"{len(missing)} audio file(s) missing from the bundle; those clips will regenerate.")
    return document, project_settings, notices


def load_project_dir(project_dir: str, project_id: str | None = None, manifest: dict | None = None) -> LoadedProject:
    """A project straight from its project dir, with no bundle and no
    session change: a subproject made this session, or one whose dir is
    already extracted (clean, or ahead of its bundle)."""
    document, project_settings, notices = _load_dir(project_dir)
    return LoadedProject(document=document, project_settings=project_settings, project_dir=project_dir,
                         project_id=project_id, manifest=dict(manifest or {}), notices=notices,
                         video_path=video_source(project_settings, None, project_dir, manifest))


def finish_open(info: BundleInfo, project_dir: str, engine_versions: dict | None = None,
                recovered: bool = False, source_path: str | None = None) -> LoadedProject:
    """Steps 6 and 7: the document from the project dir with paths made
    absolute (a missing file becomes `None`, so the clip reads as dirty),
    `project.json`, a fresh `session.json` unless the session was recovered
    (then it stays as it is, `dirty` included), and the TB9 notice when a
    manifest engine version differs from `engine_versions[id]`. An embedded
    child passes its `<parent>#<id>` `source_path`."""
    document, project_settings, notices = _load_dir(project_dir)

    for engine_id, block in (info.manifest.get("engines") or {}).items():
        if not isinstance(block, dict):
            continue
        made_with = block.get("version")
        installed = (engine_versions or {}).get(engine_id)
        if made_with and installed and made_with != installed:
            notices.append(f"generated with {engine_id} {made_with}, this machine has {installed}; "
                           f"regenerated clips will use {installed}")

    if not recovered:
        document_bytes, project_bytes = serialize_for_dir(document, project_settings, project_dir)
        previous = read_session(project_dir) or {}
        write_session(project_dir, {
            "source_path": source_path or info.path,
            "zip_size": info.zip_size,
            "zip_mtime": info.zip_mtime,
            "saved_digest": document_digest(document_bytes, project_bytes),
            "dirty": False,
            "asset_index": previous.get("asset_index", {}) if isinstance(previous.get("asset_index"), dict) else {},
        })

    return LoadedProject(document=document, project_settings=project_settings, project_dir=project_dir,
                         project_id=info.project_id, manifest=info.manifest, notices=notices,
                         video_path=video_source(project_settings, info.path, project_dir, info.manifest))


def session_matches_file(session: dict | None, info: BundleInfo) -> bool:
    """True when the dir was extracted from the file as it is now (size and
    mtime unchanged), so a clean dir can be reused without extracting."""
    if not session:
        return False
    try:
        return (int(session.get("zip_size", -1)) == info.zip_size
                and abs(float(session.get("zip_mtime", -1)) - info.zip_mtime) < 1e-6)
    except (TypeError, ValueError):
        return False


# --- save --------------------------------------------------------------------------


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


# --- imported audio ----------------------------------------------------------------

# The largest file `import_audio_file` copies into a project dir.
MAX_IMPORT_BYTES = 2 * 1024 ** 3


def _import_extension(path: str) -> str:
    """The source's extension, lowercased, letters and digits only, at most
    eight characters: it becomes part of a file name in the project dir.
    `bin` when nothing is left."""
    ext = os.path.splitext(path)[1][1:].lower()
    ext = "".join(ch for ch in ext if ch.isascii() and ch.isalnum())[:8]
    return ext or "bin"


def import_audio_file(src_path: str, project_dir: str, max_bytes: int = MAX_IMPORT_BYTES) -> str:
    """Copies `src_path` to `<project_dir>/audio/imported/<sha256[:16]>.<ext>`
    and returns the copy's absolute path. The name is the content hash, so a
    second import of the same bytes finds the copy and skips the write.
    Refuses a file over `max_bytes` or anything that isn't a regular file.
    Every writer of imported audio (music bed, source track, recording
    import) goes through here and stores the result on the clip."""
    source = os.path.realpath(os.path.abspath(src_path))
    drive = os.path.splitdrive(source)[0]
    if not source.startswith(drive + os.sep) or not os.path.isfile(source):
        raise ProjectError(f"Not a file: {src_path}")
    size = os.path.getsize(source)
    if size > max_bytes:
        raise ProjectError(f"{os.path.basename(source)} is {size / 1024 ** 3:.1f} GB; "
                           f"the import limit is {max_bytes / 1024 ** 3:.1f} GB.")
    h = hashlib.sha256()
    with open(source, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    imported = os.path.join(os.path.abspath(project_dir), *AUDIO_IMPORTED.split("/"))
    os.makedirs(imported, exist_ok=True)
    target = os.path.join(imported, f"{h.hexdigest()[:16]}.{_import_extension(source)}")
    if os.path.isfile(target):
        return target
    tmp = target + ".tmp"
    shutil.copyfile(source, tmp)
    _replace_with_retries(tmp, target)
    return target


# --- reference video (phase 5, TB16) -----------------------------------------------


def video_settings(project_settings) -> dict | None:
    """`project_settings["video"]` as `{"path": str, "offset_s": float}`, or
    None when the project has no reference video."""
    block = project_settings.get("video") if isinstance(project_settings, dict) else None
    if not isinstance(block, dict):
        return None
    path = block.get("path")
    if not isinstance(path, str) or not path.strip():
        return None
    try:
        offset = float(block.get("offset_s") or 0.0)
    except (TypeError, ValueError):
        offset = 0.0
    if offset != offset or offset in (float("inf"), float("-inf")):
        offset = 0.0
    return {"path": path, "offset_s": offset}


def video_path_for(file_path: str, project_file: str | None) -> str:
    """`file_path` as `project_settings["video"]["path"]`: relative to the
    project's `.tbaw` when there is one on the same drive, else absolute
    (the rule a linked subproject's path follows, NP4)."""
    file_path = os.path.abspath(file_path)
    if project_file:
        base_dir = os.path.dirname(os.path.abspath(project_file))
        if os.path.splitdrive(base_dir)[0].lower() == os.path.splitdrive(file_path)[0].lower():
            return os.path.relpath(file_path, base_dir).replace("\\", "/")
    return file_path


def resolve_video_path(project_settings, project_file: str | None) -> str | None:
    """The absolute path `project_settings["video"]["path"]` names, a
    relative one taken from the folder of `project_file` (the `.tbaw`),
    whether or not the file is there. None when there's no video, or a
    relative path and no file to be relative to. The path comes from
    `project.json`, so it's normalised and checked before anything reads it."""
    block = video_settings(project_settings)
    if block is None:
        return None
    raw = block["path"]
    if os.path.isabs(raw):
        candidate = raw
    elif project_file:
        candidate = os.path.join(os.path.dirname(os.path.abspath(project_file)), *raw.replace("\\", "/").split("/"))
    else:
        return None
    real = os.path.realpath(os.path.abspath(candidate))
    drive = os.path.splitdrive(real)[0]
    if not real.startswith(drive + os.sep):
        return None
    return real


def bundled_video_path(project_dir: str | None, manifest: dict | None = None) -> str | None:
    """The reference video Open extracted into `<project_dir>/video/`: the
    one `manifest.assets` names when it's there, else the first file in the
    folder by name. None when there is none."""
    if not project_dir:
        return None
    folder = os.path.realpath(os.path.join(project_dir, VIDEO_DIR))
    if not os.path.isdir(folder):
        return None
    # The manifest's names are bundle data: each is checked to stay inside
    # the folder like any other path read from a file.
    assets = manifest.get("assets") if isinstance(manifest, dict) else None
    names = [os.path.basename(k) for k in assets if isinstance(k, str) and k.startswith(VIDEO_DIR + "/")] \
        if isinstance(assets, dict) else []
    names += sorted(os.listdir(folder))
    for name in names:
        if not name or name.startswith(".") or name.endswith(".tmp"):
            continue
        full = os.path.realpath(os.path.join(folder, name))
        if full.startswith(folder + os.sep) and os.path.isfile(full):
            return full
    return None


def video_source(project_settings, project_file: str | None, project_dir: str | None,
                 manifest: dict | None = None) -> str | None:
    """The file the video dock plays and Save bundles: the settings path
    when that file exists (TB16: it always wins), else the bundled copy in
    the project dir, else None."""
    if video_settings(project_settings) is None:
        return None
    path = resolve_video_path(project_settings, project_file)
    if path and os.path.isfile(path):
        return path
    return bundled_video_path(project_dir, manifest)


def _is_video_entry(info: zipfile.ZipInfo) -> bool:
    return info.filename.replace("\\", "/").startswith(VIDEO_DIR + "/")


def bundled_video_needed(info: BundleInfo) -> bool:
    """Whether Open extracts the bundle's `video/` entry: only when the
    bundle has one and the path its `project.json` names isn't a file on
    this machine. Re-opening a bundle where the video already is skips
    gigabytes (TB16)."""
    if not any(_is_video_entry(e) for e in info.entries):
        return False
    try:
        with zipfile.ZipFile(info.path) as zf:
            settings = json.loads(zf.read(PROJECT_JSON).decode("utf-8"))
    except (OSError, zipfile.BadZipFile, KeyError, json.JSONDecodeError, UnicodeDecodeError):
        return True
    path = resolve_video_path(settings, info.path)
    return not (path and os.path.isfile(path))


def heavy_bytes(info: BundleInfo) -> int:
    """What `extract_audio` will write: `info.audio_bytes`, less the video
    entry when Open skips it."""
    if bundled_video_needed(info):
        return info.audio_bytes
    return info.audio_bytes - sum(e.file_size for e in info.entries if _is_video_entry(e))


def video_extract_pending(info: BundleInfo, project_dir: str) -> bool:
    """True when Open needs the bundled video and the project dir doesn't
    hold it (an earlier Open skipped it because the path was there then),
    so a clean dir can't be reused as it is."""
    return bundled_video_needed(info) and bundled_video_path(project_dir, info.manifest) is None


def _cached_video_digest(previous_index: dict, source: str, stat) -> str | None:
    """The digest the last Save recorded for `source` when its size and
    mtime haven't changed. Video entries are keyed by their bundle name,
    which is the hash, so the lookup goes by the source path stored with them."""
    for name, entry in (previous_index or {}).items():
        if not (isinstance(name, str) and name.startswith(VIDEO_DIR + "/")):
            continue
        if isinstance(entry, list) and len(entry) == 4 and entry[3] == source and entry[0] == stat.st_size \
                and abs(float(entry[1]) - stat.st_mtime) < 1e-6:
            return entry[2]
    return None


def used_voice_names(document: Document) -> dict:
    """`{backend_id: {voice name, ...}}` from every character's preset and
    every clip override, grouped by the clip's character's backend. A
    character's variants count too, so every reference they name is bundled."""
    names: dict = {}
    for character in document.characters:
        voice = (character.preset_data or {}).get("voice")
        if voice:
            names.setdefault(character.backend_id or "kokoro", set()).add(str(voice))
        for variant_voice in (character.variants or {}).values():
            if variant_voice:
                names.setdefault(character.backend_id or "kokoro", set()).add(str(variant_voice))
    for clip in document.clips:
        voice = (clip.overrides or {}).get("voice")
        if voice:
            character = document.get_character(clip.character_id)
            backend_id = (character.backend_id if character else None) or "kokoro"
            names.setdefault(backend_id, set()).add(str(voice))
    return names


def used_fx_preset_names(document: Document) -> set:
    names = set()
    for character in document.characters:
        preset = (character.preset_data or {}).get("fx_preset")
        if preset:
            names.add(str(preset))
    for clip in document.clips:
        preset = (clip.overrides or {}).get("fx_preset")
        if preset:
            names.add(str(preset))
    return names


def collect_assets(document: Document, backend_for, project_dir: str | None, fx_presets_dir: str) -> tuple:
    """`(assets, engines, warnings)`: every `(bundle_path, source_path)` the
    document needs, the `manifest.engines` block (version + meta per used
    backend), and one warning per asset that resolves nowhere. `backend_for(id)`
    returns an adapter or `None` for an engine that isn't registered."""
    assets = []
    engines = {}
    warnings = []
    for backend_id, names in sorted(used_voice_names(document).items()):
        backend = backend_for(backend_id)
        if backend is None:
            warnings.append(f"engine {backend_id!r} isn't installed; its voices aren't bundled")
            continue
        try:
            found, meta = backend.collect_project_assets(set(names), project_dir)
        except Exception as e:  # noqa: BLE001 - one backend's failure shouldn't stop a Save
            warnings.append(f"{backend_id}: couldn't collect voice files ({e})")
            found, meta = [], {}
        bundled = {os.path.splitext(os.path.basename(a.bundle_path))[0] for a in found}
        for name in sorted(names):
            if os.path.basename(name) not in bundled and backend.resolve_voice_file(name, project_dir) is None \
                    and getattr(backend.capabilities, "supports_voice_cloning", False):
                warnings.append(f"voice reference {name!r} not found; not bundled")
        assets.extend((a.bundle_path, a.source_path) for a in found)
        engines[backend_id] = {"version": backend.engine_version(), "meta": dict(meta or {})}
    for name in sorted(used_fx_preset_names(document)):
        safe = os.path.basename(name)
        candidates = []
        if project_dir:
            candidates.append(os.path.join(project_dir, FX_DIR, f"{safe}.json"))
        candidates.append(os.path.join(fx_presets_dir, f"{safe}.json"))
        source = next((c for c in candidates if os.path.isfile(c)), None)
        if source is None:
            warnings.append(f"FX preset {name!r} not found; not bundled")
            continue
        assets.append((f"{FX_DIR}/{safe}.json", os.path.abspath(source)))
    return assets, engines, warnings


def bundle_options(project_settings: dict) -> dict:
    options = dict(DEFAULT_BUNDLE_OPTIONS)
    block = project_settings.get("bundle") if isinstance(project_settings, dict) else None
    if isinstance(block, dict):
        options.update({k: v for k, v in block.items() if k in DEFAULT_BUNDLE_OPTIONS})
    if options["audio_format"] not in ("wav", "flac"):
        options["audio_format"] = "wav"
    return options


def required_features(document: Document) -> list:
    """`manifest.requires`: the content features a v1 reader would lose or
    misplay (section 8 of the bundle plan). Parked takes (an older reader
    would drop them on its next Save and GC their files) and subprojects
    (it would read a nested clip's source as unknown and refuse the
    document), and imported audio (grill Q30: it would find a clip with no
    segments and push its file name back through TTS)."""
    requires = []
    if any(clip.takes for clip in document.clips):
        requires.append("takes")
    if any(clip.source == "nested" for clip in document.clips):
        requires.append("nested")
    if any(clip.source == "imported" for clip in document.clips):
        requires.append("imported")
    return requires


def project_stats(document: Document) -> dict:
    """Cosmetic, for the welcome dialog: list lengths and the sum of
    `Segment.duration`. Nothing here reads audio."""
    duration = 0.0
    for clip in document.clips:
        for segment in clip.segments:
            duration += float(segment.duration or 0.0)
    stats = {"clips": len(document.clips), "characters": len(document.characters), "duration_s": round(duration, 3)}
    subprojects = sum(1 for clip in document.clips if clip.source == "nested")
    if subprojects:
        stats["subprojects"] = subprojects
    return stats


@dataclass
class SavePlan:
    """Everything Save needs, assembled on the GUI thread (a snapshot) so the
    worker never reads dock state or the live document."""
    path: str
    project_dir: str
    project_id: str
    manifest: dict
    document_bytes: bytes
    project_bytes: bytes
    assets: list  # (bundle_path, source_path)
    audio_files: list  # (bundle_path, source_path)
    previous_asset_index: dict
    # Digest of the document as autosave writes it into the project dir
    # (absolute paths), which is what `session.json`'s `saved_digest`
    # compares against; the zip's copy has bundle-relative paths.
    dir_digest: str = ""
    # The reference video to store under `video/` (`include_video` on), or
    # None. Hashed by `write_bundle`, on the worker thread.
    video_file: str | None = None


def plan_save(document: Document, project_settings: dict, path: str, project_dir: str, project_id: str,
              backend_for, fx_presets_dir: str, previous_session: dict | None = None,
              previous_manifest: dict | None = None, pending_children=()) -> tuple:
    """`(SavePlan, warnings)`. Serializes the document with bundle-relative
    audio paths, collects assets and the referenced audio files.
    `pending_children` are embedded child ids whose bundle the same Save
    writes into the project dir before this plan is written."""
    options = bundle_options(project_settings)
    data = serialization.document_to_dict(document)
    audio_files = []
    seen = set()
    project_root = os.path.realpath(project_dir) if project_dir else None

    def to_relative(abs_path):
        # Only a file inside the project dir goes into the bundle. Every
        # generated or migrated segment lives there; a path anywhere else
        # can only have come from a hand-edited or crafted document, and
        # bundling it would ship that file. It's left as written, so Open
        # reports it missing.
        if not os.path.isabs(abs_path):
            return abs_path.replace("\\", "/")
        real = os.path.realpath(abs_path)
        if project_root and real.startswith(project_root + os.sep) and os.path.isfile(real):
            rel = os.path.relpath(real, project_root).replace("\\", "/")
            if rel not in seen:
                seen.add(rel)
                audio_files.append((rel, real))
            return rel
        return abs_path.replace("\\", "/")

    serialization.rewrite_audio_paths(data, to_relative)
    # Embedded subprojects: each child's bundle as its parent's project dir
    # holds it (the app writes an open child's bundle there first).
    embedded = []
    for child_id in embedded_child_ids(document):
        child_path = embedded_child_path(project_dir, child_id)
        if child_path and (os.path.isfile(child_path) or child_id in pending_children):
            audio_files.append((f"{PROJECTS_DIR}/{child_id}{DEFAULT_EXTENSION}", os.path.abspath(child_path)))
            embedded.append(child_id)
    if not options["include_generated_audio"]:
        audio_files = [a for a in audio_files if not a[0].startswith(AUDIO_GENERATED + "/")]
    if not options["include_imported_audio"]:
        audio_files = [a for a in audio_files if not a[0].startswith(AUDIO_IMPORTED + "/")]

    assets, engines, warnings = collect_assets(document, backend_for, project_dir, fx_presets_dir)
    video_file = None
    if options["include_video"] and video_settings(project_settings) is not None:
        video_file = video_source(project_settings, path, project_dir, previous_manifest)
        if video_file is None:
            warnings.append("reference video not found; not bundled")
    now = datetime.datetime.now().replace(microsecond=0).isoformat()
    created = (previous_manifest or {}).get("created") or now
    manifest = {
        "format": FORMAT,
        "version": SUPPORTED_VERSION,
        "requires": required_features(document),
        "project_id": project_id,
        "created_by": f"KokoroGUI {APP_VERSION}",
        "created": created,
        "modified": now,
        "includes": {
            "generated_audio": bool(options["include_generated_audio"]),
            "imported_audio": bool(options["include_imported_audio"])
            and any(name.startswith(AUDIO_IMPORTED + "/") for name, _src in audio_files),
            "projects": embedded,
            "video": video_file is not None,
        },
        "audio": {"format": options["audio_format"]},
        "stats": project_stats(document),
        "engines": engines,
        "assets": {},  # filled by write_bundle once hashed
    }
    dir_document, dir_project = serialize_for_dir(document, project_settings, project_dir)
    plan = SavePlan(
        path=os.path.abspath(path), project_dir=project_dir, project_id=project_id, manifest=manifest,
        document_bytes=json.dumps(data, indent=2).encode("utf-8"),
        project_bytes=json.dumps(dict(project_settings or {}), indent=2).encode("utf-8"),
        assets=assets, audio_files=audio_files,
        previous_asset_index=dict((previous_session or {}).get("asset_index") or {}),
        dir_digest=document_digest(dir_document, dir_project), video_file=video_file,
    )
    return plan, warnings


def _owned_entry(name: str, known_engine_ids) -> bool:
    name = name.replace("\\", "/")
    if name in _OWNED_FILES or name.startswith(_OWNED_DIRS):
        return True
    for engine_id in known_engine_ids:
        if name.startswith(f"{ENGINES_DIR}/{engine_id}/"):
            return True
    return False


def _replace_with_retries(src: str, dst: str) -> None:
    """`os.replace`, retried over ~2 s on Windows where a sync client or a
    scanner holding the old file raises `PermissionError`."""
    attempts = 8 if sys.platform == "win32" else 1
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.25)


def write_bundle(plan: SavePlan, known_engine_ids, progress=None) -> SaveResult:
    """Steps 1-4 of Save, meant for a worker thread: hash assets not in the
    previous index, check free space, write `<path>.tmp` (JSON deflated,
    audio stored, unknown entries from the old file copied through), then
    replace. A crash mid-save leaves the old file intact; a replace that
    keeps failing leaves the `.tmp` and says where it is."""
    asset_index = {}
    manifest = dict(plan.manifest)
    manifest["assets"] = {}
    for bundle_path, source in plan.assets:
        try:
            stat = os.stat(source)
        except OSError:
            continue
        previous = plan.previous_asset_index.get(bundle_path)
        if isinstance(previous, list) and len(previous) == 3 and previous[0] == stat.st_size \
                and abs(float(previous[1]) - stat.st_mtime) < 1e-6:
            digest = previous[2]
        else:
            digest = _sha256_file(source)
        asset_index[bundle_path] = [stat.st_size, stat.st_mtime, digest]
        manifest["assets"][bundle_path] = digest

    # The reference video: named by its hash, which is only recomputed when
    # the file's size or mtime changed since the last Save.
    video = None
    if plan.video_file:
        try:
            stat = os.stat(plan.video_file)
        except OSError:
            stat = None
        if stat is not None:
            digest = _cached_video_digest(plan.previous_asset_index, plan.video_file, stat) \
                or _sha256_file(plan.video_file)
            name = f"{VIDEO_DIR}/{digest.split(':', 1)[-1][:16]}.{_import_extension(plan.video_file)}"
            video = (name, plan.video_file)
            asset_index[name] = [stat.st_size, stat.st_mtime, digest, plan.video_file]
            manifest["assets"][name] = digest
    manifest["includes"] = dict(manifest.get("includes") or {}, video=video is not None)

    needed = len(plan.document_bytes) + len(plan.project_bytes)
    for _bundle_path, source in plan.assets + plan.audio_files + ([video] if video else []):
        try:
            needed += os.path.getsize(source)
        except OSError:
            pass
    previous_path = plan.path if os.path.isfile(plan.path) else None
    carried = []
    if previous_path:
        try:
            with zipfile.ZipFile(previous_path) as old:
                for info in old.infolist():
                    if not _owned_entry(info.filename, known_engine_ids):
                        carried.append(info.filename)
                        needed += info.file_size
        except (OSError, zipfile.BadZipFile):
            carried = []
    check_free_space(plan.path, needed, "save the project")

    tmp = plan.path + ".tmp"
    parent = os.path.dirname(plan.path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    total = max(1, needed)
    done = 0
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
            zf.writestr(MANIFEST, json.dumps(manifest, indent=2))
            zf.writestr(DOCUMENT, plan.document_bytes)
            zf.writestr(PROJECT_JSON, plan.project_bytes)
            for bundle_path, source in plan.assets:
                if os.path.isfile(source):
                    zf.write(source, bundle_path, compress_type=zipfile.ZIP_DEFLATED)
                    done += os.path.getsize(source)
            for bundle_path, source in plan.audio_files + ([video] if video else []):
                if not os.path.isfile(source):
                    continue
                zf.write(source, bundle_path, compress_type=zipfile.ZIP_STORED)
                done += os.path.getsize(source)
                if progress is not None:
                    progress(done, total)
            if carried:
                with zipfile.ZipFile(previous_path) as old:
                    for name in carried:
                        info = old.getinfo(name)
                        if info.is_dir():
                            zf.writestr(info, b"")
                            continue
                        with old.open(info) as src, zf.open(zipfile.ZipInfo(info.filename, info.date_time), "w") as dst:
                            shutil.copyfileobj(src, dst, 1024 * 1024)
                        done += info.file_size
        _replace_with_retries(tmp, plan.path)
    except PermissionError as e:
        raise ProjectError(f"Couldn't replace {plan.path} (another program holds it): {e}. "
                           f"The new version is at {tmp}.")
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    stat = os.stat(plan.path)
    return SaveResult(zip_size=stat.st_size, zip_mtime=stat.st_mtime, asset_index=asset_index,
                      saved_digest=plan.dir_digest)


def record_save(project_dir: str, path: str, result: SaveResult, source_path: str | None = None) -> None:
    """Step 5, on the GUI thread: `session.json` after a Save. An embedded
    child passes its `<parent>#<id>` `source_path`; `path` is then the
    bundle inside the parent's project dir."""
    session = read_session(project_dir) or {}
    session.update({
        "source_path": source_path or os.path.abspath(path), "zip_size": result.zip_size,
        "zip_mtime": result.zip_mtime,
        "saved_digest": result.saved_digest, "dirty": False, "asset_index": result.asset_index,
    })
    write_session(project_dir, session)


def save_project(document: Document, path: str, project_settings: dict | None = None,
                 project_dir: str | None = None, project_id: str | None = None,
                 backend_for=None, fx_presets_dir: str = os.path.join("presets", "fx"),
                 known_engine_ids=()) -> SaveResult:
    """Synchronous Save for callers without an app (tests, tooling): plans
    and writes in one go. A `.json` path is written in the legacy shape."""
    if format_for_path(path) != "tbaw":
        save_json_project(document, path, project_settings)
        return SaveResult(zip_size=os.path.getsize(path), zip_mtime=os.path.getmtime(path), asset_index={},
                          saved_digest="")
    project_settings = dict(project_settings or {})
    if project_dir is None:
        project_dir, project_id = create_project_dir(project_id)
    plan, _warnings = plan_save(document, project_settings, path, project_dir, project_id or new_project_id(),
                                backend_for or (lambda _id: None), fx_presets_dir, read_session(project_dir))
    result = write_bundle(plan, known_engine_ids)
    record_save(project_dir, path, result)
    return result


# --- close-time GC, eviction, sweep ---------------------------------------------


def referenced_audio_paths(document: Document) -> set:
    """Every file a clip points at: its original audio, its active take's
    segments and every parked take's, so close-time GC keeps them all."""
    paths = set()
    for clip in document.clips:
        if clip.original_audio_path:
            paths.add(os.path.realpath(clip.original_audio_path))
        for segments in (clip.segments, *clip.takes.values()):
            for segment in segments:
                if segment.audio_path:
                    paths.add(os.path.realpath(segment.audio_path))
    return paths


def gc_project_dir(project_dir: str, document: Document) -> list:
    """Deletes every file under `audio/generated/` that no segment
    references (regenerated-over takes, cancelled attempts, stale
    reservation markers), and a subproject's mixdown that no longer matches
    its document (autosave's digest of the dir's `document.json` and
    `project.json`). Only at a clean close: the undo stack may still point
    at any of them while the session lives (TB11). Returns what it
    removed."""
    removed = []
    # A subproject's mixdown older than its document is dead weight: it
    # re-renders from the document anyway (phase 4).
    info = read_mixdown_info(project_dir)
    if info is not None:
        try:
            with open(os.path.join(project_dir, DOCUMENT), "rb") as f:
                document_bytes = f.read()
            project_json = os.path.join(project_dir, PROJECT_JSON)
            project_bytes = b"{}"
            if os.path.isfile(project_json):
                with open(project_json, "rb") as f:
                    project_bytes = f.read()
            current = document_digest(document_bytes, project_bytes)
        except OSError:
            current = None
        if current is not None and info.get("digest") != current:
            remove_mixdown(project_dir)
            removed.append(info["file"])
    generated = os.path.join(project_dir, *AUDIO_GENERATED.split("/"))
    if not os.path.isdir(generated):
        return removed
    keep = referenced_audio_paths(document)
    for name in os.listdir(generated):
        full = os.path.join(generated, name)
        if not os.path.isfile(full):
            continue
        if os.path.realpath(full) in keep and not name.endswith(RESERVED_SUFFIX):
            continue
        try:
            os.remove(full)
            removed.append(full)
        except OSError:
            pass
    return removed


def evict_project_dirs(keep_project_dir: str | None) -> list:
    """TB13: on a clean close every dir under `cache/projects/` except the
    one to keep (the `last_project`'s) and its subprojects' is deleted,
    unless it's locked by another window or dirty (a crash's recovery
    data). Returns the removed dirs."""
    root = projects_root()
    if not os.path.isdir(root):
        return []
    keep = os.path.realpath(keep_project_dir) if keep_project_dir else None
    # The kept project's subprojects stay with it (phase 4): their dirs
    # carry their mixdowns, which aren't in any bundle.
    keep_children = set()
    keep_session = read_session(keep_project_dir) if keep_project_dir else None
    keep_source = (keep_session or {}).get("source_path")
    if isinstance(keep_source, str) and keep_source:
        keep_children = {os.path.realpath(d) for d in child_dirs_of(keep_source)}
    removed = []
    for name in os.listdir(root):
        full = os.path.join(root, name)
        if not os.path.isdir(full) or (keep and os.path.realpath(full) == keep):
            continue
        if os.path.realpath(full) in keep_children:
            continue
        session = read_session(full)
        if session and session.get("dirty"):
            continue
        if is_locked(full):
            continue
        delete_project_dir(full)
        removed.append(full)
    return removed


def sweep_orphan_dirs() -> list:
    """On Open: any clean, unlocked dir whose `source_path` no longer exists
    (a crash after the file was moved) is deleted."""
    root = projects_root()
    if not os.path.isdir(root):
        return []
    removed = []
    for name in os.listdir(root):
        full = os.path.join(root, name)
        if not os.path.isdir(full):
            continue
        session = read_session(full)
        if not session or session.get("dirty"):
            continue
        source = session.get("source_path")
        if not isinstance(source, str) or not source:
            continue
        # An embedded child's source is `<parent file>#<id>`: it's an orphan
        # when the parent file is gone.
        source = parent_source_of(source)
        # `record_save` and `finish_open` write an absolute path; a relative
        # or drive-relative one is a corrupt session, not grounds to delete.
        norm = os.path.normpath(source)
        drive, _tail = os.path.splitdrive(norm)
        if not norm.startswith(drive + os.sep):
            continue
        if not os.path.exists(norm) and not is_locked(full):
            delete_project_dir(full)
            removed.append(full)
    return removed


# --- .json migration ---------------------------------------------------------------


def legacy_segment_key(text: str, config: dict) -> str:
    """What `dirty.py` stamped before the schema bump: the voice *name*, no
    extra inputs, `CACHE_SCHEMA_VERSION` 2."""
    return compute_cache_key(text, config.get("voice"), effective_speed(config), config.get("lang_code", "a"),
                             config.get("engine_id", "kokoro"), schema_version=2)


def migrate_segments(document: Document, project_dir: str, generation_config_for, key_fn,
                     audio_format: str = "wav") -> dict:
    """Rekeys a `.json` project's segments into the project dir (TB6, revised
    by the third review). For each clip, a segment whose stored key equals
    the legacy expected key and whose file exists is adopted: copied to
    `audio/generated/<new key>_<idx>.<ext>` and restamped. Anything else
    gets `audio_path = None` and stays dirty; migration never makes a stale
    segment look clean. The originals stay where they are. Returns counts."""
    generated = os.path.join(project_dir, *AUDIO_GENERATED.split("/"))
    os.makedirs(generated, exist_ok=True)
    adopted = dropped = 0
    for clip in document.clips:
        if not clip.segments:
            continue
        text = document.clip_text(clip)
        config = generation_config_for(clip)
        legacy = legacy_segment_key(text, config)
        new_key = key_fn(text, clip)
        for segment in clip.segments:
            source = segment.audio_path
            if segment.cache_key == legacy and source and os.path.isfile(source):
                ext = os.path.splitext(source)[1].lstrip(".").lower() or audio_format
                target = os.path.join(generated, f"{new_key}_{segment.order_index}.{ext}")
                if os.path.realpath(source) != os.path.realpath(target):
                    shutil.copyfile(source, target)
                segment.audio_path = os.path.abspath(target)
                segment.cache_key = new_key
                adopted += 1
            else:
                segment.audio_path = None
                dropped += 1
    return {"adopted": adopted, "dropped": dropped}


# --- welcome dialog ------------------------------------------------------------------


def _embedded_duration_s(path: str) -> float:
    """The summed `stats.duration_s` of every embedded child (and theirs),
    read from the manifests inside the bundle without extracting."""
    try:
        with zipfile.ZipFile(path) as zf:
            return _embedded_duration_in(zf, 0)
    except (OSError, zipfile.BadZipFile):
        return 0.0


def _embedded_duration_in(zf: zipfile.ZipFile, depth: int) -> float:
    if depth > 8:
        return 0.0
    total = 0.0
    for name in zf.namelist():
        if not (name.startswith(PROJECTS_DIR + "/") and name.endswith(DEFAULT_EXTENSION)):
            continue
        try:
            with zf.open(name) as inner_file, zipfile.ZipFile(inner_file) as inner:
                manifest = json.loads(inner.read(MANIFEST).decode("utf-8"))
                stats = manifest.get("stats") if isinstance(manifest, dict) else None
                if isinstance(stats, dict):
                    total += float(stats.get("duration_s", 0.0) or 0.0)
                total += _embedded_duration_in(inner, depth + 1)
        except (OSError, zipfile.BadZipFile, KeyError, ValueError, json.JSONDecodeError):
            continue
    return total


def project_summary(path: str) -> dict | None:
    """What the welcome dialog's details pane shows for a row. A `.tbaw`
    answers from `manifest.json` alone (`ZipFile.read`, no extraction, no
    `document.json` parse) since this runs on every selection change; a
    `.json` entry that hasn't been opened (and so migrated) yet still gets
    the raw-JSON reader. `None` when missing or unreadable."""
    if not path or not os.path.isfile(path):
        return None
    try:
        modified = datetime.datetime.fromtimestamp(os.path.getmtime(path))
    except (OSError, ValueError):
        return None
    if format_for_path(path) == "tbaw":
        try:
            with zipfile.ZipFile(path) as zf:
                manifest = json.loads(zf.read(MANIFEST).decode("utf-8"))
        except (OSError, zipfile.BadZipFile, KeyError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        if not isinstance(manifest, dict):
            return None
        stats = manifest.get("stats") if isinstance(manifest.get("stats"), dict) else {}
        engines = manifest.get("engines") if isinstance(manifest.get("engines"), dict) else {}
        stamp = manifest.get("modified")
        if isinstance(stamp, str):
            try:
                modified = datetime.datetime.fromisoformat(stamp)
            except ValueError:
                pass
        duration = float(stats.get("duration_s", 0.0) or 0.0)
        # Embedded subprojects' audio, from their own manifests (phase 4).
        duration += _embedded_duration_s(path)
        return {
            "path": os.path.abspath(path),
            "modified": modified,
            "characters": int(stats.get("characters", 0) or 0),
            "clips": int(stats.get("clips", 0) or 0),
            "subprojects": int(stats.get("subprojects", 0) or 0),
            "duration_s": duration,
            "engines": sorted(engines.keys()),
        }
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    characters = data.get("characters", [])
    clips = data.get("clips", [])
    return {
        "path": os.path.abspath(path),
        "modified": modified,
        "characters": len(characters) if isinstance(characters, list) else 0,
        "clips": len(clips) if isinstance(clips, list) else 0,
        "duration_s": None,
        "engines": [],
    }


def new_document_from(library, settings: dict | None = None) -> Document:
    """File > New: an empty document seeded with every character library
    entry, linked (WF3 through the WF4-WF7 library), and no tracks; a
    character gets its track the first time the transcript uses it (grill
    PR4). An empty library gives one local "Default" character from
    `settings`. The previous project's characters are not copied."""
    from kokoro_gui.daw.migration import migrate_legacy_settings_to_document

    return migrate_legacy_settings_to_document(settings or {}, library)
