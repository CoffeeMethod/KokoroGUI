"""Proof by ASR (plan 21): the app half.

`ProofMixin` gives `QtTTSApp` what it takes to proof clips: a worker thread
that transcribes each segment of a clip with Whisper and scores the heard
words against the clip's text (`daw/proof.py`), the results store, and the
questions the Proof dock and the timeline ask of it.

Results are derived data. They live in the root project dir's
`session.json["proof"]` (`{clip_id: entry}`, clip ids are unique across the
open projects), never in a `Clip` or the bundle, and an entry is current only
while the clip's segment keys are the ones it was scored on. The GUI thread
is the only writer: the worker hands its scores back through
`_proofFinished`.

The heard words are kept per file (`remember_heard`) so a proof after Align
words, or a second proof of the same clip, doesn't transcribe it again. The
signals `_proofProgress` and `_proofFinished` are declared on `QtTTSApp`.

Kept out of app.py, which already holds the word-alignment worker.
"""
from __future__ import annotations

import os
import threading

from PySide6.QtWidgets import QApplication

from kokoro_gui.daw import proof, segment_view
from kokoro_gui.qt import project as project_io

SCOPE_ALL, SCOPE_SELECTION, SCOPE_SUBPROJECT = "all", "selection", "subproject"
# Files whose heard words are remembered; the oldest go first.
MAX_HEARD_FILES = 4000


def _file_stamp(path: str):
    """What identifies a file's content for the heard-words memo, or None
    when it isn't there."""
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


class ProofMixin:
    def _init_proof(self) -> None:
        self._proof_thread: threading.Thread | None = None
        self._proof_cancel = threading.Event()
        # `{path: (stamp, [(word, start_s, end_s)])}`. Written by the
        # alignment and proof workers, read by the proof worker, so locked.
        self._heard_words: dict = {}
        self._heard_lock = threading.Lock()
        # `session.json["proof"]` for `_proof_dir`; None until read.
        self._proof_entries: dict | None = None
        self._proof_dir: str | None = None

    # -- heard words -----------------------------------------------------------

    def remember_heard(self, path: str, words) -> None:
        stamp = _file_stamp(path)
        if stamp is None:
            return
        with self._heard_lock:
            self._heard_words.pop(path, None)
            self._heard_words[path] = (stamp, list(words))
            while len(self._heard_words) > MAX_HEARD_FILES:
                self._heard_words.pop(next(iter(self._heard_words)))

    def heard_for(self, path: str):
        """The words Whisper heard in `path`, if this session transcribed
        that file as it is now; else None."""
        with self._heard_lock:
            hit = self._heard_words.get(path)
        if hit is None or hit[0] != _file_stamp(path):
            return None
        return hit[1]

    # -- the results store -----------------------------------------------------

    def proof_threshold(self) -> float:
        return proof.clamp_threshold(self.settings.get("proof_threshold"))

    def set_proof_threshold(self, value: float) -> None:
        self._set_setting("proof_threshold", proof.clamp_threshold(value))
        self.refresh_timeline()

    def proof_store(self) -> dict:
        """`{clip_id: entry}` from the root's `session.json`, read once per
        project dir and checked (`proof.clean_results`)."""
        directory = self.root.project_dir
        if self._proof_entries is None or self._proof_dir != directory:
            session = project_io.read_session(directory) if directory else None
            self._proof_entries = proof.clean_results((session or {}).get("proof"))
            self._proof_dir = directory
        return self._proof_entries

    def _write_proof(self) -> None:
        """Writes the store to `session.json`, without the entries of clips
        that now have other audio."""
        store = self.proof_store()
        clips = {clip.id: clip for project in self.open_projects() for clip in project.document.clips}
        for clip_id in [cid for cid, entry in store.items()
                        if cid in clips and not proof.is_current(entry, clips[cid])]:
            del store[clip_id]
        directory = self.root.project_dir
        if not directory:
            return
        session = project_io.read_session(directory) or {}
        if store:
            session["proof"] = store
        else:
            session.pop("proof", None)
        try:
            project_io.write_session(directory, session)
        except OSError:
            pass

    def carry_proof_over(self, previous: dict | None, project_dir: str) -> None:
        """Open rewrites `session.json` and keeps only `RUNTIME_STATE_KEYS`.
        When the same file reopens into its own dir, puts the previous
        session's `proof` back. Takes `previous` as read before
        `finish_open`."""
        self._proof_entries = None
        session = project_io.read_session(project_dir)
        if not previous or not session or "proof" in session or "proof" not in previous:
            return
        if previous.get("source_path") != session.get("source_path"):
            return
        session["proof"] = previous["proof"]
        try:
            project_io.write_session(project_dir, session)
        except OSError:
            pass

    def proof_entry(self, clip):
        """The current result for `clip`, or None (never proofed, or
        regenerated since)."""
        entry = self.proof_store().get(clip.id)
        return entry if entry is not None and proof.is_current(entry, clip) else None

    def is_clip_flagged(self, clip) -> bool:
        """Whether `clip` is in the review queue: its current result is
        below the threshold, it isn't marked OK, and its status isn't
        already "Needs rewrite"."""
        store = self.proof_store()
        if not store or clip.status == "needs_rewrite":
            return False
        return proof.is_flagged(store.get(clip.id), clip, self.proof_threshold())

    def mark_proof_ok(self, clip_id: str) -> bool:
        """Takes `clip_id` off the flagged list for the audio it has now. A
        regenerate brings it back to be proofed. Not undoable: it is not
        project data."""
        project = self.project_of_clip_id(clip_id)
        clip = project.document.get_clip(clip_id) if project is not None else None
        entry = self.proof_entry(clip) if clip is not None else None
        if entry is None:
            return False
        entry["ok"] = True
        self._write_proof()
        self.refresh_timeline()
        return True

    # -- running a proof -------------------------------------------------------

    @property
    def is_proofing(self) -> bool:
        return self._proof_thread is not None

    def proofable(self, clip) -> bool:
        """A clip generated from its text, with every segment's file on
        disk."""
        return (segment_view.has_pieces(clip) and bool(clip.segments)
                and all(s.audio_path and os.path.isfile(s.audio_path) for s in clip.segments))

    def clips_for_proof_scope(self, scope: str) -> tuple:
        """`(clip_ids, message)`: the proofable clips `scope` names ("all"
        the level's, the "selection", or "subproject": the one a selected
        subproject block stands for, else the level when it is a
        subproject). `message` says why the list is empty."""
        level = self.level
        project = level
        ids = None
        if scope == SCOPE_SELECTION:
            ids = self.selected_clip_ids(level)
            if not ids:
                return [], "Select a clip first."
        elif scope == SCOPE_SUBPROJECT:
            selected = level.document.get_clip(self.selection.selected_clip_id) \
                if self.selection.selected_clip_id else None
            if selected is not None and selected.is_nested:
                project = self.child_project(selected)
                if project is None:
                    return [], "Open the subproject first (double-click its block)."
            elif level is self.root:
                return [], "Select a subproject block, or enter one, first."
        clips = [c for c in project.document.clips if (ids is None or c.id in ids) and self.proofable(c)]
        return [c.id for c in clips], ("" if clips else "No generated clips to proof.")

    def run_proof(self, clip_ids) -> bool:
        """Transcribes each segment of the listed clips on one worker thread
        and scores the clip's text against what was heard. Behind the
        Whisper download prompt when a file has to be transcribed. False
        when nothing was started."""
        from kokoro_gui.qt import asr_prompt

        if self._proof_thread is not None or self._word_align_thread is not None or self.is_busy():
            self.set_status("Wait for the current job to finish before proofing.", "warning")
            return False
        jobs = []
        for clip_id in clip_ids:
            project = self.project_of_clip_id(clip_id)
            clip = project.document.get_clip(clip_id) if project is not None else None
            if clip is None or not self.proofable(clip):
                continue
            ordered = sorted(clip.segments, key=lambda s: s.order_index)
            jobs.append((clip.id, proof.segment_keys(clip), proof.expected_text(clip),
                         [s.audio_path for s in ordered]))
        if not jobs:
            self.set_status("No generated clips to proof.", "warning")
            return False
        if any(self.heard_for(path) is None for _id, _keys, _text, paths in jobs for path in paths):
            choice, _downloading = asr_prompt.confirm_whisper_download(self)
            if choice != asr_prompt.PROCEED:
                self.set_status("Proof needs the Whisper model.", "warning")
                return False

        total = sum(len(paths) for _id, _keys, _text, paths in jobs)
        cancel = self._proof_cancel = threading.Event()

        def _work():
            from kokoro_gui.engine import asr

            scored, skipped, done = [], 0, 0
            for clip_id, keys, expected, paths in jobs:
                heard, failed = [], False
                for path in paths:
                    if cancel.is_set():
                        break
                    self._proofProgress.emit(done, total)
                    words = self.heard_for(path)
                    if words is None:
                        try:
                            words = asr.transcribe_wav_words(path, "whisper")
                        except Exception:  # noqa: BLE001 - one bad file skips one clip
                            failed = True
                            break
                        self.remember_heard(path, words)
                    heard.extend(words)
                    done += 1
                else:
                    scored.append((clip_id, keys, proof.score(expected, heard)))
                    continue
                if not failed:
                    break  # cancelled between segments
                skipped += 1
            self._proofFinished.emit((scored, skipped, cancel.is_set()))

        self._proof_thread = threading.Thread(target=_work, name="proof", daemon=True)
        self._proof_thread.start()
        self.transport_dock.set_progress(0, f"Proofing 0/{total}")
        self.set_status(f"Proofing {len(jobs)} clip(s)...", "busy")
        return True

    def cancel_proof(self) -> None:
        """Stops the proof after the segment it is on. Clips scored so far
        keep their results."""
        if self._proof_thread is not None:
            self._proof_cancel.set()
            self.set_status("Cancelling the proof...", "warning")

    def wait_for_proof(self, timeout_s: float = 30.0) -> None:
        """Test hook: blocks until the proof thread finishes and its result
        has been applied."""
        thread = self._proof_thread
        if thread is not None:
            thread.join(timeout_s)
        QApplication.processEvents()

    def _on_proof_progress(self, done: int, total: int) -> None:
        percent = 100 * done / total if total else 0
        self.transport_dock.set_progress(percent, f"Proofing {done}/{total}")

    def _on_proof_finished(self, payload) -> None:
        self._proof_thread = None
        scored, skipped, cancelled = payload
        store = self.proof_store()
        stored = flagged = 0
        threshold = self.proof_threshold()
        for clip_id, keys, result in scored:
            project = self.project_of_clip_id(clip_id)
            clip = project.document.get_clip(clip_id) if project is not None else None
            if clip is None or proof.segment_keys(clip) != keys:
                continue  # edited or regenerated while the proof ran
            store[clip_id] = proof.make_entry(result, keys)
            stored += 1
            flagged += proof.flagged(result, threshold)
        self._write_proof()
        self.transport_dock.set_progress(0, "")
        self.refresh_timeline()
        parts = [f"Proofed {stored} clip(s), {flagged} flagged"]
        if skipped:
            parts.append(f"{skipped} skipped (a file couldn't be transcribed)")
        if cancelled:
            parts.append("cancelled")
        self.set_status(". ".join(parts) + ".", "warning" if flagged else "success")
