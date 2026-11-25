"""The Workspace menu and the layout's save timing (plan 47 pass 1).

`WorkspaceMenuMixin` is mixed into `QtTTSApp`. It builds the menu, runs Save
workspace as, Update, Revert, Rename, Delete and Restore default layout
through `kokoro_gui.qt.workspace.WorkspaceManager`, and lists every dock under
Panels with the dock's own `toggleViewAction()`, so closing a dock is a hide
and checking it shows the same widget again.

A layout edit (a dock closed, shown, moved, floated or docked, a separator
dragged, the window resized) starts `_layout_timer`. When it fires,
`save_layout` captures the layout and writes `config_qt.json`. That path
never touches a project: no autosave, no "*" in the title.
"""
from __future__ import annotations

from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QGuiApplication
from PySide6.QtWidgets import QInputDialog, QMenu, QMessageBox, QTabBar, QTabWidget, QToolButton

from kokoro_gui.i18n import N_, tr
from kokoro_gui.qt.workspace import (
    ADVANCED,
    SIMPLE,
    WORKSPACE_NAMES,
    WorkspaceExistsError,
    WorkspaceManager,
    WorkspaceNameError,
    menu_label,
)

LAYOUT_SAVE_DEBOUNCE_MS = 1000

NO_VOICE_EDITOR_TIP = N_("The engine in use has no voice editor, so there is no Voices panel.")

SAVE_AS_HELP = (N_("Name for this layout. The workspace keeps it as saved: moving or closing panels later "
                   "changes only the window until you choose Update workspace."))


# What each panel is for, by its dock's object name.
PANEL_TIPS = {
    "dock_transcript": N_("Write the text and give it characters."),
    "dock_settings": N_("The voice of the selected clip, its character, or the defaults."),
    "dock_fx": N_("The FX of the selected clip, its character, or the defaults."),
    "dock_lexicon": N_("Rules for how words are read aloud."),
    "dock_outline": N_("The subprojects or markers, with their length and status."),
    "dock_proof": N_("Compare the generated audio with the text."),
    "dock_queue": N_("What is waiting to be generated."),
    "dock_output": N_("The files you exported."),
    "dock_voices": N_("Mix voices or add References, depending on the engine."),
    "dock_video": N_("The reference video."),
    "dock_timeline": N_("The clips in time, on tracks."),
    "dock_transport": N_("Play, record and generate."),
}


def shown_name(name: str) -> str:
    """A built-in's name in the window's language; a custom one as typed."""
    return tr(name) if name in WORKSPACE_NAMES else name


class WorkspaceMenuMixin:
    # -- construction --------------------------------------------------------

    def _build_workspace_menu(self, bar) -> None:
        self.workspace_menu = bar.addMenu(tr("&Workspace"))
        self.workspace_menu.setToolTipsVisible(True)
        self.workspace_menu.aboutToShow.connect(self._sync_workspace_actions)
        self.workspace_group = QActionGroup(self)
        self.workspace_group.setExclusive(True)
        self.workspace_actions: dict = {}
        for name in WORKSPACE_NAMES:
            self.workspace_actions[name] = self._workspace_action(name)

        self.save_workspace_as_action = self._action(tr("Save workspace as..."), self.save_workspace_as)
        self.save_workspace_as_action.setToolTip(tr("Keep the current layout as a new named workspace."))
        self.update_workspace_action = self._action(tr("Update workspace"), self.update_workspace)
        self.update_workspace_action.setToolTip(tr("Save the current layout into the active workspace."))
        self.revert_workspace_action = self._action(tr("Revert to saved layout"), self.revert_workspace)
        self.revert_workspace_action.setToolTip(tr("Put the panels back the way the active workspace was saved."))
        self.rename_workspace_action = self._action(tr("Rename workspace..."), self.rename_workspace)
        self.rename_workspace_action.setToolTip(tr("Give a saved workspace another name."))
        self.delete_workspace_action = self._action(tr("Delete workspace..."), self.delete_workspace)
        self.delete_workspace_action.setToolTip(tr("Remove a saved workspace and its layout."))
        self.panels_menu = QMenu(tr("&Panels"), self)
        self.panels_menu.setToolTipsVisible(True)
        self.panels_menu.aboutToShow.connect(self._rebuild_panels_menu)
        self.reset_layout_action = self._action(tr("Restore default layout..."), self.reset_workspace)
        self.reset_layout_action.setToolTip(tr("Arrange the panels the way this workspace starts. A saved workspace "
                                               "keeps its saved layout until you choose Update workspace."))

        self._layout_timer = QTimer(self)
        self._layout_timer.setSingleShot(True)
        self._layout_timer.setInterval(LAYOUT_SAVE_DEBOUNCE_MS)
        self._layout_timer.timeout.connect(self.save_layout)
        self._layout_ready = False
        # Plan 47 pass 3, grill PG32: tabs on top for every dock area, native Qt.
        self.setTabPosition(Qt.DockWidgetArea.AllDockWidgetAreas, QTabWidget.TabPosition.North)
        self._tab_tune_timer = QTimer(self)
        self._tab_tune_timer.setSingleShot(True)
        self._tab_tune_timer.setInterval(0)
        self._tab_tune_timer.timeout.connect(self.tune_dock_tab_bars)

    def tune_dock_tab_bars(self) -> None:
        """Makes every dock tab readable: no elision (the bar scrolls with its
        arrow buttons when the names don't fit), and the full dock title as
        each tab's tooltip. Qt creates these bars itself and may rebuild them
        after a restore or a re-dock, so this runs again on `ChildAdded` and
        `LayoutRequest` and only writes what differs (a write that changes
        nothing would post another layout request)."""
        self.setTabPosition(Qt.DockWidgetArea.AllDockWidgetAreas, QTabWidget.TabPosition.North)
        for bar in self.findChildren(QTabBar, options=Qt.FindChildOption.FindDirectChildrenOnly):
            if bar.elideMode() != Qt.TextElideMode.ElideNone:
                bar.setElideMode(Qt.TextElideMode.ElideNone)
            if not bar.usesScrollButtons():
                bar.setUsesScrollButtons(True)
            for button in bar.findChildren(QToolButton):
                name = {Qt.ArrowType.LeftArrow: tr("Scroll tabs left"),
                        Qt.ArrowType.RightArrow: tr("Scroll tabs right")}.get(button.arrowType())
                if name and button.toolTip() != name:
                    button.setToolTip(name)
                    button.setAccessibleName(name)
            for i in range(bar.count()):
                # The tab text carries the mnemonic escape ("Voice && Clip");
                # a tooltip shows the name as it reads.
                full = bar.tabText(i).replace("&&", "&")
                if bar.tabToolTip(i) != full:
                    bar.setTabToolTip(i, full)

    def _workspace_action(self, name: str) -> QAction:
        action = QAction(menu_label(shown_name(name)), self)
        action.setToolTip(tr("Arrange the panels the way this workspace keeps them."))
        action.setCheckable(True)
        action.triggered.connect(lambda checked=False, n=name: self.activate_workspace(n))
        self.workspace_group.addAction(action)
        return action

    def _start_workspaces(self) -> None:
        """After the docks exist: restores the layout, watches every dock
        for layout edits, and starts saving them."""
        self.workspaces = WorkspaceManager(self, self.settings)
        self.workspaces.restore_on_launch()
        for dock in self._all_docks():
            self._watch_dock_layout(dock)
        self.installEventFilter(self)
        self.tune_dock_tab_bars()
        app = QGuiApplication.instance()
        if app is not None:
            app.screenRemoved.connect(self._on_screen_removed)
        self._sync_workspace_actions()
        self._launch_workspace_notice = self._take_workspace_notice()
        self._layout_ready = True

    def _take_workspace_notice(self) -> str:
        notices, self.workspaces.notices = self.workspaces.notices, []
        for notice in notices:
            print(f"Workspace: {notice}")
        return " ".join(notices)

    def _show_workspace_notice(self, text: str, box: bool = True) -> None:
        if not text:
            return
        self.set_status(text, "warning")
        if box:
            QMessageBox.warning(self, tr("Workspace"), text)

    def _show_launch_workspace_notice(self) -> None:
        text, self._launch_workspace_notice = getattr(self, "_launch_workspace_notice", ""), ""
        if text and not getattr(self, "_closed", False):
            self._show_workspace_notice(text)

    # -- the menu ------------------------------------------------------------

    def _sync_workspace_actions(self) -> None:
        """Rebuilds the Workspace menu: built-ins, custom workspaces (the
        active one marked "modified" when its layout differs from its
        snapshot), the commands, Panels and Restore default layout."""
        manager = getattr(self, "workspaces", None)
        if manager is None:
            return
        customs = manager.custom_names()
        for name in [n for n in self.workspace_actions if n not in WORKSPACE_NAMES and n not in customs]:
            action = self.workspace_actions.pop(name)
            self.workspace_group.removeAction(action)
            action.deleteLater()
        for name in customs:
            if name not in self.workspace_actions:
                self.workspace_actions[name] = self._workspace_action(name)
        modified = manager.is_modified()
        for name, action in self.workspace_actions.items():
            active = name == manager.active
            action.setChecked(active)
            action.setText(menu_label(shown_name(name)) + (tr(" (modified)") if active and modified else ""))

        custom_active = manager.is_custom()
        self.update_workspace_action.setEnabled(custom_active and modified)
        self.revert_workspace_action.setEnabled(custom_active and modified)
        self.rename_workspace_action.setEnabled(bool(customs))
        self.delete_workspace_action.setEnabled(bool(customs))

        menu = self.workspace_menu
        menu.clear()
        for name in (*WORKSPACE_NAMES, *customs):
            menu.addAction(self.workspace_actions[name])
        menu.addSeparator()
        for action in (self.save_workspace_as_action, self.update_workspace_action, self.revert_workspace_action,
                       self.rename_workspace_action, self.delete_workspace_action):
            menu.addAction(action)
        menu.addSeparator()
        menu.addMenu(self.panels_menu)
        menu.addAction(self.reset_layout_action)
        self._rebuild_panels_menu()

    def panel_docks(self) -> list:
        """The docks in Panels order. The Voices slot is whichever voice
        editor exists (both share `dock_voices`), or None when the engine
        has none."""
        voices = [dock for dock in (self.mixing_dock, self.voice_clone_dock) if dock is not None]
        before = [self.transcript_dock, self.settings_dock, self.fx_dock, self.lexicon_dock, self.outline_dock,
                  self.proof_dock, getattr(self, "queue_dock", None), self.output_dock]
        after = [self.video_dock, self.timeline_dock, self.transport_dock]
        return [*(d for d in before if d is not None), *(voices or [None]), *(d for d in after if d is not None)]

    def _rebuild_panels_menu(self) -> None:
        """Panels lists each dock's own `toggleViewAction()`, so a voice
        editor replaced by an engine change never leaves a stale entry: the
        menu is rebuilt from the docks there are now."""
        menu = getattr(self, "panels_menu", None)
        if menu is None or self.transport_dock is None:
            return
        menu.clear()
        for dock in self.panel_docks():
            if dock is None:
                placeholder = menu.addAction(tr("Voices"))
                placeholder.setEnabled(False)
                placeholder.setToolTip(tr(NO_VOICE_EDITOR_TIP))
                placeholder.setStatusTip(tr(NO_VOICE_EDITOR_TIP))
                continue
            toggle = dock.toggleViewAction()
            toggle.setToolTip(tr(PANEL_TIPS[dock.objectName()]) if dock.objectName() in PANEL_TIPS
                              else tr("Show or hide this panel."))
            menu.addAction(toggle)

    # -- layout edits --------------------------------------------------------

    def _watch_dock_layout(self, dock) -> None:
        if getattr(dock, "_layout_watched", False):
            return
        dock._layout_watched = True
        dock.dockLocationChanged.connect(self._on_layout_edited)
        dock.topLevelChanged.connect(self._on_layout_edited)
        # `visibilityChanged` also fires when a tab is raised; only a close
        # or a reopen (the dock's own hidden flag) is a layout edit. The
        # toggle action changes state without emitting `toggled`.
        dock._layout_hidden = dock.isHidden()
        dock.visibilityChanged.connect(lambda _visible, d=dock: self._on_dock_visibility(d))

    def _on_dock_visibility(self, dock) -> None:
        hidden = dock.isHidden()
        if hidden != getattr(dock, "_layout_hidden", None):
            dock._layout_hidden = hidden
            self._on_layout_edited()

    def _on_layout_edited(self, *_args) -> None:
        if not getattr(self, "_layout_ready", False) or getattr(self, "_closed", False):
            return
        if self.workspaces._loading:
            return
        self._layout_timer.start()

    def eventFilter(self, watched, event):  # noqa: N802 (Qt override)
        # A separator drag is the main window's own mouse release.
        if watched is self and event.type() == QEvent.Type.MouseButtonRelease:
            self._on_layout_edited()
        elif watched is self and event.type() in (QEvent.Type.ChildAdded, QEvent.Type.LayoutRequest):
            if getattr(self, "_layout_ready", False):
                self._tab_tune_timer.start()
        return super().eventFilter(watched, event)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        if getattr(self, "_layout_ready", False):
            self.workspaces.settle_later()
            self._on_layout_edited()

    def _on_screen_removed(self, _screen) -> None:
        if getattr(self, "_layout_ready", False) and not getattr(self, "_closed", False):
            self.workspaces.recover_offscreen_docks()

    def save_layout(self) -> None:
        """Captures the layout and writes `config_qt.json`. Layout only: no
        project autosave and no pending "*" in the title."""
        self._layout_timer.stop()
        if getattr(self, "_closed", False) or not hasattr(self, "workspaces"):
            return
        self.workspaces.capture()
        self._write_settings_file()
        self._sync_workspace_actions()

    # -- commands ------------------------------------------------------------

    def activate_workspace(self, name: str) -> None:
        manager = self.workspaces
        if name == manager.active and manager.is_custom(name):
            self.revert_workspace()
            return
        manager.activate(name)
        self._after_workspace_change()

    def _after_workspace_change(self) -> None:
        self._show_workspace_notice(self._take_workspace_notice())
        self._sync_workspace_actions()
        self.save_layout()

    def _confirm(self, title: str, text: str) -> bool:
        answer = QMessageBox.question(self, title, text,
                                      QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                      QMessageBox.StandardButton.No)
        return answer == QMessageBox.StandardButton.Yes

    def save_workspace_as(self) -> None:
        manager = self.workspaces
        suggested = manager.active if manager.is_custom() else ""
        name, ok = QInputDialog.getText(self, tr("Save Workspace As"), tr(SAVE_AS_HELP), text=suggested)
        if not ok:
            return
        try:
            manager.save_as(name)
        except WorkspaceExistsError as clash:
            if not self._confirm(tr("Replace Workspace"),
                                 tr("Replace the saved layout of {name} with the current one?", name=clash.name)):
                return
            manager.save_as(name, replace=True)
        except WorkspaceNameError as error:
            QMessageBox.warning(self, tr("Save Workspace As"), str(error))
            return
        self._after_workspace_change()

    def update_workspace(self) -> None:
        if self.workspaces.update():
            self._after_workspace_change()

    def revert_workspace(self) -> None:
        manager = self.workspaces
        if not manager.is_custom():
            return
        if manager.is_modified() and not self._confirm(
                tr("Revert to Saved Layout"), tr("Discard the changes to the panels and go back to {active} "
                                             "as it was saved?", active=manager.active)):
            self._sync_workspace_actions()
            return
        manager.revert()
        self._after_workspace_change()

    def _pick_custom_workspace(self, title: str) -> str | None:
        """The active custom workspace, else one the user picks."""
        manager = self.workspaces
        if manager.is_custom():
            return manager.active
        names = manager.custom_names()
        if not names:
            return None
        name, ok = QInputDialog.getItem(self, title, tr("Workspace:"), names, 0, False)
        return name if ok and name in names else None

    def rename_workspace(self) -> None:
        old = self._pick_custom_workspace(tr("Rename Workspace"))
        if old is None:
            return
        new, ok = QInputDialog.getText(self, tr("Rename Workspace"), tr("New name for {old}:", old=old), text=old)
        if not ok:
            return
        try:
            self.workspaces.rename(old, new)
        except WorkspaceNameError as error:
            QMessageBox.warning(self, tr("Rename Workspace"), str(error))
            return
        self._sync_workspace_actions()
        self.save_layout()

    def delete_workspace(self) -> None:
        name = self._pick_custom_workspace(tr("Delete Workspace"))
        if name is None:
            return
        active = name == self.workspaces.active
        text = tr("Delete the workspace {name}? Its saved layout can't be brought back.", name=name)
        if active:
            text += tr(" The window switches to {advanced}.", advanced=tr(ADVANCED))
        if not self._confirm(tr("Delete Workspace"), text):
            return
        self.workspaces.delete(name)
        self._after_workspace_change()

    def reset_workspace(self) -> None:
        """Restore default layout. A built-in forgets the edits it
        remembered; a custom workspace's snapshot stays until Update."""
        manager = self.workspaces
        if manager.is_custom():
            if manager.is_modified() and not self._confirm(
                    tr("Restore Default Layout"), tr("Discard the changes to the panels and arrange them the default "
                                                     "way? "
                                                 "The saved layout of {active} stays until you choose "
                                                 "Update workspace.", active=manager.active)):
                return
        elif not self._confirm(tr("Restore Default Layout"),
                               tr("Arrange the panels the way {active} starts? The changes {active} "
                                  "remembered are discarded.", active=tr(manager.active))):
            return
        manager.reset()
        self._after_workspace_change()


__all__ = ["WorkspaceMenuMixin", "ADVANCED", "SIMPLE"]
