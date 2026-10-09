"""Persistent desktop settings and shared file dialogs."""
import json
import logging
import os
import tempfile
import appdirs
from PyQt5.QtCore import QByteArray, Qt, QTimer
from PyQt5.QtWidgets import QApplication, QFileDialog, QActionGroup, QMessageBox

logger = logging.getLogger(__name__)

class DesktopSession:
    def _config_file_getname(self, fname):
        directory = appdirs.user_config_dir(appname="tmaven", appauthor="python")
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, fname)

    def config_file_read(self, fname):
        try:
            with open(self._config_file_getname(fname)) as stream:
                value = json.load(stream)
            return value if isinstance(value, dict) else {}
        except FileNotFoundError:
            return {}
        except (OSError, ValueError, TypeError):
            logger.warning('Could not read %s; using defaults for unavailable settings', fname)
            return {}

    def config_file_overwrite(self, fname, dictionary):
        temporary = None
        try:
            target = self._config_file_getname(fname)
            with tempfile.NamedTemporaryFile(mode="w", dir=os.path.dirname(target), delete=False) as stream:
                temporary = stream.name
                json.dump(dictionary, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        except (OSError, ValueError, TypeError):
            logger.exception("Could not save desktop settings")
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)

    def config_file_update(self, fname, key, value):
        info = self.config_file_read(fname)
        info[key] = value
        self.config_file_overwrite(fname, info)

    def folder(self, category):
        info = self.config_file_read('recent_files.json')
        folders = info.get('folders', {})
        candidate = folders.get(category, info.get('lwd')) if isinstance(folders, dict) else None
        if isinstance(candidate, str) and os.path.isdir(candidate):
            return candidate
        return os.path.expanduser('~')

    def lwd_load(self):
        self.lwd = self.folder('data')

    def lwd_update(self, lwd):
        if isinstance(lwd, str) and os.path.isdir(lwd):
            self.lwd = lwd
            self.config_file_update('recent_files.json', 'lwd', lwd)

    def choose_file(self, caption, category='data', save=False, multiple=False, name='', filter='', parent=None):
        initial = os.path.join(self.folder(category), name)
        owner = self if parent is None else parent
        if save:
            result = QFileDialog.getSaveFileName(owner, caption, initial, filter)[0]
        elif multiple:
            result = QFileDialog.getOpenFileNames(owner, caption, initial, filter)[0]
        else:
            result = QFileDialog.getOpenFileName(owner, caption, initial, filter)[0]
        selected = result[-1] if multiple and result else result
        if selected:
            info = self.config_file_read('recent_files.json')
            folders = info.get('folders', {})
            if not isinstance(folders, dict):
                folders = {}
            folders[category] = os.path.dirname(selected)
            info['folders'] = folders
            self.config_file_overwrite('recent_files.json', info)
            if category == 'data':
                self.lwd_update(folders[category])
        return result

    def recent_entries(self):
        entries = self.config_file_read('recent_files.json').get('data_files', [])
        if not isinstance(entries, list):
            return []
        return [e for e in entries if isinstance(e, dict) and isinstance(e.get('path'), str)
                and e.get('kind') in ('smd', 'hdf5', 'numpy', 'text', 'spartan') and isinstance(e.get('options'), dict)][:10]

    def remember_file(self, path, kind, **options):
        entry = dict(path=os.path.abspath(path), kind=kind, options=options)
        entries = [e for e in self.recent_entries() if e != entry]
        self.config_file_update('recent_files.json', 'data_files', [entry] + entries[:9])
        self.refresh_recent_menu()

    def refresh_recent_menu(self):
        self.menu_recent.clear()
        for entry in self.recent_entries():
            suffix = str(entry['options'].get('dataset', entry['options'].get('group', '')))
            label = entry['path'] + (' : ' + suffix if suffix else '') + ' (' + entry['kind'] + ')'
            action = self.menu_recent.addAction(label, lambda checked=False, e=entry: self.open_recent(e))
            action.setEnabled(os.path.isfile(entry['path']))
        if not self.menu_recent.actions():
            self.menu_recent.addAction('No recent files').setEnabled(False)
        self.menu_recent.addSeparator()
        self.menu_recent.addAction('Clear Recent Files', self.clear_recent)

    def clear_recent(self):
        self.config_file_update('recent_files.json', 'data_files', [])
        self.refresh_recent_menu()

    def open_recent(self, entry):
        path, kind, options = entry['path'], entry['kind'], entry['options']
        try:
            io = self.maven.io
            before = self.maven.data.nmol
            if kind == 'smd':
                io.load_smdtmaven_hdf5(path, options['group'])
            elif kind == 'hdf5':
                io.load_raw_hdf5(path, options['dataset'])
            elif kind == 'numpy':
                io.load_raw_numpy(path)
            elif kind == 'spartan':
                io.load_spartan(path)
            else:
                for key, value in options.get('prefs', {}).items():
                    if key.startswith('io.'):
                        self.maven.prefs.__setitem__(key, value, quiet=True)
                io.load_raw_txt(path, options['skiprows'], options['delimiter'])
            if self.maven.data.nmol <= before:
                raise ValueError('No data was loaded. Check the file and import settings.')
            self.lwd_update(os.path.dirname(path))
            self.remember_file(path, kind, **options)
            self.maven.emit_data_update()
        except Exception as error:
            QMessageBox.warning(self, 'Could not open recent file', str(error))

    def build_desktop_menus(self):
        self.menu_recent = self.menu_file.addMenu('Recent Files')
        self.menu_file.removeAction(self.menu_recent.menuAction())
        self.menu_file.insertMenu(self.action_clear_data, self.menu_recent)
        self.menu_recent.aboutToShow.connect(self.refresh_recent_menu)
        self.refresh_recent_menu()
        resets = self.menu_view.addMenu('Reset')
        resets.addAction('Window Layout', self.reset_layout)
        resets.addAction('Appearance', lambda: self.apply_theme('system'))
        resets.addAction('Clear History', self.clear_history)
        self.theme_actions = {}
        self.theme_group = QActionGroup(self)
        self.theme_group.setExclusive(True)
        for mode in ('system', 'light', 'dark'):
            action = self.menu_theme.addAction(mode.title())
            action.setCheckable(True)
            action.triggered.connect(lambda checked=False, m=mode: self.apply_theme(m))
            self.theme_group.addAction(action)
            self.theme_actions[mode] = action
        self.app.paletteChanged.connect(self.system_palette_changed)

    def system_palette_changed(self, palette):
        if not self._applying_theme:
            self._system_palette = palette
            if self.theme_mode == 'system':
                self.apply_theme('system', persist=False)

    def apply_theme(self, mode, persist=True):
        if mode not in ('system', 'light', 'dark'):
            mode = 'system'
        self.theme_mode = mode
        palette = self._system_palette if mode == 'system' else (self.dark_theme() if mode == 'dark' else self.light_theme())
        from PyQt5.QtGui import QPalette
        self.lightdark_mode = 'dark' if palette.color(QPalette.Window).lightness() < 128 else 'light'
        self._applying_theme = True
        try:
            self._change_theme(palette)
            self.maven.prefs['plot.bg_color'] = palette.color(QPalette.Window).name()
            for window in QApplication.topLevelWidgets():
                window.setPalette(palette)
                if hasattr(window, "apply_plot_theme"):
                    window.apply_plot_theme(palette)
            self.plot_container.plot.redrawplot()
        finally:
            self._applying_theme = False
        for key, action in self.theme_actions.items():
            action.setChecked(key == mode)
        if persist:
            self.config_file_update('tmaven_config.json', 'lightdark_mode', mode)

    def reset_layout(self):
        self.showNormal()
        self.size_window()
        self.restoreState(self._default_layout)
        for viewer, state in zip((self.molecules_viewer.viewer, self.preferences_viewer.viewer.proxy_view), self._default_headers):
            viewer.horizontalHeader().restoreState(state)
        self.session_save()

    def clear_history(self):
        self.scripts_toolbar.clear_actions(update=False)
        self.config_file_overwrite('recent_files.json', {})
        self.lwd_load()
        self.refresh_recent_menu()

    def session_default(self):
        # Safe startup uses defaults without erasing the user's saved settings.
        self.plot_container.change_mode('smFRET')
        self.size_window()
        self.apply_theme('system', persist=False)
        self.show()

    def size_window(self, x=None, y=None, w=None, h=None):
        screen = self.screen() or self.app.primaryScreen()
        if isinstance(x, int) and isinstance(y, int):
            from PyQt5.QtCore import QPoint
            for candidate in self.app.screens():
                if candidate.availableGeometry().contains(QPoint(x, y)):
                    screen = candidate
                    break
        bounds = screen.availableGeometry()
        valid = lambda value: isinstance(value, int) and not isinstance(value, bool)
        w = max(320, min(w, bounds.width())) if valid(w) and w > 0 else int(bounds.width() * .8)
        h = max(240, min(h, bounds.height())) if valid(h) and h > 0 else int(bounds.height() * .8)
        x = x if valid(x) else bounds.x() + (bounds.width()-w)//2
        y = y if valid(y) else bounds.y() + (bounds.height()-h)//2
        from PyQt5.QtCore import QRect
        rect = QRect(x, y, w, h)
        screens = [s.availableGeometry() for s in self.app.screens()]
        if not any(b.contains(rect) for b in screens):
            x = max(bounds.left(), min(x, bounds.right()-w+1))
            y = max(bounds.top(), min(y, bounds.bottom()-h+1))
        self.resize(w, h)
        self.move(x, y)

    def dock_sizes(self):
        sizes = {}
        for viewer in (self.molecules_viewer, self.preferences_viewer):
            dock = viewer.dock
            if dock.isFloating() or dock.isHidden():
                continue
            # In a tab group only the selected dock has current layout dimensions.
            group = [dock] + self.tabifiedDockWidgets(dock)
            active = next((other for other in group
                           if other.objectName() == self._active_dock and not other.isHidden()), dock)
            for other in group:
                sizes[other.objectName()] = dict(w=active.width(), h=active.height())
        return sizes

    def restore_dock_sizes(self, sizes):
        if not isinstance(sizes, dict):
            return
        for viewer in (self.molecules_viewer, self.preferences_viewer):
            dock = viewer.dock
            value = sizes.get(dock.objectName())
            if dock.isFloating() or dock.isHidden() or not isinstance(value, dict):
                continue
            horizontal = self.dockWidgetArea(dock) in (Qt.LeftDockWidgetArea, Qt.RightDockWidgetArea)
            dimension = value.get('w' if horizontal else 'h')
            if isinstance(dimension, int) and not isinstance(dimension, bool) and dimension > 0:
                limit = self.width() if horizontal else self.height()
                self.resizeDocks([dock], [min(dimension, limit)], Qt.Horizontal if horizontal else Qt.Vertical)

    def session_save(self):
        rect = self.normalGeometry() if self.isMaximized() or self.isFullScreen() else self.geometry()
        session = self.config_file_read('tmaven_config.json')
        session.update(version=1, window=dict(x=rect.x(), y=rect.y(), w=rect.width(), h=rect.height()),
                       maximized=self.isMaximized(), layout=bytes(self.saveState().toHex()).decode(),
                       plot_mode=self.plot_container.plot.plot_mode, lightdark_mode=self.theme_mode,
                       headers=[bytes(v.horizontalHeader().saveState().toHex()).decode() for v in
                                (self.molecules_viewer.viewer, self.preferences_viewer.viewer.proxy_view)],
                       active_dock=self._active_dock, dock_sizes=self.dock_sizes())
        self.config_file_overwrite('tmaven_config.json', session)

    def session_restore(self):
        session = self.config_file_read('tmaven_config.json')
        mode = session.get('plot_mode', 'smFRET')
        if mode not in ('smFRET', 'ND Raw', 'ND Relative', 'Normalized'):
            mode = 'smFRET'
        self.plot_container.change_mode(mode)
        window = session.get('window', {})
        if not isinstance(window, dict):
            window = {}
        self.size_window(**{key: window.get(key) for key in ('x', 'y', 'w', 'h')})
        def restore(value, callback):
            if isinstance(value, str):
                try:
                    if value and len(value) % 2 == 0 and all(c in '0123456789abcdefABCDEF' for c in value):
                        callback(QByteArray.fromHex(value.encode('ascii')))
                except (ValueError, UnicodeError):
                    pass
        restore(session.get('layout'), self.restoreState)
        headers = session.get('headers', [])
        if isinstance(headers, list):
            for viewer, value in zip((self.molecules_viewer.viewer, self.preferences_viewer.viewer.proxy_view), headers):
                restore(value, viewer.horizontalHeader().restoreState)
        for dock in (self.molecules_viewer.dock, self.preferences_viewer.dock):
            if dock.isFloating() and not any(screen.availableGeometry().contains(dock.frameGeometry()) for screen in self.app.screens()):
                bounds = self.app.primaryScreen().availableGeometry()
                dock.resize(min(dock.width(), bounds.width()), min(dock.height(), bounds.height()))
                dock.move(bounds.topLeft())
        self.apply_theme(session.get('lightdark_mode', 'system'), persist=False)
        if session.get('maximized') is True:
            self.showMaximized()
        else:
            self.show()
        dock = self.preferences_viewer.dock if session.get('active_dock') == 'preferences' else self.molecules_viewer.dock
        if not dock.isHidden():
            dock.raise_()
        self.plot_container.plot.redrawplot()
        QTimer.singleShot(0, lambda: self.restore_dock_sizes(session.get('dock_sizes')))
