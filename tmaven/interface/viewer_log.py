"""Docked, selectable view of the application log."""
from PyQt5.QtCore import QObject, Qt, QTimer
from PyQt5.QtGui import QFontDatabase, QTextCursor
from PyQt5.QtWidgets import QDockWidget, QPlainTextEdit


class log_viewer(QObject):
	def __init__(self, gui):
		super().__init__(parent=gui)
		self.gui = gui
		self.viewer = QPlainTextEdit(parent=gui)
		self.viewer.setReadOnly(True)
		self.viewer.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
		self._text = ''
		self.dock = QDockWidget('Log', gui)
		self.dock.setObjectName('log')
		self.dock.setWidget(self.viewer)
		self.dock.setFeatures(QDockWidget.AllDockWidgetFeatures)
		self.dock.setAllowedAreas(Qt.AllDockWidgetAreas)
		gui.addDockWidget(Qt.LeftDockWidgetArea, self.dock)
		self.dock.hide()
		self.timer = QTimer(self)
		self.timer.setInterval(1000)
		self.timer.timeout.connect(self.refresh)
		self.dock.visibilityChanged.connect(self._visibility_changed)

	def _visibility_changed(self, visible):
		if visible:
			self.refresh()
			self.timer.start()
		else:
			self.timer.stop()

	def refresh(self):
		text = self.gui.maven.get_log()
		if text == self._text:
			return
		bar = self.viewer.verticalScrollBar()
		follow = bar.value() == bar.maximum() and not self.viewer.textCursor().hasSelection()
		position = bar.value()
		if text.startswith(self._text):
			cursor = QTextCursor(self.viewer.document())
			cursor.movePosition(QTextCursor.End)
			cursor.insertText(text[len(self._text):])
		else:
			self.viewer.setPlainText(text)
		self._text = text
		bar.setValue(bar.maximum() if follow else position)

	def toggle(self):
		self.refresh()
		self.dock.show()
		self.dock.raise_()

	def update_theme(self, palette):
		self.dock.setPalette(palette)
		self.viewer.setPalette(palette)
		self.viewer.verticalScrollBar().setPalette(palette)
		self.viewer.horizontalScrollBar().setPalette(palette)
