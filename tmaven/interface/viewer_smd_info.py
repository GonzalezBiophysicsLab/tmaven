from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QDockWidget
from PyQt5.QtCore import QObject, Qt

class smd_info_viewer(QObject):
	def __init__(self,gui):
		super().__init__()
		self.gui = gui
		self.dock = None

	def toggle(self):
		if self.dock:
			self.remove()
		else:
			self.new()

	def new(self):
		self.viewer = smd_info_widget()
		self.dock = QDockWidget("SMD Info")
		self.gui.data_update.connect(self.update)

		from .stylesheet import ss_qdockwidget
		self.dock.setStyleSheet(ss_qdockwidget)
		self.dock.setWidget(self.viewer)
		self.dock.setFeatures(QDockWidget.AllDockWidgetFeatures)
		self.dock.setAllowedAreas(Qt.TopDockWidgetArea | Qt.BottomDockWidgetArea | Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
		self.gui.addDockWidget(Qt.LeftDockWidgetArea, self.dock)
		self.dock.closeEvent = lambda e: self.toggle()

		self.update()

	def remove(self):
		"""
		Removes the plotter pane from the application.
		"""
		if self.dock:
			# try: self.gui.new_selection_last.disconnect(self.plotter_callback)
			# except: pass
			self.viewer = None
			self.dock.setParent(None)
			self.dock.deleteLater()
			self.dock = None

	def update(self,event=None):
		if self.dock is None:
			return

		if self.gui.maven.data is None:
			self.clear()
		elif self.gui.maven.data.nmol is None:
			self.clear()
		else:
			d = self.gui.maven.data
			s = self.gui.maven.smd
			import textwrap
			self.viewer.label1.setText('({},{},{})'.format(d.nmol,d.ntime,d.ndata))
			q = '{}'.format(s.source_names)
			self.viewer.label2.setText('\n'.join(textwrap.wrap(q,width=40)))
			q = '\n'.join(['{}:{}'.format(k,v) for k,v in s.smd_dict.items()])
			self.viewer.label3.setText('\n'.join(textwrap.wrap(q,width=40)))

	def clear(self):
		self.viewer.label1.setText('')
		self.viewer.label2.setText('')
		self.viewer.label3.setText('')

class smd_info_widget(QWidget):
	def __init__(self):
		super().__init__()
		self.label1 = QLabel('')
		self.label2 = QLabel('')
		self.label3 = QLabel('')

		vbox = QVBoxLayout()
		vbox.addWidget(self.label1)
		vbox.addWidget(self.label2)
		vbox.addWidget(self.label3)
		vbox.addStretch(1)
		self.setLayout(vbox)
