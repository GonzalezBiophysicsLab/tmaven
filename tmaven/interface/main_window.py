### borrows heavily from Mu-editor. Credit to them
import logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QMainWindow,QDesktopWidget,QLabel,QMessageBox,QShortcut,QPushButton,QMenu,QAction,QStyleFactory,QSlider, QSizePolicy, QTreeWidget, QTreeWidgetItem, QDockWidget, QLabel, QHBoxLayout
from PyQt5.QtGui import QIcon, QKeySequence
from PyQt5.QtCore import Qt, pyqtSignal
import matplotlib
matplotlib.use('Qt5Agg') ## forces Qt5 early on...

import numpy as np
from . import stylesheet
from .resources import load_pixmap, load_icon

class_keys = {Qt.Key_1:1,Qt.Key_2:2,Qt.Key_3:3,Qt.Key_4:4,Qt.Key_5:5,Qt.Key_6:6,Qt.Key_7:7,Qt.Key_8:8,Qt.Key_9:9,Qt.Key_0:0}

class main_window(QMainWindow):
	new_selection_all = pyqtSignal(np.ndarray)
	new_selection_last = pyqtSignal(np.ndarray)
	# update_slider_signal = pyqtSignal(int)
	data_update = pyqtSignal()
	pref_edited = pyqtSignal()

	def __init__(self,maven,app):
		super().__init__()
		self.app = app
		self.maven = maven
		self.index = 0
		self.desired_index = 0
		self.timer = None
		self.initialize_widgets()

	def initialize_widgets(self):
		## window style
		self.setWindowTitle('tMAVEN')
		self.setStyle(QStyleFactory.create('Fusion')) ## WOW THIS THROWS RANDOM SEGFAULTS WHEN QUITTING
		self.setStyleSheet(stylesheet.ui_stylesheet)
		self.setWindowIcon(load_icon('logo.png'))
		self.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
		self.closeEvent = self.quit

		## widgets

		#### molecule number label
		self.label_molnum = QLabel('0/0')
		from PyQt5.Qt import QFont,QFontDatabase
		monospace = QFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
		self.label_molnum.setFont(monospace)

		#### molecule slider
		self.slider_select = QSlider(Qt.Horizontal)
		sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
		self.slider_select.setSizePolicy(sizePolicy)
		from .stylesheet import slider_style_sheet
		self.slider_select.setStyleSheet(slider_style_sheet)
		self.slider_select.setRange(0,0)
		self.slider_select.setValue(0)
		self.slider_select.setTracking(True)
		# self.slider_select.keyPressEvent = self.slider_keypress
		# self.update_slider_signal.connect(self.update_slider_slot)
		self.slider_select.valueChanged.connect(self.update_selection_from_slider)
		# self.molecule_group_viewer.new_model.connect(lambda: self.slider_select.setRange(0,self.molecule_group_viewer.model().rowCount(0)-1))

		from ..trace_plot.plot_container import plot_container
		self.plot_container = plot_container(self)

		from .viewer_molecules import molecules_viewer
		self.molecules_viewer = molecules_viewer(self)

		#### preferences
		from .viewer_prefs import preferences_viewer
		self.preferences_viewer = preferences_viewer(self)

		#### smd info
		from .viewer_smd_info import smd_info_viewer
		self.smd_info_viewer = smd_info_viewer(self)
		# self.smd_info_viewer.toggle()

		## hdf5 explorer
		from .hdf5_view.hdf5_view import hdf5_view_container
		self.hdf5_viewer = hdf5_view_container(self)
		# self.hdf5_viewer.launch()

		## menu
		self.menubar = self.menuBar()
		self.menubar.setNativeMenuBar(False)
		self.menubar.setStyleSheet(stylesheet.ss_qmenubar)
		self.reset_menus()

		#### total layout
		from PyQt5.QtWidgets import QVBoxLayout,QWidget
		self._qw = QWidget()
		self.vbox = QVBoxLayout()
		# self.vbox.addWidget(self.molecule_group_viewer)
		qw = QWidget()
		hbox = QHBoxLayout()
		hbox.addWidget(self.slider_select)
		hbox.addWidget(self.label_molnum)
		qw.setLayout(hbox)
		self.vbox.addStretch(1)
		self.vbox.addWidget(self.plot_container)
		self.vbox.addWidget(qw)
		self._qw.setSizePolicy(QSizePolicy.MinimumExpanding,QSizePolicy.MinimumExpanding)
		self._qw.setLayout(self.vbox)
		self.setCentralWidget(self._qw)

		# self.preferences_viewer.toggle()
		# self.molecules_viewer.toggle()
		# self.tabifyDockWidget(self.preferences_viewer.dock,self.molecules_viewer.dock)

		## connections
		self.maven.emit_data_update = lambda : self.data_update.emit()
		self.maven.prefs.emit_changed = lambda : self.pref_edited.emit()
		self.data_update.connect(self.proc_data_update)
		self.maven.emit_data_update()

		self.show()
		self.plot_container.plot.redrawplot()

	def reset_menus(self):
		from . import ui_io, ui_scripts, ui_cull, ui_corrections, ui_selection, ui_photobleaching, ui_trace_filter
		from .modeler import ui_modeler

		logger.info('Building Menus')
		self.menubar.clear()
		self.menu_file  = QMenu('File',self)
		self.menu_tools = QMenu('Tools',self)
		self.menu_other = QMenu('Other',self)
		self.menu_view = QMenu('View',self)
		self.menu_prefs = QMenu('Preferences',self)
		self.menu_plots = QMenu('Plots',self)

		self.menu_load,self.menu_save = ui_io.build_menu(self)
		self.menu_scripts = self.menu_other.addMenu('Script Runner')
		self.trace_filter = ui_trace_filter.container_trace_filter(self)
		self.menu_cull = ui_cull.build_menu(self)
		self.menu_corrections,self.menu_correction_filters = ui_corrections.build_menu(self)
		self.menu_selection,self.menu_on,self.menu_off = ui_selection.build_menu(self)
		self.menu_photobleaching = ui_photobleaching.build_menu(self)
		self.menu_modeler = ui_modeler.build_menu(self)


		self.menu_file.addMenu(self.menu_load)
		self.menu_file.addMenu(self.menu_save)
		self.menu_file.addAction('Clear Data',self.clear_data)
		self.menu_file.addMenu(self.menu_prefs)
		self.menu_prefs.addAction('Load Preferences',self.preferences_viewer.load)
		self.menu_prefs.addAction('Save Preferences',self.preferences_viewer.save)
		self.menu_file.addAction('Exit',self.quit,'Ctrl+Q')

		self.menu_view.addAction('Reset GUI Size',self.size_window)
		self.menu_view.addAction('SMD Info',self.smd_info_viewer.toggle)
		self.menu_view.addAction('Molecule Table',self.molecules_viewer.toggle,'Ctrl+T')
		self.menu_view.addAction('Preferences',self.preferences_viewer.toggle,'Ctrl+P')
		self.menu_other.addAction(self.hdf5_viewer.action)

		self.menu_scripts.addAction('Run Script',lambda:ui_scripts.run(self),'Ctrl+R')
		self.menu_scripts.addAction('Run Input',lambda:ui_scripts.input_run(self))

		self.menu_tools.addMenu(self.menu_selection)
		self.menu_tools.addMenu(self.menu_cull)
		self.menu_tools.addMenu(self.menu_corrections)
		self.menu_corrections.addMenu(self.menu_correction_filters)
		self.menu_tools.addMenu(self.menu_photobleaching)
		self.menu_tools.addAction('Filter Traces',self.trace_filter.launch,shortcut='Ctrl+F')

		from .ui_analysisplots import popplot_container
		self.menu_plots.addAction('FRET Hist 1D',lambda : popplot_container(self,self.maven.plots.fret_hist1d))
		self.menu_plots.addAction('FRET Hist 2D',lambda : popplot_container(self,self.maven.plots.fret_hist2d))
		self.menu_plots.addAction('FRET TDP',lambda : popplot_container(self,self.maven.plots.fret_tdp))
		self.menu_plots.addAction('vb Model States',lambda : popplot_container(self,self.maven.plots.model_vbstates))

		for menu in [self.menu_file,self.menu_tools,self.menu_other,self.menu_view,self.menu_prefs,self.menu_scripts,self.menu_plots]:
			menu.setStyleSheet(stylesheet.ss_qmenu)

		self.menubar.addMenu(self.menu_file)
		self.menubar.addMenu(self.menu_view)
		self.menubar.addMenu(self.menu_tools)
		self.menubar.addMenu(self.menu_modeler)
		self.menubar.addMenu(self.menu_plots)
		self.menubar.addMenu(self.menu_other)


	def clear_data(self):
		from PyQt5.QtWidgets import QMessageBox
		reply = QMessageBox.question(self,"Clear Data?","Are you sure you want to remove all the current data?",QMessageBox.Yes | QMessageBox.No)
		if reply == QMessageBox.Yes:
			self.maven.io.clear_data()

	def keyPressEvent(self,event):
		kk = event.key()
		from PyQt5.QtCore import Qt

		if kk in [Qt.Key_BracketLeft,Qt.Key_BracketRight]:
			if kk == Qt.Key_BracketLeft:
				if self.maven.data.post_list[self.index] > 0:
					self.maven.data.post_list[self.index] -= 1
			else:
				if self.maven.data.post_list[self.index] < self.maven.data.ntime:
					self.maven.data.post_list[self.index] += 1
			self.plot_container.plot.softredraw()
		# elif kk == Qt.Key_Space:
		# 	self.maven.data.flag_ons[self.index] = not self.maven.data.flag_ons[self.index]
		elif kk == Qt.Key_R:
			self.maven.data.pre_list[self.index] = 0
			self.maven.data.post_list[self.index] = self.maven.data.ntime
			self.plot_container.plot.softredraw()
		elif kk == Qt.Key_G:
			try:
				self.plot_container.plot.ax[0,0].grid()
				self.plot_container.plot.ax[1,0].grid()
				self.plot_container.plot.softredraw()
				self.plot_container.plot.update_blits()
			except:
				pass
		elif kk == Qt.Key_P:
			try:
				self.maven.photobleaching.calc_single_photobleach(self.index)
				self.plot_container.plot.softredraw()
			except:
				pass
		elif kk in [Qt.Key_Right, Qt.Key_Down]:
			new_index = self.desired_index + 1
			if new_index >= self.maven.data.nmol:
				new_index = self.maven.data.nmol-1
			self.change_index(new_index)
		elif kk in [Qt.Key_Left, Qt.Key_Up]:
			new_index = self.desired_index - 1
			if new_index < 0:
				new_index = 0
			self.change_index(new_index)
		elif event.key() in class_keys:
			new_class_ind = class_keys[kk]
			self.maven.data.classes[self.index] = new_class_ind
		else:
			super().keyPressEvent(event)
			return
		self.molecules_viewer.update()
		self.plot_container.plot.setFocus()
		return

	def update_selection_from_slider(self):
		self.slider_select.blockSignals(True)
		self.change_index(self.slider_select.value())
		self.slider_select.blockSignals(False)

	def proc_data_update(self):
		self.plot_container.plot.update_data(self.index)
		self.label_molnum.setText('{}/{}'.format(self.index,self.maven.data.nmol-1))
		self.slider_select.blockSignals(True)
		self.slider_select.setValue(self.index)
		self.slider_select.setRange(0,self.maven.data.nmol-1)
		self.slider_select.blockSignals(False)

	def change_index(self,new_index):
		if self.maven.data.nmol == 0:
			return
		self.desired_index = new_index
		if not self.plot_container.plot.flag_drawing:
			self.timer = None
			self.index = self.desired_index
			self.proc_data_update()
		else:
			## This is the trick to make scrolling look smooth
			## If plots are requested to fast, store that last requested plot and start a timer
			## the timer lasts 1/25 sec (e.g. video rate), so our eyes dont distinguish a lag
			## once the timer is up, check to see if it's done plotting the blocking plot
			## then plot the latest requested plot
			## otherwise, start up the timer again
			if self.timer is None:
				self._newtimer()

	def _newtimer(self):
		from PyQt5.QtCore import QTimer
		self.timer = QTimer(singleShot=True)
		self.timer.setInterval(40) ## 25 FPS = 40 msec
		self.timer.timeout.connect(self._timesup)
		self.timer.start()

	def _timesup(self):
		self.timer = None
		if self.plot_container.plot.flag_drawing:
			self._newtimer()
		else:
			self.index = self.desired_index
			self.proc_data_update()


	def screen_size(self):
		"""
		Returns an (width, height) tuple with the screen geometry.
		"""

		screen = QDesktopWidget().screenGeometry()
		return screen.width(), screen.height()

	def size_window(self, x=None, y=None, w=None, h=None):
		"""
		Makes the editor 80% of the width*height of the screen and centres it
		when none of x, y, w and h is passed in; otherwise uses the passed in
		values to position and size the editor window.
		"""
		screen_width, screen_height = self.screen_size()
		w = int(screen_width * 0.8) if w is None else w
		h = int(screen_height * 0.8) if h is None else h
		self.resize(w,h)
		x = (screen_width - w) // 2 if x is None else x
		y = (screen_height - h) // 2 if y is None else y
		logger.info('trying to resize x{} y{} w{} h{}'.format(x,y,w,h))
		self.move(x,y)

	def save_session(self,session):
		import os
		import json
		import appdirs
		config_dir = appdirs.user_config_dir(appname="tmaven", appauthor="python")
		config_file = os.path.join(config_dir, "tmaven_config.json")
		if not os.path.exists(config_dir):
			os.makedirs(config_dir)

		with open(config_file, 'w') as f:
			json.dump(session, f)

		session_str =  ",".join(f"{key}:{value}" for key, value in session.items())
		logger.info('Saved session {} to {}'.format(session_str,config_file))

	def restore_session(self):
		import os
		import json
		import appdirs
		config_dir = appdirs.user_config_dir(appname="tmaven", appauthor="python")
		config_file = os.path.join(config_dir, "tmaven_config.json")

		if not os.path.isfile(config_file):
			logger.info('Cannot find a previous session to load')
			return

		with open(config_file,'r') as f:
			session = json.load(f)
		session_str =  ",".join(f"{key}:{value}" for key, value in session.items())
		logger.info('Loaded session data from {}.\nSession data - {}'.format(config_file,session_str))
		x = session['window']['x'] if 'x' in session['window'] else None
		y = session['window']['y'] if 'y' in session['window'] else None
		w = session['window']['w'] if 'w' in session['window'] else None
		h = session['window']['h'] if 'h' in session['window'] else None
		self.size_window(x,y,w,h)

	def quit(self, event=None):
		"""
		Exit the application.
		"""
		from PyQt5.QtWidgets import QMessageBox
		message_box = QMessageBox(self)
		message_box.setText("Are you sure you want to quit?")
		message_box.setWindowTitle("tMAVEN")
		message_box.setStandardButtons(message_box.Cancel | message_box.Ok)
		message_box.setDefaultButton(message_box.Cancel)
		message_box.setStyleSheet('''background-color:white;font-size: 12px;''')

		result = message_box.exec()

		if result == QMessageBox.Cancel:
			if not event is None:
				event.ignore()
			return

		session = {
			"window": {
				"x": self.x(),
				"y": self.y(),
				"w": self.width(),
				"h": self.height(),
			}#,
			# "window.group": {
			# 	"on": self.gui.molecule_group_viewer.menu_tabs.actions()[0].isChecked(),
			# 	"id": self.gui.molecule_group_viewer.menu_tabs.actions()[2].isChecked(),
			# 	"class": self.gui.molecule_group_viewer.menu_tabs.actions()[3].isChecked(),
			# 	"onoff": self.gui.molecule_group_viewer.menu_tabs.actions()[4].isChecked(),
			# 	"source": self.gui.molecule_group_viewer.menu_tabs.actions()[5].isChecked(),
			# }
		}
		self.save_session(session)

		logger.info("Quitting.\n\n")
		super(type(self),self).closeEvent(event)
