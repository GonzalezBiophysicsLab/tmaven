### borrows heavily from Mu-editor. Credit to them
import logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QApplication,QMainWindow,QDesktopWidget,QLabel,QMessageBox,QShortcut,QPushButton,QMenu,QAction,QStyleFactory,QSlider, QSizePolicy, QTreeWidget, QTreeWidgetItem, QDockWidget, QLabel, QHBoxLayout
from PyQt5.QtGui import QIcon, QKeySequence, QPalette, QColor
from PyQt5.QtCore import Qt, pyqtSignal
import matplotlib
matplotlib.use('Qt5Agg') ## forces Qt5 early on...

import numpy as np
from .resources import load_pixmap, load_icon

class_keys = {Qt.Key_1:1,Qt.Key_2:2,Qt.Key_3:3,Qt.Key_4:4,Qt.Key_5:5,Qt.Key_6:6,Qt.Key_7:7,Qt.Key_8:8,Qt.Key_9:9,Qt.Key_0:0}

class main_window(QMainWindow):
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
		self.show()

	def initialize_widgets(self):
		## window style
		self.setWindowTitle('tMAVEN')
		from .stylesheet import ui_stylesheet
		self.setStyleSheet(ui_stylesheet)
		self.setStyle(QStyleFactory.create('Fusion')) ## WOW THIS THROWS RANDOM SEGFAULTS WHEN QUITTING?
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

		# #### smd info
		# from .viewer_smd_info import smd_info_viewer
		# self.smd_info_viewer = smd_info_viewer(self)
		# # self.smd_info_viewer.toggle()

		## hdf5 explorer
		from .hdf5_view.hdf5_view import hdf5_view_container
		self.hdf5_viewer = hdf5_view_container(self)
		# self.hdf5_viewer.launch()

		## menu
		self.menubar = self.menuBar()
		self.menubar.setNativeMenuBar(False)
		# self.menubar.setStyleSheet(stylesheet.ss_qmenubar)
		self.reset_menus()

		#### total layout
		from PyQt5.QtWidgets import QVBoxLayout,QWidget
		self._qw = QWidget()
		self.vbox = QVBoxLayout()
		# self.vbox.addWidget(self.molecule_group_viewer)
		self.qw = QWidget()
		hbox = QHBoxLayout()
		hbox.addWidget(self.slider_select)
		hbox.addWidget(self.label_molnum)
		self.qw.setLayout(hbox)
		# self.vbox.addStretch(1)
		self.vbox.addWidget(self.plot_container)
		self.vbox.addWidget(self.qw)
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

		self.menu_view.addAction('Reset GUI',self.default_session)
		self.menu_traj = self.menu_view.addMenu('Plot Type')
		self.menu_traj.addAction('ND',lambda : self.plot_container.change_mode('ND'))
		self.menu_traj.addAction('smFRET',lambda : self.plot_container.change_mode('smFRET'))
		self.menu_theme = self.menu_view.addMenu('Theme')
		self.menu_theme.addAction('Light',self.change_theme_light)
		self.menu_theme.addAction('Dark',self.change_theme_dark)
		# self.menu_view.addAction('SMD Info',self.smd_info_viewer.toggle)
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

		# for menu in [self.menu_file,self.menu_tools,self.menu_other,self.menu_view,self.menu_prefs,self.menu_scripts,self.menu_plots]:
			# menu.setStyleSheet(stylesheet.ss_qmenu)

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
			self.update_mol_label()
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
		if not self.plot_mode is None:
			self.plot_container.plot.update_data(self.index)
		self.update_mol_label()
		self.slider_select.blockSignals(True)
		self.slider_select.setValue(self.index)
		self.slider_select.setRange(0,self.maven.data.nmol-1)
		self.slider_select.blockSignals(False)

	def update_mol_label(self):
		msg = '{}/{}'.format(self.index, self.maven.data.nmol-1)
		if self.maven.data.nmol > 0:
			msg += ' {}'.format(self.maven.data.classes[self.index])
		self.label_molnum.setText(msg)

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

	def default_session(self):
		self.plot_container.change_mode('smFRET')
		self.size_window()
		self.show()
		# self.plot_container.plot.redrawplot() ## it is necessary to draw/resize after showing the interface, otherwise the DPI is messed up
		self.change_theme_light() ## redrawplot called in this fxn too

	def dark_theme(self):
		dark_palette = QPalette()
		dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
		dark_palette.setColor(QPalette.WindowText, Qt.white)
		dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
		dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
		dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
		dark_palette.setColor(QPalette.ToolTipText, Qt.white)
		dark_palette.setColor(QPalette.Text, Qt.white)
		dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
		dark_palette.setColor(QPalette.ButtonText, Qt.white)
		dark_palette.setColor(QPalette.BrightText, Qt.red)
		dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
		dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
		dark_palette.setColor(QPalette.HighlightedText, Qt.black)
		return dark_palette

	def light_theme(self):

		# app.setStyle("Fusion")
		light_palette = QPalette()
		light_palette.setColor(QPalette.Window, QColor(255,255,255))
		light_palette.setColor(QPalette.WindowText, Qt.black)
		light_palette.setColor(QPalette.Base, QColor(230, 230, 230))
		####
		light_palette.setColor(QPalette.AlternateBase, QColor(202,202,202))
		light_palette.setColor(QPalette.ToolTipBase, Qt.black)
		light_palette.setColor(QPalette.ToolTipText, Qt.black)
		light_palette.setColor(QPalette.Text, Qt.black)
		light_palette.setColor(QPalette.Button, QColor(202,202,202))
		light_palette.setColor(QPalette.ButtonText, Qt.black)
		light_palette.setColor(QPalette.BrightText, Qt.blue)
		light_palette.setColor(QPalette.Link, QColor(213, 125, 37))
		light_palette.setColor(QPalette.Highlight, QColor(213, 125, 37))
		light_palette.setColor(QPalette.HighlightedText, Qt.white)
		return light_palette

	def _change_theme(self,palette):
		app = QApplication.instance()
		app.setPalette(palette)
		self.menubar.setPalette(palette)
		self.preferences_viewer.update_theme(palette)
		self.molecules_viewer.update_theme(palette)
		self.plot_container.update_toolbar_theme()
		self.label_molnum.setPalette(palette)

	def change_theme_dark(self):
		palette = self.dark_theme()
		self.lightdark_mode = 'dark'
		self.maven.prefs['plot.bg_color'] = '#353535'
		self._change_theme(palette)

	def change_theme_light(self):
		palette = self.light_theme()
		self.lightdark_mode = 'light'
		self.maven.prefs['plot.bg_color'] = '#FFFFFF'
		self._change_theme(palette)

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
			self.default_session()
			return

		with open(config_file,'r') as f:
			session = json.load(f)
		session_str =  ",".join(f"{key}:{value}" for key, value in session.items())
		logger.info('Loaded session data from {}.\nSession data - {}'.format(config_file,session_str))
		x = session['window']['x'] if 'x' in session['window'] else None
		y = session['window']['y'] if 'y' in session['window'] else None
		w = session['window']['w'] if 'w' in session['window'] else None
		h = session['window']['h'] if 'h' in session['window'] else None

		lightdark_mode = session['lightdark_mode'] if 'lightdark_mode' in session else 'light'
		if lightdark_mode == 'light':
			palette = self.light_theme()
			self.maven.prefs['plot.bg_color'] = '#FFFFFF'
			self.lightdark_mode = 'light'
		elif lightdark_mode == 'dark':
			palette = self.dark_theme()
			self.maven.prefs['plot.bg_color'] = '#353535'
			self.lightdark_mode = 'dark'

		plot_mode = session['plot_mode'] if 'plot_mode' in session else 'smFRET'
		self.plot_container.change_mode(plot_mode)
		self.size_window(x,y,w,h)
		app = QApplication.instance()
		app.setPalette(palette)
		# app.setStyleSheet("QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }")
		app.processEvents()
		self.show()
		self.plot_container.plot.redrawplot() ## it is necessary to draw/resize after showing the interface, otherwise the DPI is messed up


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
		# message_box.setStyleSheet('''background-color:white;font-size: 12px;''')

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
			},
			"plot_mode": self.plot_mode,
			"lightdark_mode" : self.lightdark_mode,
		}
		self.save_session(session)

		logger.info("Quitting.\n\n")
		return super(type(self),self).closeEvent(event)
