### borrows heavily from Mu-editor. Credit to them
import logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger("tmaven")

from PyQt5.QtWidgets import QApplication,QMainWindow,QDesktopWidget,QLabel,QMessageBox,QShortcut,QPushButton,QMenu,QAction,QSlider, QSizePolicy, QTreeWidget, QTreeWidgetItem, QDockWidget, QLabel, QHBoxLayout
from PyQt5.QtGui import QIcon, QKeySequence, QPalette, QColor
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QSize
import matplotlib
matplotlib.use('Qt5Agg') ## forces Qt5 early on...

import numpy as np
from .resources import load_pixmap, load_icon
import appdirs
import json
import os

class_keys = {Qt.Key_1:1,Qt.Key_2:2,Qt.Key_3:3,Qt.Key_4:4,Qt.Key_5:5,Qt.Key_6:6,Qt.Key_7:7,Qt.Key_8:8,Qt.Key_9:9,Qt.Key_0:0}

from .session import DesktopSession

class main_window(DesktopSession, QMainWindow):
	data_update = pyqtSignal()
	pref_edited = pyqtSignal()

	def __init__(self,maven,app):
		super().__init__(parent=None)
		self.app = app
		self.maven = maven
		self.index = 0
		self.desired_index = 0
		self.timer = None
		self.lightdark_mode = 'light'
		self.theme_mode = 'system'
		self._system_palette = app.palette()
		self._applying_theme = False
		self.initialize_widgets()
		self.lwd_load()
		self._active_dock = "molecules"
		for name, dock in (("molecules", self.molecules_viewer.dock), ("preferences", self.preferences_viewer.dock), ("log", self.log_viewer.dock)):
			dock.visibilityChanged.connect(lambda visible, n=name: setattr(self, "_active_dock", n) if visible else None)
		self._default_layout = self.saveState()
		self._default_headers = [v.horizontalHeader().saveState() for v in
			(self.molecules_viewer.viewer, self.preferences_viewer.viewer.proxy_view)]

	def initialize_widgets(self):
		## window style
		from .. import __version__
		self.setWindowTitle('tMAVEN (%s)'%(str(__version__)))
		from .stylesheet import ui_stylesheet
		self.setStyleSheet(ui_stylesheet)
		self.setWindowIcon(load_icon('logo.png'))
		self.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
		self.closeEvent = self.quit

		## widgets

		#### molecule number label
		self.label_molnum = QLabel('0/0')
		from PyQt5.Qt import QFont,QFontDatabase
		monospace = QFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
		self.label_molnum.setFont(monospace)
		
		## lock
		self.flag_locked = False
		self.button_lock = QPushButton()
		# from .stylesheet import slider_style_sheet
		# self.button_lock.setStyleSheet(slider_style_sheet)
		self.button_lock.setCheckable(True)
		self.button_lock.setIconSize(QSize(16, 16))
		self.button_lock.clicked.connect(self.toggle_lock)
		self.update_lock_icon()

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
		from .viewer_log import log_viewer
		self.log_viewer = log_viewer(self)
		
		from .ui_scripts import scripts_toolbar
		self.scripts_toolbar = scripts_toolbar(self)
		self.addToolBar(Qt.TopToolBarArea, self.scripts_toolbar)
		# self.scripts_toolbar.hide()
		
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
		hbox.addWidget(self.button_lock)
		self.qw.setLayout(hbox)
		# self.vbox.addStretch(1)
		self.vbox.addWidget(self.plot_container)
		self.vbox.addWidget(self.qw)
		self._qw.setSizePolicy(QSizePolicy.MinimumExpanding,QSizePolicy.MinimumExpanding)
		self._qw.setLayout(self.vbox)
		self.setCentralWidget(self._qw)

		# self.preferences_viewer.toggle()
		# self.molecules_viewer.toggle()
		self.tabifyDockWidget(self.preferences_viewer.dock,self.molecules_viewer.dock)
		self.splitDockWidget(self.preferences_viewer.dock, self.log_viewer.dock, Qt.Vertical)

		## connections
		self._maven_unsubscribe = [
			self.maven.subscribe('data_changed', self.data_update.emit),
			self.maven.subscribe('preferences_changed', self.pref_edited.emit),
		]
		self.pref_edited.connect(self.preferences_viewer.prefs_model.refresh)
		self.data_update.connect(self.proc_data_update)
		self.maven.emit_data_update()

	def toggle_lock(self):
		self.flag_locked = not self.flag_locked
		self.update_lock_icon()
		
	def update_lock_icon(self):
		if self.flag_locked:
			self.button_lock.setIcon(load_icon('locked.png'))
		else:
			self.button_lock.setIcon(load_icon('unlocked.png'))

	def reset_menus(self):
		from . import ui_io, ui_scripts, ui_cull, ui_corrections, ui_normalize, ui_selection, ui_photobleaching, ui_trace_filter, ui_experimental
		from .modeler import ui_modeler

		logger.info('Building Menus')
		self.menubar.clear()
		self.menu_file  = QMenu('File',self)
		self.menu_tools = QMenu('Tools',self)
		# self.menu_other = QMenu('Other',self)
		self.menu_view = QMenu('View',self)
		# self.menu_prefs = QMenu('Preferences',self)
		self.menu_plots = QMenu('Plots',self)

		self.menu_load,self.menu_save = ui_io.build_menu(self)
		self.menu_scripts = QMenu('Scripts',self)
		# self.menu_experimental = self.menu_other.addMenu('Experimental')
		self.trace_filter = ui_trace_filter.container_trace_filter(self)
		self.menu_cull = ui_cull.build_menu(self)
		self.menu_corrections,self.menu_correction_filters = ui_corrections.build_menu(self)
		self.menu_normalize = ui_normalize.build_menu(self)
		self.menu_selection, self.menu_order, self.menu_classes = ui_selection.build_menu(self)
		self.menu_photobleaching = ui_photobleaching.build_menu(self)
		self.menu_modeler = ui_modeler.build_menu(self)


		self.menu_file.addMenu(self.menu_load)
		self.menu_file.addMenu(self.menu_save)
		self.action_clear_data = self.menu_file.addAction('Clear Data',self.clear_data)
		# self.menu_file.addMenu(self.menu_prefs)
		# self.menu_prefs.addAction('Load Preferences',self.preferences_viewer.load)
		# self.menu_prefs.addAction('Save Preferences',self.preferences_viewer.save)
		self.menu_file.addAction('Exit',self.close,'Ctrl+Q')


		self.menu_traj = self.menu_view.addMenu('Plot Type')

		# self.menu_rel = self.menu_traj.addMenu('Relative')
		self.menu_traj.addAction('Raw',lambda : self.plot_container.change_mode('ND Raw'))
		self.menu_traj.addAction('Normalized',lambda : self.plot_container.change_mode('Normalized'))
		self.menu_traj.addAction('Relative',lambda : self.plot_container.change_mode('ND Relative'))
		self.menu_traj.addAction('smFRET',lambda : self.plot_container.change_mode('smFRET'))

		self.menu_theme = self.menu_view.addMenu('Theme')
		self.build_desktop_menus()
		# self.menu_view.addAction('SMD Info',self.smd_info_viewer.toggle)
		self.menu_view.addAction('Show Log',self.show_log)
		self.menu_view.addAction(self.hdf5_viewer.action)
		self.menu_view.addAction('Molecule Table',self.molecules_viewer.toggle,'Ctrl+T')
		self.menu_view.addAction('Preferences',self.preferences_viewer.toggle,'Ctrl+P')
		self.menu_view.addAction('Scripts Toolbar',self.scripts_toolbar.toggle,'Ctrl+R')

		self.menu_scripts.addAction('Run Script',lambda:ui_scripts.run(self))
		self.menu_scripts.addAction('Run Input',lambda:ui_scripts.input_run(self))
		# self.menu_experimental.addAction('Dwell Inversion',lambda:ui_experimental.dwell_inversion(self))

		self.menu_tools.addMenu(self.menu_selection)
		self.menu_tools.addMenu(self.menu_classes)
		self.menu_tools.addMenu(self.menu_order)
		self.menu_tools.addMenu(self.menu_cull)
		self.menu_tools.addMenu(self.menu_corrections)
		self.menu_corrections.addMenu(self.menu_correction_filters)
		self.menu_tools.addMenu(self.menu_normalize)
		self.menu_tools.addMenu(self.menu_photobleaching)
		self.menu_tools.addAction('Filter Traces',self.trace_filter.launch,shortcut='Ctrl+F')

		self.popplots = {'1D':None,'2D':None,'TDP':None,'TM':None,'VB':None,'Dwell':None}
		self.menu_plots.addAction('1D Histogram',lambda : self.popplot_launch('1D'))
		self.menu_plots.addAction('2D Histogram',lambda : self.popplot_launch('2D'))
		self.menu_plots.addAction('Transition Density Plot',lambda : self.popplot_launch('TDP'))
		self.menu_plots.addAction('Transition Prob Hist',lambda : self.popplot_launch('TM'))
		self.menu_plots.addAction('vb Model States',lambda : self.popplot_launch('VB'))
		self.menu_plots.addAction('Dwell Times Plot',lambda : self.popplot_launch('Dwell'))

		# for menu in [self.menu_file,self.menu_tools,self.menu_other,self.menu_view,self.menu_prefs,self.menu_scripts,self.menu_plots]:
			# menu.setStyleSheet(stylesheet.ss_qmenu)

		self.menubar.addMenu(self.menu_file)

		self.menubar.addMenu(self.menu_tools)
		self.menubar.addMenu(self.menu_modeler)
		self.menubar.addMenu(self.menu_plots)
		# self.menubar.addMenu(self.menu_other)
		self.menubar.addMenu(self.menu_scripts)
		self.menubar.addMenu(self.menu_view)

	def popplot_launch(self,popplottype):
		from .ui_analysisplots import popplot_container
		if self.popplots[popplottype] is None:
			if popplottype == '1D':
				self.popplots[popplottype] = popplot_container(self,self.maven.plots.data_hist1d)
			elif popplottype == '2D':
				self.popplots[popplottype] = popplot_container(self,self.maven.plots.data_hist2d)
			elif popplottype == 'TDP':
				self.popplots[popplottype] = popplot_container(self,self.maven.plots.data_tdp)
			elif popplottype == 'TM':
				self.popplots[popplottype] = popplot_container(self,self.maven.plots.tm_hist)
			elif popplottype == 'VB':
				self.popplots[popplottype] = popplot_container(self,self.maven.plots.model_vbstates)
			elif popplottype == 'Dwell':
				self.popplots[popplottype] = popplot_container(self,self.maven.plots.survival_dwell)
		else:
			self.popplots[popplottype].plot()
			self.popplots[popplottype].show()
			self.popplots[popplottype].raise_()

	def clear_data(self):
		from PyQt5.QtWidgets import QMessageBox
		reply = QMessageBox.question(self,"Clear Data?","Are you sure you want to remove all the current data?",QMessageBox.Yes | QMessageBox.No)
		if reply == QMessageBox.Yes:
			self.index = 0
			self.maven.io.clear_data()
			self.proc_data_update()

	def keyPressEvent(self,event):
		try: ## don't want the user to blow it up
			kk = event.key()
			if kk == Qt.Key_G:
					self.plot_container.plot.ax[0,0].grid()
					self.plot_container.plot.ax[1,0].grid()
					self.plot_container.plot.softredraw()
					self.plot_container.plot.update_blits()
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
			elif kk in [Qt.Key_Minus,Qt.Key_Equal] and not self.flag_locked:
				start = self.maven.data.pre_list[self.index] + (-1 if kk == Qt.Key_Minus else 1)
				self.maven.data.set_trace_boundaries(self.index, start=start)
			elif kk in [Qt.Key_BracketLeft,Qt.Key_BracketRight] and not self.flag_locked:
				stop = self.maven.data.post_list[self.index] + (-1 if kk == Qt.Key_BracketLeft else 1)
				self.maven.data.set_trace_boundaries(self.index, stop=stop)
			elif kk == Qt.Key_R and not self.flag_locked:
				self.maven.data.reset_trace_boundaries(self.index)
			elif kk == Qt.Key_C and not self.flag_locked:
				self.maven.data.split_trace(self.index,self.maven.data.post_list[self.index])
			elif kk == Qt.Key_V and not self.flag_locked:
				self.change_index(self.maven.data.collect_trace(self.index))
			elif kk == Qt.Key_P and not self.flag_locked:
				try:
					self.maven.photobleaching.calc_single_photobleach(self.index)
				except:
					pass
			elif event.key() in class_keys and not self.flag_locked:
				new_class_ind = class_keys[kk]
				self.maven.selection.set_class_from_selection(self.index, new_class_ind)
			else:
				super().keyPressEvent(event)
				return
			self.molecules_viewer.update()
			self.plot_container.plot.setFocus()
		except:
			pass
		return

	def update_selection_from_slider(self):
		self.slider_select.blockSignals(True)
		self.change_index(self.slider_select.value())
		self.slider_select.blockSignals(False)

	def proc_data_update(self):
		###
		if self.index >= self.maven.data.nmol:
			self.index = 0
		###
		if not self.plot_container.plot.plot_mode is None:
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
		self.timer = QTimer(self, singleShot=True)
		self.timer.setInterval(40) ## 25 FPS = 40 msec
		self.timer.timeout.connect(self._timesup)
		self.timer.start()

	def _timesup(self):
		if self.timer is not None:
			self.timer.deleteLater()
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

	def dark_theme(self):
		dark_palette = QPalette()
		dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
		dark_palette.setColor(QPalette.WindowText, Qt.white)
		dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
		dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
		dark_palette.setColor(QPalette.ToolTipBase, QColor(53, 53, 53))
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
		light_palette.setColor(QPalette.ToolTipBase, Qt.white)
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
		self.log_viewer.update_theme(palette)
		self.molecules_viewer.update_theme(palette)
		self.plot_container.update_toolbar_theme(palette)
		from .toolbar_theme import apply_toolbar_theme
		apply_toolbar_theme(self.scripts_toolbar, palette)
		self.label_molnum.setPalette(palette)
		self.button_lock.setPalette(palette)

	def change_theme_dark(self):
		self.apply_theme('dark')

	def change_theme_light(self):
		self.apply_theme('light')

	def show_log(self,event=None):
		self.log_viewer.toggle()


	def quit(self, event=None):
		"""
		Exit the application.
		"""
		from PyQt5.QtWidgets import QMessageBox
		message_box = QMessageBox(parent=self)
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

		self.session_save()
		if self.timer is not None:
			self.timer.stop()
			self.timer.deleteLater()
			self.timer = None

		for unsubscribe in self._maven_unsubscribe:
			unsubscribe()
		self._maven_unsubscribe = []
		logger.info("Quitting...")
		return super().closeEvent(event)
