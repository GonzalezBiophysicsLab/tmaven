from PyQt5.QtWidgets import QWidget,QSizePolicy,QVBoxLayout,QWidget, QMenu, QAction,QApplication
from PyQt5.QtCore import pyqtSignal,QSize,Qt
from PyQt5.QtGui import QScreen
import numpy as np
import logging
logger = logging.getLogger(__name__)

class plot_container(QWidget):
	def __init__(self,gui):
		super().__init__()
		self.gui = gui

		self.vbox = QVBoxLayout()
		self.plot = QWidget()
		self.toolbar = QWidget()
		self.vbox.addStretch(1)
		self.vbox.addWidget(self.plot)
		self.vbox.addWidget(self.toolbar)
		self.setLayout(self.vbox)

		self.setup_fret()

	def setup_fret(self):
		from .fret_plot import fret_canvas,default_prefs
		self.gui.maven.prefs.add_dictionary(default_prefs)
		self.change_plotter(fret_canvas)
		self.plot.initialize_plots()

	def change_plotter(self,canvas):
		logger.info('New plot container')

		#### Unhook plotter
		try: self.gui.pref_edited.disconnect(self.plot.redrawplot)
		except: pass
		# except Exception as e: logger.error(e)
		# try:self.gui.new_selection_last.disconnect(self.catch_selection_change)
		# except Exception as e: logger.error(e)
		try: self.gui.data_update.disconnect(self.plot.initialize_plots)
		# except Exception as e: logger.error(e)
		except: pass
		self.timer = None

		from ..interface.stylesheet import ui_stylesheet
		temp_plot = canvas(self.gui)
		temp_plot.setStyleSheet(ui_stylesheet)
		temp_plot.toolbar.setStyleSheet('''
			QToolBar{border:none; background-color:white; spacing:0px;}
			QToolBar::separator{background: white}
			QToolButton::hover{background-color:lightgray;}''')

		self.vbox.replaceWidget(self.toolbar,temp_plot.toolbar)
		self.vbox.replaceWidget(self.plot,temp_plot)
		del self.toolbar
		del self.plot
		self.toolbar = temp_plot.toolbar
		self.plot = temp_plot

		self.gui.pref_edited.connect(self.plot.redrawplot)
		# self.gui.new_selection_last.connect(self.catch_selection_change)
		self.gui.data_update.connect(self.plot.initialize_plots)
		# self.plot.redrawplot()


		# sel = self.gui.maven.selection.selected
		# if not sel is None:
		# 	self.catch_selection_change(sel)
		# else:
		# 	self.catch_selection_change([0])
