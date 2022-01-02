from PyQt5.QtWidgets import QWidget,QVBoxLayout
import numpy as np
import logging
logger = logging.getLogger(__name__)

class plot_container(QWidget):
	def __init__(self,gui):
		super().__init__()
		self.gui = gui
		self.gui.plot_mode = None

		self.vbox = QVBoxLayout()
		self.plot = QWidget()
		self.toolbar = QWidget()
		self.vbox.addStretch(1)
		self.vbox.addWidget(self.plot)
		self.vbox.addWidget(self.toolbar)
		self.setLayout(self.vbox)

	def change_mode(self,mode):
		if mode == 'smFRET':
			from .fret_plot import fret_canvas
			self.change_plotter(fret_canvas)
			self.gui.plot_mode = 'smFRET'

		elif mode == 'ND':
			from .nd_plot import nd_canvas
			self.change_plotter(nd_canvas)
			self.gui.plot_mode = 'ND'

	def default_prefs(self):
		return {}

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
		self.gui.timer = None

		from ..interface.stylesheet import ui_stylesheet
		temp_plot = canvas(self.gui)
		# temp_plot.setStyleSheet(ui_stylesheet)

		self.vbox.replaceWidget(self.toolbar,temp_plot.toolbar)
		self.vbox.replaceWidget(self.plot,temp_plot)
		self.toolbar.deleteLater()
		self.plot.deleteLater()
		self.toolbar = temp_plot.toolbar
		self.update_toolbar_theme()
		self.plot = temp_plot

		self.gui.pref_edited.connect(self.plot.redrawplot)
		# self.gui.new_selection_last.connect(self.catch_selection_change)
		self.gui.data_update.connect(self.plot.initialize_plots)

		# sel = self.gui.maven.selection.selected
		# if not sel is None:
		# 	self.catch_selection_change(sel)
		# else:
		# 	self.catch_selection_change([0])

		# self.plot.initialize_plots()

		# from PyQt5.QtWidgets import QApplication
		# QApplication.instance().processEvents()
		# self.gui.show()
		# self.gui.proc_data_update()
		# self.plot.draw()

		# self.plot.show()
		self.plot.redrawplot()

	def update_toolbar_theme(self):
		if self.gui.lightdark_mode == 'light':
			ldc = 'white'
		elif self.gui.lightdark_mode == 'dark':
			ldc = '#353535'
		self.toolbar.setStyleSheet('''QToolBar{border:none; background-color:%s; spacing:0px;}
			QToolBar::separator{background: %s}
			QToolButton::hover{background-color:lightgray;}'''%(ldc,ldc))
		self.toolbar.repaint()
