from PyQt5.QtWidgets import QWidget,QVBoxLayout
import numpy as np
import logging
logger = logging.getLogger(__name__)

class plot_container(QWidget):
	def __init__(self,gui):
		super().__init__(parent=gui)
		self.gui = gui

		self.vbox = QVBoxLayout()
		self.plot = QWidget()
		self.plot.plot_mode = None
		self.toolbar = QWidget()
		self.vbox.addStretch(1)
		self.vbox.addWidget(self.plot)
		self.vbox.addWidget(self.toolbar)
		self.setLayout(self.vbox)

	def change_mode(self,mode):			
		self.change_plotter(mode)

	def default_prefs(self):
		return {}

	def change_plotter(self,plot_mode):
		from .multi_plot import multi_canvas as canvas
		logger.info('New plot container')

		#### Unhook plotter
		try: self.gui.pref_edited.disconnect(self.plot.redrawplot)
		except: pass
		try: self.gui.data_update.disconnect(self.plot.initialize_plots)
		except: pass
		self.gui.timer = None

		from ..interface.stylesheet import ui_stylesheet
		temp_plot = canvas(self.gui)
		# temp_plot.setStyleSheet(ui_stylesheet)

		self.vbox.replaceWidget(self.toolbar,temp_plot.toolbar)
		self.vbox.replaceWidget(self.plot,temp_plot)
		self.toolbar.deleteLater()
		self.plot.deleteLater()
		del self.plot
		del self.toolbar
		self.toolbar = temp_plot.toolbar
		self.update_toolbar_theme()
		self.plot = temp_plot
		self.plot.plot_mode = plot_mode

		if plot_mode == 'ND Raw':
			self.plot.ndraw_prefs()		
		elif plot_mode == 'Normalized':
			self.plot.normalized_prefs()
		elif plot_mode == 'ND Relative':
			self.plot.ndrelative_prefs()
		elif plot_mode == 'smFRET':
			self.plot.smFRET_prefs()


		self.gui.pref_edited.connect(self.plot.redrawplot)
		# self.gui.new_selection_last.connect(self.catch_selection_change)
		self.gui.data_update.connect(self.plot.initialize_plots)

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
