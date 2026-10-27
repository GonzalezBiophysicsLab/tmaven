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
		if plot_mode not in ('ND Raw', 'Normalized', 'ND Relative', 'smFRET'):
			raise ValueError('Unknown plot mode: {}'.format(plot_mode))
		if self.plot.plot_mode == plot_mode:
			return
		logger.info('Changing plot mode to %s', plot_mode)
		if self.gui.timer is not None:
			self.gui.timer.stop()
			self.gui.timer.deleteLater()
			self.gui.timer = None

		initialized = isinstance(self.plot, canvas)
		if initialized:
			self.gui.pref_edited.disconnect(self.plot.redrawplot)
			self.gui.data_update.disconnect(self.plot.initialize_plots)
			self.gui.data_update.disconnect(self.plot.softredraw)
		else:
			# Create the Qt canvas/toolbar once. Mode changes reuse their ownership,
			# event handlers and navigation actions rather than deleting live widgets.
			old_plot, old_toolbar = self.plot, self.toolbar
			self.plot = canvas(self.gui)
			self.toolbar = self.plot.toolbar
			for old, new in ((old_plot, self.plot), (old_toolbar, self.toolbar)):
				self.vbox.removeWidget(old)
				old.hide()
				old.deleteLater()
				self.vbox.addWidget(new)
			self.toolbar.setObjectName('plot_toolbar')

		try:
			if initialized:
				defaults = self.plot.default_prefs()
				defaults['plot.bg_color'] = self.gui.maven.prefs['plot.bg_color']
				self.gui.preferences_viewer.prefs_model.update_preferences(
					lambda: self.gui.maven.prefs.add(defaults))
			self.plot.plot_mode = plot_mode
			configure = {'ND Raw': self.plot.ndraw_prefs,
				'Normalized': self.plot.normalized_prefs,
				'ND Relative': self.plot.ndrelative_prefs,
				'smFRET': self.plot.smFRET_prefs}[plot_mode]
			configure(redraw=False)
			self.plot.initialize_plots()
			self.toolbar.update()
		finally:
			self.gui.pref_edited.connect(self.plot.redrawplot)
			self.gui.data_update.connect(self.plot.initialize_plots)
			# Repaint the current trace after the preceding slot resets its lines.
			self.gui.data_update.connect(self.plot.softredraw)
		self.update_toolbar_theme()
		self.plot.redrawplot()

	def update_toolbar_theme(self, palette=None):
		from PyQt5.QtWidgets import QApplication
		from ..interface.toolbar_theme import apply_toolbar_theme
		if palette is None:
			palette = QApplication.instance().palette()
		apply_toolbar_theme(self.toolbar, palette)
