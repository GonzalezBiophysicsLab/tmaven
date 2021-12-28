from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
from matplotlib.widgets import  RectangleSelector

from PyQt5.QtWidgets import (QSizePolicy,QVBoxLayout,QWidget,QToolBar,QAction,QHBoxLayout,
							QPushButton,QMainWindow,QDockWidget,QStyleFactory)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QMouseEvent
import numpy as np

from ..controllers.prefs import prefs_object
from ..interface.viewer_prefs import prefs_widget,pref_model
from .ui_plots import tryexcept

class popout_plot_container(QMainWindow):
	'''
	main ui - gui
	this window - window
	plot canvas - this.plot
	'''
	def __init__(self,nplots_x=1, nplots_y=1, gui=None):
		super(QMainWindow,self).__init__()
		self.gui = gui

		self.prefs = prefs_object()
		self.prefs.add_dictionary({
			'fig_width':2.5,
			'fig_height':2.5,

			'label_fontsize':10.0,
			'ylabel_offset':-0.165,
			'xlabel_offset':-0.25,
			'font':'Arial',
			'axes_linewidth':1.0,
			'axes_topright':False,
			'tick_fontsize':10.0,
			'tick_length_minor':2.0,
			'tick_length_major':4.0,
			'tick_linewidth':1.0,
			'tick_direction':'out',
			'subplots_left':0.125,
			'subplots_right':0.97,
			'subplots_top':0.97,
			'subplots_bottom':0.155,
			'subplots_hspace':0.04,
			'subplots_wspace':0.03,
		})

		self.prefs_widget = prefs_widget()
		self.prefs_model = pref_model(self)
		self.prefs_widget.set_model(self.prefs_model)

		self.prefs.edit_callback = self.replot
		self.qd_prefs = QDockWidget("Preferences",self)
		self.qd_prefs.setWidget(self.prefs_widget)
		self.qd_prefs.setAllowedAreas( Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
		self.addDockWidget(Qt.RightDockWidgetArea, self.qd_prefs)

		self.plot = popout_plot_container_widget(nplots_x, nplots_y, self.gui, self)

		self.toolbar = NavigationToolbar(self.plot,None)
		self.toolbar.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed))

		qwtotal = QWidget()
		qw = QWidget()
		self.buttonbox = QHBoxLayout()
		self.button_refresh = QPushButton("Refresh")
		button_prefs = QPushButton("Preferences")
		self.button_refresh.clicked.connect(self.replot)
		button_prefs.clicked.connect(self.open_preferences)
		self.buttonbox.addWidget(self.button_refresh)
		self.buttonbox.addWidget(button_prefs)

		self.buttonbox.addStretch(1)
		qw.setLayout(self.buttonbox)

		self.vbox = QVBoxLayout()
		self.vbox.addWidget(self.plot)
		self.vbox.addStretch(1)
		self.vbox.addWidget(self.toolbar)
		self.vbox.addWidget(qw)
		qwtotal.setLayout(self.vbox)
		self.setCentralWidget(qwtotal)
		self.plot.resize_figure()

		## Set the looks
		self.setStyle(QStyleFactory.create('Fusion'))
		from ..interface.stylesheet import ui_stylesheet
		self.setStyleSheet(ui_stylesheet)

		self.show()

	def really_close(self):
		super(popout_plot_container,self).close()

	def closeEvent(self,event):
		try:
			self.gui.activateWindow()
			self.gui.raise_()
			self.gui.setFocus()
		except:
			pass

	def resizeEvent(self,event):
		self.plot.resize_figure()
		super(popout_plot_container,self).resizeEvent(event)

	def replot_fxn(self,window): ## overload me !!!
		pass

	def replot(self):
		tryexcept(self.replot_fxn(self))

	# def set_callback(self,fxn):
	# 	self.replot_fxn = fxn
	# 	self.prefs.edit_callback = self.replot
	# 	self.button_refresh.clicked.connect(self.replot)

	def keyPressEvent(self,event):
		if event.key() == Qt.Key_Escape and not self.prefs_widget.le_filter.hasFocus():
			self.open_preferences()
			self.prefs_widget.le_filter.setFocus()
			return
		super(popout_plot_container,self).keyPressEvent(event)

	def open_preferences(self):
		try:
			if not self.qd_prefs.isVisible():
				self.qd_prefs.setVisible(True)
			self.qd_prefs.raise_()
		except:
			self.qd_prefs.show()


class popout_plot_container_widget(FigureCanvas):
	def __init__(self,nplots_x=1, nplots_y=1, gui=None,_window=None):
		self.gui = gui ## main ui
		self._window = _window ## popplot


		self.nplots_x = nplots_x
		self.nplots_y = nplots_y

		# self.f,self.ax = plt.subplots(nplots_x, nplots_y, figsize=(self.prefs['fig_width']/QPixmap().devicePixelRatio(),self.prefs['fig_height']/QPixmap().devicePixelRatio()))
		fig,self.ax = plt.subplots(self.nplots_x,self.nplots_y)
		super(popout_plot_container_widget,self).__init__(fig)

		self.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))
		# self.figure.set_dpi(self.figure.get_dpi()/self.devicePixelRatio())
		plt.close(self.figure)

		self.updateGeometry()
		self.fix_ax()
		# self.mpl_connect('button_press_event', self.mouse_handler)

	@tryexcept
	def resize_figure(self):
		## OKAY -- call this when the mainwindow that houses the traj_plot_container is resized, because this widget never resizes otherwise
		self.figure.set_figwidth(self._window.prefs['fig_width']/self.devicePixelRatio())
		self.figure.set_figheight(self._window.prefs['fig_height']/self.devicePixelRatio())
		self.updateGeometry()
		self.draw()

	def keyPressEvent(self,event):
		super(popout_plot_container_widget,self).keyPressEvent(event)

	@tryexcept
	def fix_ax(self):
		pp = self._window.prefs
		self.figure.subplots_adjust(left=pp['subplots_left'],right=pp['subplots_right'],top=pp['subplots_top'],bottom=pp['subplots_bottom'],hspace=pp['subplots_hspace'],wspace=pp['subplots_wspace'])

		for aa in self.figure.axes:
			dpr = self.devicePixelRatio()
			for asp in ['top','bottom','left','right']:
				aa.spines[asp].set_linewidth(pp['axes_linewidth']/dpr)
				if asp in ['top','right']:
					aa.spines[asp].set_visible(pp['axes_topright'])

				tickdirection = pp['tick_direction']
				if not tickdirection in ['in','out']: tickdirection = 'in'
				aa.tick_params(labelsize=pp['tick_fontsize']/dpr, axis='both', direction=tickdirection , width=pp['tick_linewidth']/dpr, length=pp['tick_length_minor']/dpr)
				aa.tick_params(axis='both',which='major',length=pp['tick_length_major']/dpr)
				for label in aa.get_xticklabels():
					label.set_family(pp['font'])
				for label in aa.get_yticklabels():
					label.set_family(pp['font'])
		self.draw()

	def cla(self):
		for a in self.figure.axes:
			a.cla()
		self.fix_ax()

	def best_ticks(self,ymin,ymax,nticks):
		m = nticks
		if m <= 0: return ()
		if ymax <= ymin: return ()
		delta = ymax-ymin

		d = 10.0**(np.floor(np.log10(delta)))
		ind = np.arange(1,10)
		ind = np.concatenate((1./ind[::-1],ind))
		di = d*ind
		for i in range(ind.size):
			if np.floor(delta/di[i]) < m:
				s = di[i]
				break
		y0 = np.ceil(ymin/s)*s
		delta = ymax - y0

		d = 10.0**(np.floor(np.log10(delta)))
		ind = np.arange(1,10)
		ind = np.concatenate((1./ind[::-1],ind))
		di = d*ind
		for i in range(ind.size):
			if np.floor(delta/di[i]) < m:
				s = di[i]
				break
		y0 = np.ceil(ymin/s)*s
		delta = ymax - y0
		n = np.floor(delta/s+1e-10)
		return y0 + np.arange(n+1)*s
