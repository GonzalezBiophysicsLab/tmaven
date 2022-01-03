from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import QApplication,QSizePolicy,QVBoxLayout,QWidget,QHBoxLayout,QMainWindow,QStyleFactory
from PyQt5.QtCore import Qt,QSize
import numpy as np

from .viewer_prefs import prefs_widget,pref_model

class popplot_container(QMainWindow):
	'''
	gui = main ui, not this window
	maven_plot = the controller in maven that will be doing the plotting
	'''
	def __init__(self,gui,maven_plot):
		super(QMainWindow,self).__init__()
		self.gui = gui
		self.maven_plot = maven_plot

		app = QApplication.instance()
		screen = app.screens()[0]
		self.dpi = screen.physicalDotsPerInch()
		self.dpr = screen.devicePixelRatio()

		self.prefs_widget = prefs_widget()
		self.prefs_model = pref_model(self.maven_plot)
		self.prefs_widget.set_model(self.prefs_model)

		self.maven_plot.prefs.emit_changed = lambda : self.plot()

		self.fig,self.ax = plt.subplots(1,dpi=self.dpi)
		self.canvas = FigureCanvas(self.fig)
		# plt.close(self.figure)
		self.canvas.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))
		self.canvas.sizeHint = self.canvas_size_hint
		self.fig.set_dpi(self.dpi*self.dpr)

		self.toolbar = NavigationToolbar(self.canvas,None)
		self.toolbar.setIconSize(QSize(24,24))
		self.toolbar.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed))
		# self.toolbar.addAction('Refresh',self.plot)

		self.menubar = self.menuBar()
		self.menubar.setNativeMenuBar(False)
		self.menubar.addAction('Refresh',self.plot)
		self.menubar.addAction('Reset Preferences',self.reset_prefs)

		qw = QWidget()
		vbox = QVBoxLayout()
		vbox.addWidget(self.canvas)
		vbox.addStretch(1)
		vbox.addWidget(self.toolbar)
		qw.setLayout(vbox)

		qwtot = QWidget()
		hbox = QHBoxLayout()
		hbox.addWidget(self.prefs_widget)
		hbox.addWidget(qw)
		hbox.addStretch(1)
		qwtot.setLayout(hbox)

		## Set the looks
		self.setCentralWidget(qwtot)
		self.setStyle(QStyleFactory.create('Fusion'))
		from .stylesheet import ui_stylesheet
		self.setStyleSheet(ui_stylesheet)

		self.resize_figure()
		self.show()
		self.plot()

	def canvas_size_hint(self):
		return QSize(self.maven_plot.prefs['fig_width']*self.dpi,self.maven_plot.prefs['fig_height']*self.dpi)

	def resize_figure(self):
		## OKAY -- call this when the mainwindow that houses the traj_plot_container is resized, because this widget never resizes otherwise
		self.fig.set_figwidth(self.maven_plot.prefs['fig_width']/self.dpr)
		self.fig.set_figheight(self.maven_plot.prefs['fig_height']/self.dpr)
		self.canvas.updateGeometry()
		self.canvas.draw()

	def plot(self,event=None):
		self.resize_figure()
		self.ax.cla()
		self.maven_plot.plot(self.fig,self.ax)
		self.canvas.draw()

	def reset_prefs(self,event=None):
		self.maven_plot.defaults()
		# from PyQt5.QtCore import QModelIndex
		# self.prefs_model.dataChanged.emit(QModelIndex(),QModelIndex())
		self.prefs_widget.proxy_model.layoutChanged.emit()

	def keyPressEvent(self,event):
		if event.key() == Qt.Key_Escape and not self.prefs_widget.le_filter.hasFocus():
			self.prefs_widget.le_filter.setFocus()
			return
		super().keyPressEvent(event)
