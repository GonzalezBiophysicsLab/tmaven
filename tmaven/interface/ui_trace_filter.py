import logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
from PyQt5.QtWidgets import QListWidget,QMainWindow,QWidget,QAbstractItemView,QFileDialog,QPushButton,QVBoxLayout,QHBoxLayout,QLabel,QFormLayout,QLineEdit,QComboBox,QAction, QApplication, QSizePolicy
from PyQt5.QtCore import Qt,  QObject, QSize
from PyQt5.QtGui import QDoubleValidator, QIntValidator, QPixmap

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import matplotlib.pyplot as plt
import numpy as np


class container_trace_filter(QObject):
	'''Wrapper for better interfacing with the GUI'''
	def __init__(self,gui):
		super(container_trace_filter,self).__init__()
		self.gui = gui

	def launch(self):
		try:
			del self._window
		except:
			pass
		self._window = gui_trace_filter(self.gui.maven,self.gui)
		self._window.setWindowTitle('Selection Classifier')
		self._window.show()


class gui_trace_filter(QMainWindow):
	'''QMainWindow to popout the UI'''
	def __init__(self,maven,parent):
		super(QMainWindow,self).__init__(parent=parent)
		self.maven = maven

		cqw = QWidget()

		self.le_low_sbr = QLineEdit()
		self.le_high_sbr = QLineEdit()
		self.le_min_frames = QLineEdit()
		self.le_skip_frames = QLineEdit()
		self.combo_data = QComboBox()
		self.combo_data.addItems(['Donor','Acceptor','Donor+Acceptor'])

		dv = QDoubleValidator(0.1, 1000, 2)
		dv.setNotation(QDoubleValidator.StandardNotation)
		iv = QIntValidator(0, 10000)
		[le.setValidator(dv) for le in [self.le_low_sbr,self.le_high_sbr]]
		[le.setValidator(iv) for le in [self.le_min_frames,self.le_skip_frames]]


		self.label_proportions = QLabel('[0, 0, 0, 0, 0]')

		app = QApplication.instance()
		screen = app.screens()[0]
		self.dpi = screen.physicalDotsPerInch()
		self.dpr = screen.devicePixelRatio()
		self.fig, self.ax = plt.subplots(1,dpi=self.dpi)#(4.0/QPixmap().devicePixelRatio(), 2.5/QPixmap().devicePixelRatio()),sharex=True)
		self.canvas = FigureCanvas(self.fig)
		# self.toolbar = NavigationToolbar(self.canvas,None)
		self.fig = self.canvas.figure
		self.canvas.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding))
		# self.toolbar.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))
		self.fig.set_dpi(self.dpi)
		self.canvas.sizeHint = lambda : QSize(5*self.dpi, 3*self.dpi)
		self.maven.trace_filter._plot(self.fig,self.ax)


		combo_options = ['Nothing','Remove Traces']
		[combo_options.append('Classify %d'%(i)) for i in range(10)]
		self.class_actions = []
		for i in range(5):
			self.class_actions.append(QComboBox())
			self.class_actions[i].addItems(combo_options)
			self.class_actions[i].setCurrentIndex(0)
		self.model_labels = ['Dead','Low SBR, Bleach', 'High SBR, Bleach','Low SBR, No bleach', 'High SBR, No bleach']
		self.copy_params()

		self.button_defaults = QPushButton('Defaults')
		self.button_calculate = QPushButton('Calculate')
		self.button_process = QPushButton('Process')

		#####################################
		#####################################

		wid1 = QWidget()
		wid1.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))
		vbox = QVBoxLayout()
		vbox.addWidget(QLabel('Parameters'))
		qw = QWidget()
		fbox = QFormLayout()
		fbox.addRow("Low SBR",self.le_low_sbr)
		fbox.addRow("High SBR",self.le_high_sbr)
		fbox.addRow("Min. frames (dead)",self.le_min_frames)
		fbox.addRow("Skip frames (start)",self.le_skip_frames)
		fbox.addRow("Source Data",self.combo_data)
		qw.setLayout(fbox)
		vbox.addWidget(qw)

		qw = QWidget()
		hbox = QHBoxLayout()
		hbox.addWidget(self.button_defaults)
		hbox.addWidget(self.button_calculate)
		qw.setLayout(hbox)
		vbox.addWidget(qw)
		vbox.addStretch(1)
		wid1.setLayout(vbox)

		#####################################
		#####################################

		wid2 = QWidget()
		wid2.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
		vbox = QVBoxLayout()
		vbox.addWidget(QLabel('Classifications'))
		vbox.addWidget(self.canvas)
		# vbox.addWidget(self.toolbar)
		# vbox.addStretch(1)
		vbox.addWidget(self.label_proportions)
		wid2.setLayout(vbox)

		#####################################
		#####################################

		wid3 = QWidget()
		wid3.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))
		vbox = QVBoxLayout()
		qw = QWidget()
		fbox = QFormLayout()
		for i in range(len(self.model_labels)):
			fbox.addRow(self.model_labels[i],self.class_actions[i])
		qw.setLayout(fbox)
		vbox.addWidget(qw)
		vbox.addWidget(self.button_process)
		wid3.setLayout(vbox)

		#####################################
		#####################################

		tothbox = QHBoxLayout()
		tothbox.addWidget(wid1)
		tothbox.addWidget(wid2)
		tothbox.addWidget(wid3)
		cqw.setLayout(tothbox)

		#####################################
		#####################################

		self.button_defaults.clicked.connect(self.reset_defaults)
		self.button_calculate.clicked.connect(self.calculate_model)
		self.button_process.clicked.connect(self.process_filters)

		self.setCentralWidget(cqw)
		# self.resize(400,300)
		self.show()

	def reset_defaults(self,event):
		self.maven.trace_filter.reset_defaults()
		self.copy_params()
	def copy_params(self):
		self.le_low_sbr.setText('%.1f'%(self.maven.trace_filter.low_sbr))
		self.le_high_sbr.setText('%.1f'%(self.maven.trace_filter.high_sbr))
		self.le_min_frames.setText('%d'%(self.maven.trace_filter.min_frames))
		self.le_skip_frames.setText('%d'%(self.maven.trace_filter.skip_frames))
		self.combo_data.setCurrentIndex(self.maven.trace_filter.mode)
		for w in [self.le_low_sbr,self.le_high_sbr,self.le_min_frames,self.le_skip_frames,self.combo_data]:
			w.repaint()
	def push_params(self):
		self.maven.trace_filter.low_sbr = float(self.le_low_sbr.text())
		self.maven.trace_filter.high_sbr = float(self.le_high_sbr.text())
		self.maven.trace_filter.min_frames = int(self.le_min_frames.text())
		self.maven.trace_filter.skip_frames = int(self.le_skip_frames.text())
		self.maven.trace_filter.mode = self.combo_data.currentIndex()


	def calculate_model(self,event):
		self.push_params()
		self.maven.trace_filter.calculate_model()
		self.ax.cla()
		self.maven.trace_filter._plot(self.fig,self.ax)
		self.canvas.draw()
		self.canvas.repaint()

	def process_filters(self,event):
		if np.any([ml is None for ml in self.maven.trace_filter.model_lists]):
			logger.error('trace filter: need to calculate first')
			return

		wipe_flag = False
		new_classes = np.copy(self.maven.data.classes)
		keep = np.ones(new_classes.size,dtype='bool')
		for i in range(len(self.maven.trace_filter.model_lists)):
			ml = self.maven.trace_filter.model_lists[i]
			ci = self.class_actions[i].currentIndex()

			if ci == 1: ## remove
				keep[ml] = False
				wipe_flag = True
			if ci > 1: ## class = ci-2
				new_classes[ml] = ci-2

		old = self.maven.data.classes.copy()
		self.maven.data.classes = new_classes.copy()
		if self.maven.cull.cull_remove_traces(keep):
			msg = "Filtered traces: kept %d out of %d = %f"%(keep.sum(),keep.size,keep.sum()/float(keep.size))
			logger.info(msg)
		else:
			self.maven.data.classes = old.copy()
		# self.maven.data_update.emit()

		if wipe_flag:
			self.maven.trace_filter.model_lists = [None]*5
			self.ax.cla()
			self.label_proportions.setText("")


		self.calculate_model(None)
		self.maven.emit_data_update()
