import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from PyQt5.QtWidgets import (QDialog, QMainWindow, QPushButton, QProgressBar, QStyleFactory,
								QGridLayout, QLabel, QSpinBox, QLayout, QLineEdit, QPlainTextEdit,
								QComboBox, QCheckBox, QGroupBox, QStyleFactory)

class _model_dialog(QDialog):
	'''
	Defines and sets the layout for the base dialog box that can be called
	for specific modelling methods.
	'''
	def __init__(self,gui):
		super(QDialog, self).__init__()
		self.gui = gui

		self.setWindowTitle("Run Modeling")
		self.setWindowModality(Qt.ApplicationModal)

		self.grid = QGridLayout()
		self.run = QPushButton("Run", self)
		self.grid.addWidget(self.run, 5, 0)
		self.setLayout(self.grid)
		self.grid.setSizeConstraint(QLayout.SetFixedSize)

		self.setStyle(QStyleFactory.create('Fusion'))
		from ..stylesheet import ui_stylesheet
		self.setStyleSheet(ui_stylesheet)

	def add_spin_box(self, label, grid, position, pref= None, range = None):

		grid.addWidget(QLabel(label), position[0], position[1])
		spin = pref_spin_box(pref)

		if range is None:
			spin.setRange(1,100000)
		else:
			spin.setRange(range[0], range[1])

		spin.set_value(self.gui)
		grid.addWidget(spin, position[0], position[1] + 1)
		spin.valueChanged.connect(lambda: spin.spin_value_change(self.gui))

		return spin

	def add_line_edit(self, label, grid, position, pref=None, validate = None, range = None):

		grid.addWidget(QLabel(label), position[0], position[1])
		line = pref_line_edit(pref)

		'''
		Doesn't work. Check Validators
		'''
		if validate == "int":
			if range is None:
				line.setValidator(QIntValidator(1, 100000))
			else:
				line.setValidator(QIntValidator(range[0], range[1]))
		elif validate == "double":
			if range is None:
				line.setValidator(QDoubleValidator(0, 100, 12))
			else:
				line.setValidator(QDoubleValidator(range[0], range[1], range[2]))
		else:
			pass

		line.set_value(self.gui)
		grid.addWidget(line, position[0], position[1] + 1)
		line.editingFinished.connect(lambda: line.line_value_change(self.gui))
		return line

	def add_combo_box(self, label, grid, position, items):

		grid.addWidget(QLabel(label), position[0], position[1])
		combo = QComboBox(self)

		for thing in items:
			combo.addItem(thing)

		combo.setCurrentIndex(0)
		grid.addWidget(combo, position[0], position[1] + 1)
		return combo

	def start(self):
		self.exec_()

class pref_spin_box(QSpinBox):
	def __init__(self, pref=None):
		super(QSpinBox, self).__init__()
		self.pref = pref

	def set_value(self, gui):
		if self.pref in gui.maven.prefs.keys():
			self.setValue(gui.maven.prefs[self.pref])
		else:
			pass

	def spin_value_change(self, gui):
		if self.pref in gui.maven.prefs.keys():
			gui.maven.prefs[self.pref] = self.value()
		else:
			pass

class pref_line_edit(QLineEdit):
	def __init__(self, pref =None):
		super(QLineEdit, self).__init__()
		self.pref = pref

	def set_value(self, gui):
		if self.pref in gui.maven.prefs.keys():
			self.setText(str(gui.maven.prefs[self.pref]))
		else:
			pass

	def line_value_change(self, gui):
		if self.pref in gui.maven.prefs.keys():
			gui.maven.prefs[self.pref] = float(self.text())
		else:
			pass

def dialog_vbhmm(gui,fxn,title,model_selection=False):
	model_dialog = _model_dialog(gui)

	groupbox1 = QGroupBox("{} algorithm parameters".format(title))
	grid1 = QGridLayout()
	model_dialog.add_spin_box("CPUs:", grid1, [0,0], pref='ncpu')
	model_dialog.add_spin_box("Restarts:", grid1, [1,0], pref='modeler.nrestarts')
	model_dialog.add_spin_box("Max iterations:", grid1, [2,0], pref='modeler.maxiters')
	model_dialog.add_line_edit("Convergence:", grid1, [3,0], pref='modeler.converge', validate = "double")
	groupbox1.setLayout(grid1)
	model_dialog.grid.addWidget(groupbox1, 1, 0)

	groupbox2 = QGroupBox("{} model parameters".format(title))
	grid2 = QGridLayout()
	if model_selection:
		model_dialog.nstates_min = model_dialog.add_spin_box("Number of states: (low)", grid2, [0,0])
		model_dialog.nstates_max = model_dialog.add_spin_box("(high)", grid2, [0,2])
	else:
		model_dialog.nstates = model_dialog.add_spin_box("Number of states:", grid2, [0,0])
	model_dialog.add_line_edit("Prior beta:", grid2, [1,0], pref='modeler.vbhmm.prior.beta', validate = 'double')
	model_dialog.add_line_edit("Prior a:", grid2, [2,0], pref='modeler.vbhmm.prior.a', validate = 'double')
	model_dialog.add_line_edit("Prior b:", grid2, [3,0], pref='modeler.vbhmm.prior.b', validate = 'double')
	model_dialog.add_line_edit("Prior pi:", grid2, [4,0], pref='modeler.vbhmm.prior.pi', validate = 'double')
	model_dialog.add_line_edit("Prior alpha:", grid2, [5,0], pref='modeler.vbhmm.prior.alpha', validate = 'double')
	groupbox2.setLayout(grid2)
	model_dialog.grid.addWidget(groupbox2, 0, 0)

	gui.model_dialog = model_dialog
	gui.model_dialog.run.clicked.connect(fxn)

def dialog_vbgmm(gui,fxn,title,model_selection=False):
	model_dialog = _model_dialog(gui)

	groupbox1 = QGroupBox("{} algorithm parameters".format(title))
	grid1 = QGridLayout()
	model_dialog.add_spin_box("CPUs:", grid1, [0,0], pref='ncpu')
	model_dialog.add_spin_box("Restarts:", grid1, [1,0], pref='modeler.nrestarts')
	model_dialog.add_spin_box("Max iterations:", grid1, [2,0], pref='modeler.maxiters')
	model_dialog.add_line_edit("Convergence:", grid1, [3,0], pref='modeler.converge', validate = "double")
	groupbox1.setLayout(grid1)
	model_dialog.grid.addWidget(groupbox1, 1, 0)

	groupbox2 = QGroupBox("{} model parameters".format(title))
	grid2 = QGridLayout()
	if model_selection:
		model_dialog.nstates_min = model_dialog.add_spin_box("Number of states: (low)", grid2, [0,0])
		model_dialog.nstates_max = model_dialog.add_spin_box("(high)", grid2, [0,2])
	else:
		model_dialog.nstates = model_dialog.add_spin_box("Number of states:", grid2, [0,0])
	model_dialog.add_line_edit("Prior beta:", grid2, [1,0], pref='modeler.vbgmm.prior.beta', validate = 'double')
	model_dialog.add_line_edit("Prior a:", grid2, [2,0], pref='modeler.vbgmm.prior.a', validate = 'double')
	model_dialog.add_line_edit("Prior b:", grid2, [3,0], pref='modeler.vbgmm.prior.b', validate = 'double')
	model_dialog.add_line_edit("Prior pi:", grid2, [4,0], pref='modeler.vbgmm.prior.pi', validate = 'double')
	groupbox2.setLayout(grid2)
	model_dialog.grid.addWidget(groupbox2, 0, 0)

	gui.model_dialog = model_dialog
	gui.model_dialog.run.clicked.connect(fxn)

def dialog_vbgmm_vbhmm(gui,fxn,title,model_selection=False):
	model_dialog = _model_dialog(gui)

	groupbox1 = QGroupBox("{} algorithm parameters".format(title))
	grid1 = QGridLayout()
	model_dialog.add_spin_box("CPUs:", grid1, [0,0], pref='ncpu')
	model_dialog.add_spin_box("Restarts:", grid1, [1,0], pref='modeler.nrestarts')
	model_dialog.add_spin_box("Max iterations:", grid1, [2,0], pref='modeler.maxiters')
	model_dialog.add_line_edit("Convergence:", grid1, [3,0], pref='modeler.converge', validate = "double")
	groupbox1.setLayout(grid1)
	model_dialog.grid.addWidget(groupbox1, 1, 0)

	groupbox2 = QGroupBox("VB GMM model parameters".format(title))
	grid2 = QGridLayout()
	if model_selection:
		model_dialog.nstates_min = model_dialog.add_spin_box("Number of states: (low)", grid2, [0,0])
		model_dialog.nstates_max = model_dialog.add_spin_box("(high)", grid2, [0,2])
	else:
		model_dialog.nstates = model_dialog.add_spin_box("Number of states:", grid2, [0,0])
	model_dialog.add_line_edit("Prior beta:", grid2, [1,0], pref='modeler.vbgmm.prior.beta', validate = 'double')
	model_dialog.add_line_edit("Prior a:", grid2, [2,0], pref='modeler.vbgmm.prior.a', validate = 'double')
	model_dialog.add_line_edit("Prior b:", grid2, [3,0], pref='modeler.vbgmm.prior.b', validate = 'double')
	model_dialog.add_line_edit("Prior pi:", grid2, [4,0], pref='modeler.vbgmm.prior.pi', validate = 'double')
	groupbox2.setLayout(grid2)
	model_dialog.grid.addWidget(groupbox2, 0, 0)

	groupbox3 = QGroupBox("VB HMM model parameters".format(title))
	grid3 = QGridLayout()
	model_dialog.add_line_edit("Prior beta:", grid3, [1,0], pref='modeler.vbhmm.prior.beta', validate = 'double')
	model_dialog.add_line_edit("Prior a:", grid3, [2,0], pref='modeler.vbhmm.prior.a', validate = 'double')
	model_dialog.add_line_edit("Prior b:", grid3, [3,0], pref='modeler.vbhmm.prior.b', validate = 'double')
	model_dialog.add_line_edit("Prior pi:", grid3, [4,0], pref='modeler.vbhmm.prior.pi', validate = 'double')
	model_dialog.add_line_edit("Prior alpha:", grid3, [5,0], pref='modeler.vbhmm.prior.alpha', validate = 'double')
	groupbox3.setLayout(grid3)
	model_dialog.grid.addWidget(groupbox3, 0, 1)

	gui.model_dialog = model_dialog
	gui.model_dialog.run.clicked.connect(fxn)

def dialog_vbconhmm(gui,fxn,model_selection=False):
	model_dialog = _model_dialog(gui)

	groupbox1 = QGroupBox("Consensus vbHMM algorithm parameters")
	grid1 = QGridLayout()
	model_dialog.add_spin_box("CPUs:", grid1, [0,0], pref='ncpu')
	model_dialog.add_spin_box("Restarts:", grid1, [1,0], pref='modeler.nrestarts')
	model_dialog.add_spin_box("Max iterations:", grid1, [2,0], pref='modeler.maxiters')
	model_dialog.add_line_edit("Convergence:", grid1, [3,0], pref='modeler.converge', validate = "double")
	groupbox1.setLayout(grid1)
	model_dialog.grid.addWidget(groupbox1, 1, 0)

	groupbox2 = QGroupBox("Consensus vbHMM model parameters")
	grid2 = QGridLayout()
	if model_selection:
		model_dialog.nstates_min = model_dialog.add_spin_box("Number of states: (low)", grid2, [0,0])
		model_dialog.nstates_max = model_dialog.add_spin_box("(high)", grid2, [0,2])
		model_dialog.nstates_max.setValue(6)
	else:
		model_dialog.nstates = model_dialog.add_spin_box("Number of states:", grid2, [0,0])
	model_dialog.add_line_edit("Prior beta:", grid2, [1,0], pref='modeler.vbconhmm.prior.beta', validate = 'double')
	model_dialog.add_line_edit("Prior a:", grid2, [2,0], pref='modeler.vbconhmm.prior.a', validate = 'double')
	model_dialog.add_line_edit("Prior b:", grid2, [3,0], pref='modeler.vbconhmm.prior.b', validate = 'double')
	model_dialog.add_line_edit("Prior pi:", grid2, [4,0], pref='modeler.vbconhmm.prior.pi', validate = 'double')
	model_dialog.add_line_edit("Prior alpha:", grid2, [5,0], pref='modeler.vbconhmm.prior.alpha', validate = 'double')
	groupbox2.setLayout(grid2)
	model_dialog.grid.addWidget(groupbox2, 0, 0)

	gui.model_dialog = model_dialog
	gui.model_dialog.run.clicked.connect(fxn)

def dialog_mlmm(gui,fxn,title):
	model_dialog = _model_dialog(gui)

	groupbox1 = QGroupBox("{} algorithm parameters".format(title))
	grid1 = QGridLayout()
	model_dialog.add_spin_box("CPUs:", grid1, [0,0], pref='ncpu')
	model_dialog.add_spin_box("Restarts:", grid1, [1,0], pref='modeler.nrestarts')
	model_dialog.add_spin_box("Max iterations:", grid1, [2,0], pref='modeler.maxiters')
	model_dialog.add_line_edit("Convergence:", grid1, [3,0], pref='modeler.converge', validate = "double")
	groupbox1.setLayout(grid1)
	model_dialog.grid.addWidget(groupbox1, 1, 0)

	groupbox2 = QGroupBox("{} model parameters".format(title))
	grid2 = QGridLayout()
	model_dialog.nstates = model_dialog.add_spin_box("Number of states:", grid2, [0,0])
	groupbox2.setLayout(grid2)
	model_dialog.grid.addWidget(groupbox2, 0, 0)

	gui.model_dialog = model_dialog
	gui.model_dialog.run.clicked.connect(fxn)

def dialog_kmeans(gui,fxn):
	model_dialog = _model_dialog(gui)

	'''
	groupbox1 = QGroupBox("K-means algorithm parameters")
	grid1 = QGridLayout()
	# model_dialog.combo = model_dialog.add_combo_box("Run on:", grid1, [0,0], ['Raw'])
	groupbox1.setLayout(grid1)
	model_dialog.grid.addWidget(groupbox1, 1, 0)
	'''

	groupbox2 = QGroupBox("K-means model parameters")
	grid2 = QGridLayout()
	model_dialog.nstates = model_dialog.add_spin_box("Number of states:", grid2, [0,0])
	groupbox2.setLayout(grid2)
	model_dialog.grid.addWidget(groupbox2, 0, 0)

	gui.model_dialog = model_dialog
	gui.model_dialog.run.clicked.connect(fxn)

def dialog_threshold(gui,fxn):
	model_dialog = _model_dialog(gui)

	#groupbox1 = QGroupBox("Threshold algorithm parameters")
	#grid1 = QGridLayout()
	# model_dialog.combo = model_dialog.add_combo_box("Run on:", grid1, [0,0], ['Raw'])
	#groupbox1.setLayout(grid1)
	#model_dialog.grid.addWidget(groupbox1, 1, 0)

	groupbox2 = QGroupBox("Threshold model parameters")
	grid2 = QGridLayout()
	model_dialog.threshold = model_dialog.add_line_edit("Threshold:", grid2, [1,0],validate = 'double', range= [0,1, 4])
	groupbox2.setLayout(grid2)
	model_dialog.grid.addWidget(groupbox2, 0, 0)

	gui.model_dialog = model_dialog
	gui.model_dialog.run.clicked.connect(fxn)

def dialog_ebhmm(gui,fxn,model_selection=False):
	model_dialog = _model_dialog(gui)

	groupbox1 = QGroupBox("ebHMM algorithm parameters")
	grid1 = QGridLayout()
	model_dialog.add_spin_box("CPUs:", grid1, [0,0], pref='ncpu')
	model_dialog.add_spin_box("Restarts:", grid1, [1,0], pref='modeler.nrestarts')
	model_dialog.add_spin_box("Max iterations:", grid1, [2,0], pref='modeler.maxiters')
	model_dialog.add_line_edit("Convergence:", grid1, [3,0], pref='modeler.converge', validate = "double")
	groupbox1.setLayout(grid1)
	model_dialog.grid.addWidget(groupbox1, 1, 0)

	groupbox2 = QGroupBox("ebHMM model parameters")
	grid2 = QGridLayout()
	if model_selection:
		model_dialog.nstates_min = model_dialog.add_spin_box("Number of states: (low)", grid2, [0,0])
		model_dialog.nstates_max = model_dialog.add_spin_box("(high)", grid2, [0,2])
		model_dialog.nstates_max.setValue(6)
	else:
		model_dialog.nstates = model_dialog.add_spin_box("Number of states:", grid2, [0,0])
	model_dialog.add_line_edit("Prior beta:", grid2, [1,0], pref='modeler.ebhmm.prior.beta', validate = 'double')
	model_dialog.add_line_edit("Prior a:", grid2, [2,0], pref='modeler.ebhmm.prior.a', validate = 'double')
	model_dialog.add_line_edit("Prior b:", grid2, [3,0], pref='modeler.ebhmm.prior.b', validate = 'double')
	model_dialog.add_line_edit("Prior pi:", grid2, [4,0], pref='modeler.ebhmm.prior.pi', validate = 'double')
	model_dialog.add_line_edit("Prior alpha:", grid2, [5,0], pref='modeler.ebhmm.prior.alpha', validate = 'double')
	groupbox2.setLayout(grid2)
	model_dialog.grid.addWidget(groupbox2, 0, 0)

	gui.model_dialog = model_dialog
	gui.model_dialog.run.clicked.connect(fxn)
