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


class _dwell_dialog(QDialog):
	'''
	Defines and sets the layout for the base dialog box that can be called
	for specific modelling methods.
	'''
	def __init__(self,gui):
		super(QDialog, self).__init__()
		self.gui = gui

		self.setWindowTitle("Run Dwell Analysis")
		self.setWindowModality(Qt.NonModal)

		self.grid = QGridLayout()
		#self.run = QPushButton("Run", self)
		#self.grid.addWidget(self.run, 5, 0)
		self.setLayout(self.grid)
		self.grid.setSizeConstraint(QLayout.SetFixedSize)

		self.setStyle(QStyleFactory.create('Fusion'))
		from ..stylesheet import ui_stylesheet
		self.setStyleSheet(ui_stylesheet)

	def add_combo_box(self, label, grid, position, items):

		grid.addWidget(QLabel(label), position[0], position[1])
		combo = QComboBox(self)

		for thing in items:
			combo.addItem(thing)

		combo.setCurrentIndex(0)
		grid.addWidget(combo, position[0], position[1] + 1)
		return combo

	def add_push_button(self, label, grid, position, fxn):
		button = QPushButton(label, self)
		grid.addWidget(button, position[0], position[1])
		button.clicked.connect(fxn)

		return button

	def add_check_box(self, label, grid, position, fxn, default):
		check = QCheckBox(label, self)
		grid.addWidget(check, position[0], position[1])
		check.setChecked(default)
		check.stateChanged.connect(fxn)

		return check

	def start(self):
		self.show()

	def update_model(self):
		model = self.gui.maven.modeler.model

		self.model_label.setText(model.type)
		self.model_tm.setText(str(model.tmatrix))

	def update_dwell(self):
		model = self.gui.maven.modeler.model
		if model.dwells is None:
			self.dwell_label.setText('Dwells not calculated')
		else:
			self.dwell_label.setText('Dwells calculated')

	def update_analysis(self):
		model = self.gui.maven.modeler.model
		self.state_combo.clear()

		items = [str(i) for i in range(model.nstates)]
		for thing in items:
			self.state_combo.addItem(thing)

		self.state_combo.setCurrentIndex(0)


	def update_result(self):
		model = self.gui.maven.modeler.model
		state_str = "State = {}\n".format(self.active_state)
		rate_type_str = "Rate type = {}\n".format(model.rate_type)

		if model.rate_type == "Transition Matrix":
			rate = model.rates[self.active_state]
			rate_str = "Rates = \n {} \n".format(str(rate))
		elif model.rate_type == "Dwell Analysis":
			if self.active_state in model.rates:
				rate = model.rates[self.active_state]
				rate_str = "Rates = \n {} \n".format(str(rate['ks']))
				rate_str += "Coefficients = \n {} \n".format(str(rate['As']))
				if 'betas' in rate:
					rate_str += "Betas = \n {} \n".format(str(rate['betas']))
			else:
				rate_str = "Rates = N/A"
		else:
			rate_str = ""

		disp_str =  state_str+rate_type_str+rate_str
		self.te.setPlainText(disp_str)
		self.te.setReadOnly(True)

	def model_change(self):
		from .ui_modeler import change_model

		self.gui.change_model()
		self.update_model()
		self.update_dwell()
		self.update_analysis()
		self.update_result()

	def fixA_change(self):
		if self.fixA_check.isChecked():
			self.fixA = True
		else:
			self.fixA = False

	def state_change(self):
		self.active_state = int(self.state_combo.currentText())
		self.update_result()

	def func_change(self):
		self.active_func = self.func_combo.currentText()

	def add_dwells(self):
		from ...controllers.modeler.dwells import calculate_dwells
		model = self.gui.maven.modeler.model
		calculate_dwells(model)
		self.update_dwell()

	def plot_dwells(self):
		model = self.gui.maven.modeler.model
		if model.dwells is None:
			return
		from ..ui_analysisplots import popplot_container
		popplot_container(self.gui,self.gui.maven.plots.survival_dwell)

	def run_dwell_analysis(self):
		model = self.gui.maven.modeler.model
		if model.dwells is None:
			return

		self.gui.maven.modeler.run_fret_dwell_analysis(self.active_func, self.active_state, fix_A = self.fixA)
		self.update_result()

	def run_tmatrix(self):
		self.gui.maven.modeler.run_fret_tmatrix()
		self.update_result()

def dialog_dwell_analysis(gui,model):
	dwell_dialog = _dwell_dialog(gui)

	# Pre-calc model groupbox
	groupbox1 = QGroupBox("State model")
	grid1 = QGridLayout()

	grid1.addWidget(QLabel('Model Type = '), 0,0)
	dwell_dialog.model_label = QLabel(model.type)
	grid1.addWidget(dwell_dialog.model_label, 0,1)

	grid1.addWidget(QLabel('Transition Matrix = '), 1,0)
	dwell_dialog.model_tm = QLabel(str(model.tmatrix))
	grid1.addWidget(dwell_dialog.model_tm, 1,1)

	dwell_dialog.add_push_button('Change Active', grid1, [2,1], fxn = lambda : dwell_dialog.model_change())
	groupbox1.setLayout(grid1)
	dwell_dialog.grid.addWidget(groupbox1, 0, 0)

	# Dwell times groupbox
	groupbox2 = QGroupBox("Dwell times")
	grid2 = QGridLayout()
	if model.dwells is None:
		dwell_dialog.dwell_label = QLabel('Dwells not calculated')
	else:
		dwell_dialog.dwell_label = QLabel('Dwells calculated')
	grid2.addWidget(dwell_dialog.dwell_label, 0,0)

	dwell_dialog.add_push_button('Calculate', grid2, [1,0], fxn = lambda : dwell_dialog.add_dwells())
	dwell_dialog.add_push_button('Plot', grid2, [1,1], fxn = lambda : dwell_dialog.plot_dwells())
	groupbox2.setLayout(grid2)
	dwell_dialog.grid.addWidget(groupbox2, 1, 0)

	# Rate analysis groupbox
	groupbox3 = QGroupBox("Rate Analysis")
	grid3 = QGridLayout()

	state_items = [str(i) for i in range(model.nstates)]
	dwell_dialog.state_combo = dwell_dialog.add_combo_box('Active State =', grid3, [0,0], state_items)
	dwell_dialog.active_state = int(dwell_dialog.state_combo.currentText())

	func_items = ['Single Exponential','Double Exponential','Triple Exponential','Stretched Exponential']
	dwell_dialog.func_combo = dwell_dialog.add_combo_box('Rate function =', grid3, [1,0], func_items)
	dwell_dialog.active_func = dwell_dialog.func_combo.currentText()
	dwell_dialog.func_combo.activated.connect(lambda: dwell_dialog.func_change())
	dwell_dialog.add_push_button("T-matrix", grid3, [3,0], fxn = lambda : dwell_dialog.run_tmatrix())
	dwell_dialog.add_push_button("Run", grid3, [3,1], fxn = lambda : dwell_dialog.run_dwell_analysis())
	dwell_dialog.fixA_check = dwell_dialog.add_check_box("Enforce Normalisation", grid3, [4,1], fxn = lambda: dwell_dialog.fixA_change(), default = False)
	dwell_dialog.fixA = False

	groupbox3.setLayout(grid3)
	dwell_dialog.grid.addWidget(groupbox3, 2, 0)

	# Result groupbox
	groupbox4 = QGroupBox("Results")
	grid4 = QGridLayout()

	state_str = "State = {}\n".format(dwell_dialog.active_state)
	rate_type_str = "Rate_type = {}\n".format(model.rate_type)
	if model.rate_type == "Transition Matrix":
		rate = model.rates[dwell_dialog.active_state]
		rate_str = "Rates = \n {} \n".format(str(rate))
	elif model.rate_type == "Dwell Analysis":
		if dwell_dialog.active_state in model.rates:
			rate = model.rates[dwell_dialog.active_state]
			rate_str = "Rates = \n {} \n".format(str(rate['ks']))
			rate_str += "Coefficients = \n {} \n".format(str(rate['As']))
			if 'betas' in rate:
				rate_str += "Betas = \n {} \n".format(str(rate['betas']))
		else:
			rate_str = "Rates = N/A"
	else:
		rate_str = ""

	disp_str =  state_str + rate_type_str + rate_str
	dwell_dialog.te = QPlainTextEdit(parent=dwell_dialog)
	dwell_dialog.te.setPlainText(disp_str)
	#te.verticalScrollBar().setValue(te.verticalScrollBar().maximum())
	dwell_dialog.te.setReadOnly(True)
	grid4.addWidget(dwell_dialog.te, 0,0)

	groupbox4.setLayout(grid4)
	dwell_dialog.grid.addWidget(groupbox4, 0, 1, 3, 1)
	dwell_dialog.state_combo.activated.connect(lambda: dwell_dialog.state_change())

	gui.dwell_dialog = dwell_dialog
