import numpy as np
import logging
logger = logging.getLogger(__name__)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from PyQt5.QtWidgets import (QDialog, QPushButton, QStyleFactory,
							 QGridLayout, QLabel, QLayout, QPlainTextEdit,
							 QComboBox, QCheckBox, QGroupBox)

class _dwell_dialog(QDialog):
	'''
	Defines and sets the layout for the base dialog box that can be called
	for specific modelling methods.
	'''
	def __init__(self,gui,model):
		super(QDialog, self).__init__(parent=gui)
		self.gui = gui

		self.setWindowTitle("Run Dwell Analysis")
		self.setWindowModality(Qt.NonModal)

		self.grid = QGridLayout()
		self.setLayout(self.grid)
		self.grid.setSizeConstraint(QLayout.SetFixedSize)

		#self.setStyle(QStyleFactory.create('Fusion'))
		from .stylesheet import ui_stylesheet
		self.setStyleSheet(ui_stylesheet)

		# Pre-calc model groupbox
		groupbox1 = QGroupBox("State model")
		grid1 = QGridLayout()

		grid1.addWidget(QLabel('Model Type = '), 0,0)
		self.model_label = QLabel(model.type)
		grid1.addWidget(self.model_label, 0,1)

		grid1.addWidget(QLabel('Transition Matrix = '), 1,0)
		self.model_tm = QLabel(str(model.tmatrix))
		grid1.addWidget(self.model_tm, 1,1)

		self.add_push_button('Change Active', grid1, [2,1], fxn = lambda : self.model_change())
		groupbox1.setLayout(grid1)
		self.grid.addWidget(groupbox1, 0, 0)

		# Dwell times groupbox
		groupbox2 = QGroupBox("Dwell times")
		grid2 = QGridLayout()
		if model.dwells is None:
			self.dwell_label = QLabel('Dwells not calculated')
		else:
			self.dwell_label = QLabel('Dwells calculated')
		grid2.addWidget(self.dwell_label, 0,0)

		self.add_push_button('Calculate', grid2, [1,0], fxn = lambda : self.add_dwells())
		self.add_push_button('Plot', grid2, [1,1], fxn = lambda : self.plot_dwells())
		self.first_check = self.add_check_box("Include first", grid2, [2,0])
		groupbox2.setLayout(grid2)
		self.grid.addWidget(groupbox2, 1, 0)

		# Rate analysis groupbox
		groupbox3 = QGroupBox("Rate Analysis")
		grid3 = QGridLayout()

		state_items = [str(i) for i in range(model.nstates)]
		self.state_combo = self.add_combo_box('Active State =', grid3, [0,0], state_items)
		self.active_state = int(self.state_combo.currentText())

		func_items = ['Single Exponential','Double Exponential','Triple Exponential','Stretched Exponential']
		self.func_combo = self.add_combo_box('Rate function =', grid3, [1,0], func_items)
		self.active_func = self.func_combo.currentText()
		self.func_combo.activated.connect(lambda: self.func_change())
		self.add_push_button("T-matrix", grid3, [3,0], fxn = lambda : self.run_tmatrix())
		self.add_push_button("Run", grid3, [3,1], fxn = lambda : self.run_dwell_analysis())
		self.fixA_check = self.add_check_box("Enforce Normalisation", grid3, [4,1])

		groupbox3.setLayout(grid3)
		self.grid.addWidget(groupbox3, 2, 0)

		# Result groupbox
		groupbox4 = QGroupBox("Results")
		grid4 = QGridLayout()

		state_str = "State = {}\n".format(self.active_state)
		rate_type_str = "Rate_type = {}\n".format(model.rate_type)

		if model.rate_type == "Transition Matrix":
			rate = model.rates[self.active_state]
			rate_str = "Rates = \n {} \n".format(str(rate))

		elif model.rate_type == "Dwell Analysis":
			if self.active_state in model.rates:
				rate = model.rates[self.active_state]
				rate_str = "Rates = \n {} \n".format(str(rate['ks']))
				rate_str += "Error = \n {} \n".format(str(rate['error'][0]))
				rate_str += "Coefficients = \n {} \n".format(str(rate['As']))
				rate_str += "Error = \n {} \n".format(str(rate['error'][-1]))
				if 'betas' in rate:
					rate_str += "Betas = \n {} \n".format(str(rate['betas']))
					rate_str += "Error = \n {} \n".format(str(rate['error'][1]))
				rate_str += "R2 of fit = \n {} \n".format(str(rate['R2']))
			else:
				rate_str = "Rates = N/A"
		else:
			rate_str = ""


		disp_str =  state_str + rate_type_str + rate_str
		self.te = QPlainTextEdit(parent=self)
		self.te.setPlainText(disp_str)
		#te.verticalScrollBar().setValue(te.verticalScrollBar().maximum())
		self.te.setReadOnly(True)
		grid4.addWidget(self.te, 0,0)


		groupbox4.setLayout(grid4)
		self.grid.addWidget(groupbox4, 0, 1, 3, 1)

		self.state_combo.activated.connect(lambda: self.state_change())


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

	def add_check_box(self, label, grid, position, default=False):
		#check = flag_check(label,default)
		check = QCheckBox(label, self)
		check.setChecked(default)
		grid.addWidget(check, position[0], position[1])

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
				rate_str += "Error = \n {} \n".format(str(rate['error'][0]))
				rate_str += "Coefficients = \n {} \n".format(str(rate['As']))
				rate_str += "Error = \n {} \n".format(str(rate['error'][-1]))
				if 'betas' in rate:
					rate_str += "Betas = \n {} \n".format(str(rate['betas']))
					rate_str += "Error = \n {} \n".format(str(rate['error'][1]))
				rate_str += "R2 of fit = \n {} \n".format(str(rate['R2']))
			else:
				rate_str = "Rates = N/A"
		else:
			rate_str = ""

		disp_str =  state_str+rate_type_str+rate_str
		self.te.setPlainText(disp_str)
		self.te.setReadOnly(True)

	def model_change(self):
		#from .modeler.ui_modeler import change_model

		self.gui.change_model()
		self.update_model()
		self.update_dwell()
		self.update_analysis()
		self.update_result()

	def state_change(self):
		self.active_state = int(self.state_combo.currentText())
		self.update_result()

	def func_change(self):
		self.active_func = self.func_combo.currentText()

	def add_dwells(self):
		from ..controllers.modeler.dwells import calculate_dwells
		model = self.gui.maven.modeler.model
		first_flag = self.first_check.isChecked()
		calculate_dwells(model,first_flag)
		self.update_dwell()

	def plot_dwells(self):
		model = self.gui.maven.modeler.model
		if model.dwells is None:
			return
		from .ui_analysisplots import popplot_container
		popplot_container(self.gui,self.gui.maven.plots.survival_dwell)

	def run_dwell_analysis(self):
		model = self.gui.maven.modeler.model
		if model.dwells is None:
			return

		self.gui.maven.modeler.run_dwell_analysis(self.active_func, self.active_state,
						 				 fix_A = self.fixA_check.isChecked())
		self.update_result()

	def run_tmatrix(self):
		self.gui.maven.modeler.run_tmatrix()
		self.update_result()

def dialog_dwell_analysis(gui,model):
	dwell_dialog = _dwell_dialog(gui, model)
	gui.dwell_dialog = dwell_dialog

def launch_dwell_analysis(gui):
	logger.info('Launching Dwell Analysis')
	if gui.maven.modeler.model is None:
		pass
	else:
		model = gui.maven.modeler.model
		dialog_dwell_analysis(gui,model)
		gui.dwell_dialog.start()

#class flag_check(QCheckBox):
#	def __init__(self, label, default=False):
#		super(QCheckBox, self).__init__(label)
#		self.setChecked(default)

