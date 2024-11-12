import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from PyQt5.QtWidgets import (QDialog, QPushButton, QFileDialog, QStyleFactory,QGridLayout, QLabel, QSpinBox, QLayout, QLineEdit,QComboBox, QGroupBox, QStyleFactory)

class model_dialog_base(QDialog):
	'''
	Defines and sets the layout for the base dialog box that can be called
	for specific modelling methods.
	'''
	def __init__(self,gui):
		super(QDialog, self).__init__()
		self.gui = gui
		self.widgets = []

		self.grid = QGridLayout()
		self.run = QPushButton("Run", self)
		self.grid.addWidget(self.run, 5, 0)
		self.setLayout(self.grid)
		self.grid.setSizeConstraint(QLayout.SetFixedSize)

		self.setStyle(QStyleFactory.create('Fusion'))
		from ..stylesheet import ui_stylesheet
		self.setStyleSheet(ui_stylesheet)

		self.setWindowTitle("Run Modeling")
		self.setWindowModality(Qt.ApplicationModal)

	def update_prefs(self):
		for wp in self.widgets:
			widget,pref = wp
			if not wp is None:
				if type(widget) is QSpinBox:
					value = int(widget.value())
				elif type(widget) is QLineEdit:
					if type(widget.validator()) is QIntValidator:
						value = int(widget.text())
					elif type(widget.validator()) is QDoubleValidator:
						value = float(widget.text())
				elif type(widget) is QComboBox:
					if widget.currentText() in ['True','False']:
						value = widget.currentText() == 'True'
					else:
						value = widget.currentText()
				elif type(widget) is list:
					if widget[0] == 'biasd prior':
						_,dist,p1,p2 = widget
						self.gui.maven.prefs.__setitem__(pref+'.p1',float(p1.text()),quiet=True)
						self.gui.maven.prefs.__setitem__(pref+'.p2',float(p2.text()),quiet=True)
						self.gui.maven.prefs.__setitem__(pref+'.type',str(dist.currentText()),quiet=True)
						continue
					elif widget[0] == 'file':
						_,filename = widget
						self.gui.maven.prefs.__setitem__(pref,str(filename.text()),quiet=True)
						continue
				else:
					raise Exception('no value')
				# print(pref,value) ## for debugging
				self.gui.maven.prefs.__setitem__(pref,value,quiet=True)
		self.gui.maven.prefs.emit_changed()

	def add_file(self,label,grid,position,pref):
		qfname = QLineEdit(self.gui)
		qbutton = QPushButton(self.gui)
		grid.addWidget(QLabel(label),position[0],position[1]+0)
		grid.addWidget(qfname,position[0],position[1]+1)
		grid.addWidget(qbutton,position[0],position[1]+2)
		
		qfname.setText(str(self.gui.maven.prefs[pref]))
		def open_file_dialog():
			filename, _ = QFileDialog.getSaveFileName(self.gui, "Select HDF5 File", "", "HDF5 Files (.hdf5,.h5)")
			if filename: 
				if not filename.endswith((".hdf5",".h5")):
					filename += ".hdf5"
				qfname.setText(filename)
		qbutton.clicked.connect(open_file_dialog)
		self.widgets.append([['file',qfname],pref])

	def add_biasd_prior(self, label, grid, position, pref):
		dist = QComboBox(self.gui)
		for disttype in ['Normal','Uniform','Log-uniform']:
			dist.addItem(disttype)
		p1 = QLineEdit(self.gui)
		p2 = QLineEdit(self.gui)
		p1.setValidator(QDoubleValidator(-1e300, 1e300, 12))
		p2.setValidator(QDoubleValidator(-1e300, 1e300, 12))
		grid.addWidget(QLabel(label),position[0],position[1]+0)
		grid.addWidget(dist,position[0],position[1]+1)
		grid.addWidget(p1,position[0],position[1]+2)
		grid.addWidget(p2,position[0],position[1]+3)

		dist.setCurrentText(str(self.gui.maven.prefs[pref+'.type']))
		p1.setText(str(self.gui.maven.prefs[pref+'.p1']))
		p2.setText(str(self.gui.maven.prefs[pref+'.p2']))
		self.widgets.append([['biasd prior',dist,p1,p2],pref])

	def add_spin_box(self, label, grid, position, pref= None, range = None):
		grid.addWidget(QLabel(label), position[0], position[1])
		spin = QSpinBox(self.gui)
		if not pref is None:
			spin.setValue(self.gui.maven.prefs[pref])
		self.widgets.append([spin,pref])
		
		if range is None:
			spin.setRange(1,100000)
		else:
			spin.setRange(range[0], range[1])
		grid.addWidget(spin, position[0], position[1] + 1)	

	def add_line_edit(self, label, grid, position, pref=None, validate = None, range = None):
		grid.addWidget(QLabel(label), position[0], position[1])
		line = QLineEdit(self.gui)
		if not pref is None:
			line.setText(f'{self.gui.maven.prefs[pref]}')
		self.widgets.append([line,pref])

		if validate == "int":
			if range is None:
				line.setValidator(QIntValidator(1, 100000))
			else:
				line.setValidator(QIntValidator(range[0], range[1]))
		elif validate == "double":
			if range is None:
				line.setValidator(QDoubleValidator(-1e300, 1e300, 12))
			else:
				line.setValidator(QDoubleValidator(range[0], range[1], range[2]))
		grid.addWidget(line, position[0], position[1] + 1)

	def add_combo_box(self, label, grid, position, items, pref=None):
		grid.addWidget(QLabel(label), position[0], position[1])
		combo = QComboBox(self.gui)
		for thing in items:
			combo.addItem(thing)
		if not pref is None:
			combo.setCurrentText(str(self.gui.maven.prefs[pref]))
		self.widgets.append([combo,pref])
		grid.addWidget(combo, position[0], position[1] + 1)

	def start(self):
		self.exec_()

def dialog_threshold(gui,fxn):
	model_dialog = model_dialog_base(gui)

	groupbox2 = QGroupBox("Threshold model parameters")
	grid2 = QGridLayout()
	model_dialog.add_line_edit("Threshold:", grid2, [0,0],validate = 'double', pref='modeler.threshold')
	groupbox_data = QGroupBox("Data parameters")
	grid_data = QGridLayout()
	model_dialog.add_combo_box("Data Source:", grid_data, [0,0], ['FRET','Sum','0','1','2','3','Rel 0','Rel 1','Rel 2','Rel 3'],pref='modeler.dtype')
	model_dialog.add_combo_box("Clip Data?", grid_data, [1,0],['True','False'], pref='modeler.clip')

	groupbox2.setLayout(grid2)
	groupbox_data.setLayout(grid_data)
	model_dialog.grid.addWidget(groupbox2, 0, 0)
	model_dialog.grid.addWidget(groupbox_data, 1, 0)

	gui.model_dialog = model_dialog
	gui.model_dialog.run.clicked.connect(fxn)

def dialog_kmeans(gui,fxn):
	model_dialog = model_dialog_base(gui)

	groupbox2 = QGroupBox("K-means model parameters")
	grid2 = QGridLayout()
	model_dialog.add_spin_box("Number of states:", grid2, [0,0], pref='modeler.kmeans.nstates')
	groupbox2.setLayout(grid2)
	model_dialog.grid.addWidget(groupbox2, 0, 0)

	groupbox_data = QGroupBox("Data parameters")
	grid_data = QGridLayout()
	model_dialog.add_combo_box("Data Source:", grid_data, [0,0], ['FRET','Sum','0','1','2','3','Rel 0','Rel 1','Rel 2','Rel 3'],pref='modeler.dtype')
	model_dialog.add_combo_box("Clip Data?", grid_data, [1,0],['True','False'], pref='modeler.clip')
	groupbox_data.setLayout(grid_data)
	model_dialog.grid.addWidget(groupbox_data, 1, 0)

	gui.model_dialog = model_dialog
	gui.model_dialog.run.clicked.connect(fxn)

def dialog_vbhmm(gui,fxn,title,model_selection=False,threshold=False):
	model_dialog = model_dialog_base(gui)

	groupbox1 = QGroupBox("{} algorithm parameters".format(title))
	grid1 = QGridLayout()
	# model_dialog.add_spin_box("CPUs:", grid1, [0,0], pref='ncpu')
	model_dialog.add_spin_box("Restarts:", grid1, [1,0], pref='modeler.nrestarts')
	model_dialog.add_spin_box("Max iterations:", grid1, [2,0], pref='modeler.maxiters')
	model_dialog.add_line_edit("Convergence:", grid1, [3,0], pref='modeler.converge', validate = "double")
	groupbox1.setLayout(grid1)
	model_dialog.grid.addWidget(groupbox1, 1, 0)

	groupbox2 = QGroupBox("{} model parameters".format(title))
	grid2 = QGridLayout()
	if model_selection:
		model_dialog.add_spin_box("Number of states: (low)", grid2, [0,0],pref='modeler.vbhmm.nstates_min')
		model_dialog.add_spin_box("(high)", grid2, [0,2],pref='modeler.vbhmm.nstates_max')
	else:
		model_dialog.add_spin_box("Number of states:", grid2, [0,0],pref='modeler.vbhmm.nstates')
	model_dialog.add_line_edit("Prior beta:", grid2, [1,0], pref='modeler.vbhmm.prior.beta', validate = 'double')
	model_dialog.add_line_edit("Prior a:", grid2, [2,0], pref='modeler.vbhmm.prior.a', validate = 'double')
	model_dialog.add_line_edit("Prior b:", grid2, [3,0], pref='modeler.vbhmm.prior.b', validate = 'double')
	model_dialog.add_line_edit("Prior pi:", grid2, [4,0], pref='modeler.vbhmm.prior.pi', validate = 'double')
	model_dialog.add_line_edit("Prior alpha:", grid2, [5,0], pref='modeler.vbhmm.prior.alpha', validate = 'double')
	groupbox2.setLayout(grid2)
	model_dialog.grid.addWidget(groupbox2, 0, 0)

	if threshold:
		groupbox3 = QGroupBox("Threshold model parameters")
		grid3 = QGridLayout()
		model_dialog.add_line_edit("Threshold:", grid3, [1,0],validate = 'double', range= [0,1, 4],pref='modeler.threshold')
		groupbox3.setLayout(grid3)
		model_dialog.grid.addWidget(groupbox3, 2, 0)
	
	groupbox_data = QGroupBox("Data parameters")
	grid_data = QGridLayout()
	model_dialog.add_combo_box("Data Source:", grid_data, [0,0], ['FRET','Sum','0','1','2','3','Rel 0','Rel 1','Rel 2','Rel 3'],pref='modeler.dtype')
	model_dialog.add_combo_box("Clip Data?", grid_data, [1,0],['True','False'], pref='modeler.clip')
	groupbox_data.setLayout(grid_data)
	model_dialog.grid.addWidget(groupbox_data, 3, 0)

	gui.model_dialog = model_dialog
	gui.model_dialog.run.clicked.connect(fxn)

def dialog_vbgmm(gui,fxn,title,model_selection=False):
	model_dialog = model_dialog_base(gui)

	groupbox1 = QGroupBox("{} algorithm parameters".format(title))
	grid1 = QGridLayout()
	# model_dialog.add_spin_box("CPUs:", grid1, [0,0], pref='ncpu')
	model_dialog.add_spin_box("Restarts:", grid1, [1,0], pref='modeler.nrestarts')
	model_dialog.add_spin_box("Max iterations:", grid1, [2,0], pref='modeler.maxiters')
	model_dialog.add_line_edit("Convergence:", grid1, [3,0], pref='modeler.converge', validate = "double")
	groupbox1.setLayout(grid1)
	model_dialog.grid.addWidget(groupbox1, 1, 0)

	groupbox2 = QGroupBox("{} model parameters".format(title))
	grid2 = QGridLayout()
	if model_selection:
		model_dialog.add_spin_box("Number of states: (low)", grid2, [0,0],pref='modeler.vbgmm.nstates_min')
		model_dialog.add_spin_box("(high)", grid2, [0,2],pref='modeler.vbgmm.nstates_max')
	else:
		model_dialog.add_spin_box("Number of states:", grid2, [0,0],pref='modeler.vbgmm.nstates')
	model_dialog.add_line_edit("Prior beta:", grid2, [1,0], pref='modeler.vbgmm.prior.beta', validate = 'double')
	model_dialog.add_line_edit("Prior a:", grid2, [2,0], pref='modeler.vbgmm.prior.a', validate = 'double')
	model_dialog.add_line_edit("Prior b:", grid2, [3,0], pref='modeler.vbgmm.prior.b', validate = 'double')
	model_dialog.add_line_edit("Prior pi:", grid2, [4,0], pref='modeler.vbgmm.prior.pi', validate = 'double')
	groupbox2.setLayout(grid2)
	model_dialog.grid.addWidget(groupbox2, 0, 0)

	groupbox_data = QGroupBox("Data parameters")
	grid_data = QGridLayout()
	model_dialog.add_combo_box("Data Source:", grid_data, [0,0], ['FRET','Sum','0','1','2','3','Rel 0','Rel 1','Rel 2','Rel 3'],pref='modeler.dtype')
	model_dialog.add_combo_box("Clip Data?", grid_data, [1,0],['True','False'], pref='modeler.clip')
	groupbox_data.setLayout(grid_data)
	model_dialog.grid.addWidget(groupbox_data, 2, 0)

	gui.model_dialog = model_dialog
	gui.model_dialog.run.clicked.connect(fxn)

def dialog_vbgmm_vbhmm(gui,fxn,title,model_selection=False):
	model_dialog = model_dialog_base(gui)

	groupbox1 = QGroupBox("{} algorithm parameters".format(title))
	grid1 = QGridLayout()
	# model_dialog.add_spin_box("CPUs:", grid1, [0,0], pref='ncpu')
	model_dialog.add_spin_box("Restarts:", grid1, [1,0], pref='modeler.nrestarts')
	model_dialog.add_spin_box("Max iterations:", grid1, [2,0], pref='modeler.maxiters')
	model_dialog.add_line_edit("Convergence:", grid1, [3,0], pref='modeler.converge', validate = "double")
	groupbox1.setLayout(grid1)
	model_dialog.grid.addWidget(groupbox1, 1, 0)

	groupbox2 = QGroupBox("VB GMM model parameters".format(title))
	grid2 = QGridLayout()
	if model_selection:
		model_dialog.add_spin_box("Number of states: (low)", grid2, [0,0],pref='modeler.vbgmm.nstates_min')
		model_dialog.add_spin_box("(high)", grid2, [0,2],pref='modeler.vbgmm.nstates_max')
	else:
		model_dialog.add_spin_box("Number of states:", grid2, [0,0],pref='modeler.vbgmm.nstates')
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

	groupbox_data = QGroupBox("Data parameters")
	grid_data = QGridLayout()
	model_dialog.add_combo_box("Data Source:", grid_data, [0,0], ['FRET','Sum','0','1','2','3','Rel 0','Rel 1','Rel 2','Rel 3'],pref='modeler.dtype')
	model_dialog.add_combo_box("Clip Data?", grid_data, [1,0],['True','False'], pref='modeler.clip')
	groupbox_data.setLayout(grid_data)
	model_dialog.grid.addWidget(groupbox_data, 2, 0)

	gui.model_dialog = model_dialog
	gui.model_dialog.run.clicked.connect(fxn)

def dialog_vbconhmm(gui,fxn,model_selection=False,threshold=False):
	model_dialog = model_dialog_base(gui)

	groupbox1 = QGroupBox("Consensus vbHMM algorithm parameters")
	grid1 = QGridLayout()
	# model_dialog.add_spin_box("CPUs:", grid1, [0,0], pref='ncpu')
	model_dialog.add_spin_box("Restarts:", grid1, [1,0], pref='modeler.nrestarts')
	model_dialog.add_spin_box("Max iterations:", grid1, [2,0], pref='modeler.maxiters')
	model_dialog.add_line_edit("Convergence:", grid1, [3,0], pref='modeler.converge', validate = "double")
	groupbox1.setLayout(grid1)
	model_dialog.grid.addWidget(groupbox1, 1, 0)

	groupbox2 = QGroupBox("Consensus vbHMM model parameters")
	grid2 = QGridLayout()
	if model_selection:
		model_dialog.add_spin_box("Number of states: (low)", grid2, [0,0],pref='modeler.vbconhmm.nstates_min')
		model_dialog.add_spin_box("(high)", grid2, [0,2],pref='modeler.vbconhmm.nstates_max')
	else:
		model_dialog.add_spin_box("Number of states:", grid2, [0,0],pref='modeler.vbconhmm.nstates')
	model_dialog.add_line_edit("Prior beta:", grid2, [1,0], pref='modeler.vbconhmm.prior.beta', validate = 'double')
	model_dialog.add_line_edit("Prior a:", grid2, [2,0], pref='modeler.vbconhmm.prior.a', validate = 'double')
	model_dialog.add_line_edit("Prior b:", grid2, [3,0], pref='modeler.vbconhmm.prior.b', validate = 'double')
	model_dialog.add_line_edit("Prior pi:", grid2, [4,0], pref='modeler.vbconhmm.prior.pi', validate = 'double')
	model_dialog.add_line_edit("Prior alpha:", grid2, [5,0], pref='modeler.vbconhmm.prior.alpha', validate = 'double')
	groupbox2.setLayout(grid2)
	model_dialog.grid.addWidget(groupbox2, 0, 0)

	if threshold:
		groupbox3 = QGroupBox("Threshold model parameters")
		grid3 = QGridLayout()
		model_dialog.add_line_edit("Threshold:", grid3, [1,0],validate = 'double', range= [0,1, 4],pref='modeler.threshold')
		groupbox3.setLayout(grid3)
		model_dialog.grid.addWidget(groupbox3, 2, 0)

	groupbox_data = QGroupBox("Data parameters")
	grid_data = QGridLayout()
	model_dialog.add_combo_box("Data Source:", grid_data, [0,0], ['FRET','Sum','0','1','2','3','Rel 0','Rel 1','Rel 2','Rel 3'],pref='modeler.dtype')
	model_dialog.add_combo_box("Clip Data?", grid_data, [1,0],['True','False'], pref='modeler.clip')
	groupbox_data.setLayout(grid_data)
	model_dialog.grid.addWidget(groupbox_data, 3, 0)

	gui.model_dialog = model_dialog
	gui.model_dialog.run.clicked.connect(fxn)

def dialog_mlmm(gui,fxn,title,hmm_not_gmm=True):
	model_dialog = model_dialog_base(gui)

	groupbox1 = QGroupBox("{} algorithm parameters".format(title))
	grid1 = QGridLayout()
	# model_dialog.add_spin_box("CPUs:", grid1, [0,0], pref='ncpu')
	model_dialog.add_spin_box("Restarts:", grid1, [1,0], pref='modeler.nrestarts')
	model_dialog.add_spin_box("Max iterations:", grid1, [2,0], pref='modeler.maxiters')
	model_dialog.add_line_edit("Convergence:", grid1, [3,0], pref='modeler.converge', validate = "double")
	groupbox1.setLayout(grid1)
	model_dialog.grid.addWidget(groupbox1, 1, 0)

	groupbox2 = QGroupBox("{} model parameters".format(title))
	grid2 = QGridLayout()
	if hmm_not_gmm:
		model_dialog.add_spin_box("Number of states:", grid2, [0,0],pref='modeler.mlhmm.nstates')
	else:
		model_dialog.add_spin_box("Number of states:", grid2, [0,0],pref='modeler.mlgmm.nstates')
	groupbox2.setLayout(grid2)
	model_dialog.grid.addWidget(groupbox2, 0, 0)

	groupbox_data = QGroupBox("Data parameters")
	grid_data = QGridLayout()
	model_dialog.add_combo_box("Data Source:", grid_data, [0,0], ['FRET','Sum','0','1','2','3','Rel 0','Rel 1','Rel 2','Rel 3'],pref='modeler.dtype')
	model_dialog.add_combo_box("Clip Data?", grid_data, [1,0],['True','False'], pref='modeler.clip')
	groupbox_data.setLayout(grid_data)
	model_dialog.grid.addWidget(groupbox_data, 2, 0)

	gui.model_dialog = model_dialog
	gui.model_dialog.run.clicked.connect(fxn)

def dialog_ebhmm(gui,fxn,model_selection=False):
	model_dialog = model_dialog_base(gui)

	groupbox1 = QGroupBox("ebHMM algorithm parameters")
	grid1 = QGridLayout()
	# model_dialog.add_spin_box("CPUs:", grid1, [0,0], pref='ncpu')
	model_dialog.add_spin_box("Restarts:", grid1, [1,0], pref='modeler.nrestarts')
	model_dialog.add_spin_box("Max iterations:", grid1, [2,0], pref='modeler.maxiters')
	model_dialog.add_line_edit("Convergence:", grid1, [3,0], pref='modeler.converge', validate = "double")
	groupbox1.setLayout(grid1)
	model_dialog.grid.addWidget(groupbox1, 1, 0)

	groupbox2 = QGroupBox("ebHMM model parameters")
	grid2 = QGridLayout()
	if model_selection:
		model_dialog.add_spin_box("Number of states: (low)", grid2, [0,0],pref='modeler.ebhmm.nstates_min')
		model_dialog.add_spin_box("(high)", grid2, [0,2],pref='modeler.ebhmm.nstates_max')
	else:
		model_dialog.add_spin_box("Number of states:", grid2, [0,0],pref='modeler.ebhmm.nstates')
	model_dialog.add_line_edit("Prior beta:", grid2, [1,0], pref='modeler.ebhmm.prior.beta', validate = 'double')
	model_dialog.add_line_edit("Prior a:", grid2, [2,0], pref='modeler.ebhmm.prior.a', validate = 'double')
	model_dialog.add_line_edit("Prior b:", grid2, [3,0], pref='modeler.ebhmm.prior.b', validate = 'double')
	model_dialog.add_line_edit("Prior pi:", grid2, [4,0], pref='modeler.ebhmm.prior.pi', validate = 'double')
	model_dialog.add_line_edit("Prior alpha:", grid2, [5,0], pref='modeler.ebhmm.prior.alpha', validate = 'double')
	groupbox2.setLayout(grid2)
	model_dialog.grid.addWidget(groupbox2, 0, 0)

	groupbox_data = QGroupBox("Data parameters")
	grid_data = QGridLayout()
	model_dialog.add_combo_box("Data Source:", grid_data, [0,0], ['FRET','Sum','0','1','2','3','Rel 0','Rel 1','Rel 2','Rel 3'],pref='modeler.dtype')
	model_dialog.add_combo_box("Clip Data?", grid_data, [1,0],['True','False'], pref='modeler.clip')
	groupbox_data.setLayout(grid_data)
	model_dialog.grid.addWidget(groupbox_data, 2, 0)

	gui.model_dialog = model_dialog
	gui.model_dialog.run.clicked.connect(fxn)

def dialog_biasdsetup(gui,fxn):
	model_dialog = model_dialog_base(gui)

	groupbox1 = QGroupBox("MCMC parameters")
	grid1 = QGridLayout()
	model_dialog.add_line_edit("Tau:", grid1, [0,0], pref='modeler.biasd.tau', validate = 'double')
	model_dialog.add_combo_box("Likelihood type:", grid1, [1,0], ['Python','C','CUDA'], pref='modeler.biasd.likelihood')
	model_dialog.add_spin_box("Num. Walkers:", grid1, [2,0], pref='modeler.biasd.nwalkers')
	model_dialog.add_line_edit("Data Thinning:", grid1, [3,0], pref='modeler.biasd.thin',validate='int')
	model_dialog.add_line_edit("MCMC Steps:", grid1, [4,0], pref='modeler.biasd.steps',validate='int')
	model_dialog.add_file("File Name",grid1,[7,0],'modeler.biasd.filename')
	groupbox1.setLayout(grid1)
	model_dialog.grid.addWidget(groupbox1, 1, 0)

	groupbox2 = QGroupBox("BIASD Priors")
	grid2 = QGridLayout()
	model_dialog.add_biasd_prior("Epsilon_1",grid2,[1,0],'modeler.biasd.prior.e1')
	model_dialog.add_biasd_prior("Epsilon_2",grid2,[2,0],'modeler.biasd.prior.e2')
	model_dialog.add_biasd_prior("Sigma_1",grid2,[3,0],'modeler.biasd.prior.sigma1')
	model_dialog.add_biasd_prior("Sigma_2",grid2,[4,0],'modeler.biasd.prior.sigma2')
	model_dialog.add_biasd_prior("k_1",grid2,[5,0],'modeler.biasd.prior.k1')
	model_dialog.add_biasd_prior("k_2",grid2,[6,0],'modeler.biasd.prior.k2')
	groupbox2.setLayout(grid2)
	model_dialog.grid.addWidget(groupbox2, 0, 0)

	groupbox_data = QGroupBox("Data parameters")
	grid_data = QGridLayout()
	model_dialog.add_combo_box("Data Source:", grid_data, [0,0], ['FRET','Sum','0','1','2','3','Rel 0','Rel 1','Rel 2','Rel 3'],pref='modeler.dtype')
	model_dialog.add_combo_box("Clip Data?", grid_data, [1,0],['True','False'], pref='modeler.clip')
	groupbox_data.setLayout(grid_data)
	model_dialog.grid.addWidget(groupbox_data, 2, 0)

	def confirm_and_execute():
		model_dialog.update_prefs()
		success,fname = gui.maven.modeler.run_biasd_checkfname()
		if success:
			from PyQt5.QtWidgets import QMessageBox

			reply = QMessageBox.question(
				gui.model_dialog,
				'Confirm Overwrite',
				f'This will overwrite the existing file: {fname}.\nAre you sure you want to continue???',
				QMessageBox.Yes | QMessageBox.No,
				QMessageBox.No
			)
			if reply == QMessageBox.Yes:
				fxn()
		else:
			fxn()

	model_dialog.run.setText('Create')
	gui.model_dialog = model_dialog
	gui.model_dialog.run.clicked.connect(confirm_and_execute)
