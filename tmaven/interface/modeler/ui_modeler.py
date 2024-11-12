import logging
logger = logging.getLogger(__name__)
import numpy as np
import os

def build_menu(gui):
	from . import launchers
	from ..ui_dwell import launch_dwell_analysis

	from PyQt5.QtWidgets import QMenu
	logger.info('Adding menu modeler')

	menu_modeler = QMenu('Modeling',gui)
	# from ..stylesheet import ss_qmenu
	# menu_modeler.setStyleSheet(ss_qmenu)

	gui.clear_model = lambda : clear_model(gui)
	gui.change_model = lambda : change_model(gui)
	gui.remove_models = lambda : remove_models(gui)
	gui.update_idealization = lambda : update_idealization(gui)
	gui.export_model = lambda : export(gui)
	gui.load_model = lambda : load(gui)
	gui.generate_report = lambda : generate_report(gui)
	menu_modeler.addAction('Remove Models',gui.remove_models)
	menu_modeler.addAction('Clear Active',gui.clear_model)
	menu_modeler.addAction('Change Active',gui.change_model)
	menu_modeler.addAction('Export Active',gui.export_model)
	menu_modeler.addAction('Load Active',gui.load_model)
	menu_modeler.addAction('Update Idealized',gui.update_idealization)
	menu_modeler.addAction('Generate Report',gui.generate_report)


	# ind = menu_modeler.addMenu('Individual (FRET)')
# 	indall = ind.addMenu('Run all')
# 	indall.addAction('mlHMM',lambda : launchers.launch_mlhmm(gui))
# 	indall.addAction('vbHMM',lambda : launchers.launch_vbhmm(gui))
# 	indall.addAction('vbHMM + Model selection',lambda : launchers.launch_vbhmm_modelselection(gui))
# 	indone = ind.addMenu('Run one, Apply all (FRET)')
# 	indone.addAction('mlHMM',lambda : launchers.launch_mlhmm_one(gui))
# 	indone.addAction('vbHMM', lambda: launchers.launch_vbhmm_one(gui))

	menu_modeler.addSeparator()
	

	# menu_modeler.addSeparator()

	ens = menu_modeler.addMenu("Models")
	
	ens.addAction('Threshold',lambda : launchers.launch_threshold(gui))
	mixture = ens.addMenu('Mixtures')
	composites = ens.addMenu('Composite HMMs')
	globalhmm = ens.addMenu('Global HMMs')
	mixture.addAction('K-means',lambda : launchers.launch_kmeans(gui))
	mixture.addAction('vbGMM',lambda : launchers.launch_vbgmm(gui))
	mixture.addAction('vbGMM + Model selection',lambda : launchers.launch_vbgmm_modelselection(gui))
	mixture.addAction('mlGMM',lambda : launchers.launch_mlgmm(gui))
	composites.addAction('vbHMM -> K-means', lambda: launchers.launch_kmeans_vbhmm(gui))
	composites.addAction('mlHMM -> K-means', lambda: launchers.launch_kmeans_mlhmm(gui))
	composites.addAction('vbHMM -> vbGMM', lambda: launchers.launch_vbgmm_vbhmm(gui))
	composites.addAction('vbHMM -> vbGMM + Model selection', lambda: launchers.launch_vbgmm_vbhmm_modelselection(gui))
	composites.addAction('vbHMM -> Threshold', lambda: launchers.launch_threshold_vbhmm(gui))
	globalhmm.addAction('vbConsensus', lambda: launchers.launch_vbconhmm(gui))
	globalhmm.addAction('vbConsensus + Model selection', lambda: launchers.launch_vbconhmm_modelselection(gui))
	globalhmm.addAction('vbConsensus -> Threshold', lambda: launchers.launch_threshold_vbconhmm(gui))
	globalhmm.addAction('ebHMM', lambda: launchers.launch_ebhmm(gui))
	globalhmm.addAction('ebHMM + Model selection', lambda: launchers.launch_ebhmm_modelselection(gui))

	#menu_modeler.addAction('Calculate Dwell Times', lambda: analyze_dwells(gui))
	menu_modeler.addAction('Analyze Dwell Times', lambda: launch_dwell_analysis(gui))

	exptl = menu_modeler.addMenu("Experimental")
	biasd = exptl.addMenu('BIASD')
	biasd.addAction("Setup", lambda : launchers.launch_biasd_setup(gui))
	biasd.addAction("Run MCMC", lambda: gui.maven.modeler.run_biasd_mcmc(stochastic=False))
	biasd.addAction("Run MCMC (Stochastic)", lambda: gui.maven.modeler.run_biasd_mcmc(stochastic=True))
	biasd.addAction("Randomize All Walkers", lambda: gui.maven.modeler.run_biasd_randomizep0(justdead=False))
	biasd.addAction("Randomize Dead Walkers", lambda: gui.maven.modeler.run_biasd_randomizep0(justdead=True))
	biasd.addAction("Analyze Chain", lambda: gui.maven.modeler.run_biasd_analyze())

	# hhmm = exptl.addMenu('hHMM')
	# hhmm.addAction('Coming Soon', lambda: print('Coming Soon'))

	return menu_modeler


def select_model_popup(gui,title='',flag_select_multiple=False):
	'''
	Sort of important that this stays blocking b/c I'm too lazy to get updates from modeler if something changes
	'''
	if len(gui.maven.modeler.models) == 0:
		return False, []

	from PyQt5.QtWidgets import (QDialog,QListWidget,QVBoxLayout,QHBoxLayout,
								QPushButton,QWidget,QAbstractItemView, QStyleFactory)

	class model_dialog(QDialog):
		def __init__(self,parent,items,title,flag_select_multiple):
			super().__init__(parent)
			self.setWindowTitle(title)
			self.flag_select_multiple = flag_select_multiple

			#### setup widgets
			self.lw = QListWidget(parent)
			if self.flag_select_multiple:
				self.lw.setSelectionMode(QAbstractItemView.MultiSelection)
			else:
				self.lw.setSelectionMode(QAbstractItemView.SingleSelection)
			self.lw.addItems(items)
			# self.lw.setCurrentRow(0)
			from PyQt5.Qt import QFont,QFontDatabase
			monospace = QFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
			self.lw.setFont(monospace)
			button_okay = QPushButton('Select')
			button_cancel = QPushButton('Cancel')
			qw = QWidget()
			vbox = QVBoxLayout()
			hbox = QHBoxLayout()

			#### setup layout
			hbox.addWidget(button_cancel)
			hbox.addWidget(button_okay)
			qw.setLayout(hbox)
			vbox.addWidget(self.lw)
			vbox.addWidget(qw)
			self.setLayout(vbox)

			#### sizing
			sp = self.sizePolicy()
			self.setSizePolicy(sp.MinimumExpanding,sp.MinimumExpanding)
			maxcolumn = self.lw.sizeHintForColumn(0)
			maxrow = np.max([self.lw.sizeHintForRow(i) for i in range(self.lw.count())])
			self.lw.setFixedSize(maxcolumn + 2*self.lw.frameWidth(), maxrow*self.lw.count() + 2*self.lw.frameWidth())

			#### button connections
			button_cancel.clicked.connect(self.reject)
			button_okay.clicked.connect(self.accept)
			button_okay.setDefault(True)

			#### set style
			self.setStyle(QStyleFactory.create('Fusion'))
			from ..stylesheet import ui_stylesheet
			self.setStyleSheet(ui_stylesheet)

		def exec_(self):
			result = super().exec_()
			indexes = [self.lw.row(s) for s in self.lw.selectedItems()]
			for i in range(len(indexes)):
				if indexes[i] == -1:
					indexes[i] = None
			return result == 1, indexes

	items = gui.maven.modeler.get_model_descriptions()

	dialog = model_dialog(gui,items,title,flag_select_multiple)
	success,value = dialog.exec_()
	return success,value

def get_fret_traces(gui):
	return gui.maven.modeler.get_traces('FRET')

def update_idealization(gui):
	try:
		gui.maven.modeler.model.idealize()
		logger.info('Updated idealization for active model')
	except Exception as e:
		logger.info('Failed to update idealization for active model\n{}'.format(e))
	gui.maven.emit_data_update()

def change_model(gui):
	old = gui.maven.modeler._active_model_index
	success,indexes = select_model_popup(gui,'Change Active Model',False)
	if success:
		gui.maven.modeler.set_model(indexes[0])
		logger.info('Changed active model from {} to {}'.format(old,gui.maven.modeler._active_model_index))

def clear_model(gui):
	logger.info('Cleared active model')
	gui.maven.modeler.set_model(None)

def remove_models(gui):
	success,indexes = select_model_popup(gui,'Remove Models',True)
	if success:
		gui.maven.modeler.remove_models(indexes)
		logger.info('Removed models {}. Activate model is now {}'.format(indexes,gui.maven.modeler._active_model_index))

def generate_report(gui):
	if len(gui.maven.modeler.models) == 0:
		return
	gui.maven.modeler.make_report(gui.maven.modeler.model)

def export(gui):
	if len(gui.maven.modeler.models) == 0:
		return
	from PyQt5.QtWidgets import QFileDialog
	oname = QFileDialog.getSaveFileName(gui, 'Export Data', os.path.join(gui.lwd,'result.hdf5'),'*.hdf5')[0]
	if oname == "":
		return
	success = gui.maven.modeler.export_result_to_hdf5(oname)
	if success:
		lwd = os.path.dirname(oname)
		gui.lwd_update(lwd)
	else:
		logger.error("Failed to export model to {}".format(oname))

def load(gui):
	from PyQt5.QtWidgets import QFileDialog
	oname = QFileDialog.getOpenFileName(gui, 'Load Data', gui.lwd, '*.hdf5')[0]
	if oname == "":
		return
	success = gui.maven.modeler.load_result_from_hdf5(oname)
	if success:
		lwd = os.path.dirname(oname)
		gui.lwd_update(lwd)
	else:
		logger.error("Failed to export model to {}".format(oname))

def close_dialog(gui):
	gui.model_dialog.update_prefs()
	gui.model_dialog.close()