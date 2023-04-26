import logging
logger = logging.getLogger(__name__)
import numpy as np

def build_menu(gui):
	from . import launchers
	from ..ui_dwell import launch_fret_dwell_analysis

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
	menu_modeler.addAction('Remove Models',gui.remove_models)
	menu_modeler.addAction('Clear Active',gui.clear_model)
	menu_modeler.addAction('Change Active',gui.change_model)
	menu_modeler.addAction('Export Active',gui.export_model)
	menu_modeler.addAction('Load Active',gui.load_model)
	menu_modeler.addAction('Update Idealized',gui.update_idealization)


	# ind = menu_modeler.addMenu('Individual (FRET)')
# 	indall = ind.addMenu('Run all')
# 	indall.addAction('mlHMM',lambda : launchers.launch_fret_mlhmm(gui))
# 	indall.addAction('vbHMM',lambda : launchers.launch_fret_vbhmm(gui))
# 	indall.addAction('vbHMM + Model selection',lambda : launchers.launch_fret_vbhmm_modelselection(gui))
# 	indone = ind.addMenu('Run one, Apply all (FRET)')
# 	indone.addAction('mlHMM',lambda : launchers.launch_fret_mlhmm_one(gui))
# 	indone.addAction('vbHMM', lambda: launchers.launch_fret_vbhmm_one(gui))

	menu_modeler.addSeparator()
	#menu_modeler.addAction('Calculate Dwell Times', lambda: analyze_dwells(gui))
	menu_modeler.addAction('Analyze Dwell Times', lambda: launch_fret_dwell_analysis(gui))

	menu_modeler.addSeparator()


	ens = menu_modeler.addMenu("FRET Modeling")
	ens.addAction('Threshold',lambda : launchers.launch_fret_threshold(gui))
	ens.addAction('K-means',lambda : launchers.launch_fret_kmeans(gui))
	ens.addAction('vbGMM',lambda : launchers.launch_fret_vbgmm(gui))
	ens.addAction('vbGMM + Model selection',lambda : launchers.launch_fret_vbgmm_modelselection(gui))
	ens.addAction('mlGMM',lambda : launchers.launch_fret_mlgmm(gui))
	ens.addAction('vbHMM -> K-means', lambda: launchers.launch_fret_kmeans_vbhmm(gui))
	ens.addAction('mlHMM -> K-means', lambda: launchers.launch_fret_kmeans_mlhmm(gui))
	ens.addAction('vbHMM -> vbGMM', lambda: launchers.launch_fret_vbgmm_vbhmm(gui))
	ens.addAction('vbHMM -> vbGMM + Model selection', lambda: launchers.launch_fret_vbgmm_vbhmm_modelselection(gui))
	ens.addAction('vbHMM -> Threshold', lambda: launchers.launch_fret_threshold_vbhmm(gui))
	ens.addAction('vbConsensus', lambda: launchers.launch_fret_vbconhmm(gui))
	ens.addAction('vbConsensus + Model selection', lambda: launchers.launch_fret_vbconhmm_modelselection(gui))
	ens.addAction('vbConsensus -> Threshold', lambda: launchers.launch_fret_threshold_vbconhmm(gui))
	ens.addAction('ebHMM', lambda: launchers.launch_fret_ebhmm(gui))
	ens.addAction('ebHMM + Model selection', lambda: launchers.launch_fret_ebhmm_modelselection(gui))


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
	return gui.maven.modeler.get_fret_traces()

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

def export(gui):
	if len(gui.maven.modeler.models) == 0:
		return
	from PyQt5.QtWidgets import QFileDialog
	oname = QFileDialog.getSaveFileName(gui, 'Export Data', 'result.hdf5','*.hdf5')[0]
	if oname == "":
		return
	success = gui.maven.modeler.export_result_to_hdf5(oname)
	if not success:
		logger.error("Failed to export model to {}".format(oname))

def load(gui):
	from PyQt5.QtWidgets import QFileDialog
	oname = QFileDialog.getOpenFileName(gui, 'Load Data', '','*.hdf5')[0]
	if oname == "":
		return
	success = gui.maven.modeler.load_result_from_hdf5(oname)
	if not success:
		logger.error("Failed to export model to {}".format(oname))

def close_dialog(gui):
	gui.model_dialog.close()
