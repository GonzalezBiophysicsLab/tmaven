import logging
logger = logging.getLogger(__name__)
import numpy as np
import h5py
import os



def build_menu(gui):
	from PyQt5.QtWidgets import QMenu, QAction

	menu_load = QMenu('Load',gui)
	menu_save = QMenu('Export',gui)

	### Load
	menu_load.addAction('SMD',lambda : load_interactive(gui),shortcut='Ctrl+O')

	menu_hdf5 = menu_load.addMenu('HDF5 Dataset')
	menu_hdf5.addAction('Raw',lambda : load_raw_hdf5dataset(gui), shortcut='Ctrl+Shift+O')
	menu_hdf5.addAction('All tMAVEN',lambda : load_tmaven_dataset_hdf5(gui, "all"))
	menu_hdf5.addAction('Classes',lambda : load_tmaven_dataset_hdf5(gui, "class"))
	menu_hdf5.addAction('Pre-Post Times',lambda : load_tmaven_dataset_hdf5(gui, "pre-post"))

	menu_txt = menu_load.addMenu('ASCII Text files')
	menu_txt.addAction('Raw',lambda : load_raw_text(gui))
	menu_txt.addAction('All tMAVEN',lambda : load_tmaven_dataset_txt(gui, "all"))
	menu_txt.addAction('Classes',lambda : load_tmaven_dataset_txt(gui, "class"))
	menu_txt.addAction('Pre-Post Times',lambda : load_tmaven_dataset_txt(gui, "pre-post"))

	menu_npy = menu_load.addMenu('Numpy arrays')
	menu_npy.addAction('Raw',lambda : load_raw_numpy(gui))

	menu_load.addAction('SPARTAN traces',lambda : load_raw_spartan(gui))
	### save
	menu_save.addAction('Save (SMD)',lambda : export_smd(gui),shortcut='Ctrl+S')
	menu_other = menu_save.addMenu('Other')
	menu_other.addAction('Save raw (numpy)',lambda : export_numpy(gui))
	menu_other.addAction('Save classes (txt)',lambda : export_text(gui))

	# from ..interface.stylesheet import ss_qmenu
	# [m.setStyleSheet(ss_qmenu) for m in [menu_load,menu_save,menu_raw,menu_other]]
	return menu_load,menu_save

def get_filenames(gui):
	from PyQt5.QtWidgets import QFileDialog
	fnames = QFileDialog.getOpenFileNames(gui,'Choose one or more files to load data','./')[0]
	return fnames


def get_items_dialog(gui,items,title,subtitle,flag_select_multiple):
	from PyQt5.QtWidgets import QDialog, QListWidget,QVBoxLayout,QHBoxLayout,QPushButton,QWidget,QAbstractItemView, QStyleFactory, QSizePolicy, QLabel
	class _get_items_dialog(QDialog):
		def __init__(self,parent,items,title,subtitle,flag_select_multiple):
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
			vbox.addWidget(QLabel(subtitle))
			vbox.addWidget(self.lw)
			vbox.addStretch(1)
			vbox.addWidget(qw)
			self.setLayout(vbox)

			#### sizing
			sp = self.sizePolicy()
			self.setSizePolicy(sp.MinimumExpanding,sp.MinimumExpanding)
			maxcolumn = self.lw.sizeHintForColumn(0)
			maxrow = np.max([self.lw.sizeHintForRow(i) for i in range(self.lw.count())])
			# self.lw.setFixedSize(maxcolumn + 2*self.lw.frameWidth(), maxrow*self.lw.count() + 2*self.lw.frameWidth())
			self.lw.setSizePolicy(sp.Expanding,sp.Fixed)
			self.lw.setFixedHeight(maxrow*self.lw.count() + 2*self.lw.frameWidth())

			#### button connections
			button_cancel.clicked.connect(self.reject)
			button_okay.clicked.connect(self.accept)
			button_okay.setDefault(True)

			# self.lw.selectAll()

			#### set style
			self.setStyle(QStyleFactory.create('Fusion'))
			from .stylesheet import ui_stylesheet
			self.setStyleSheet(ui_stylesheet)

		def exec_(self):
			result = super().exec_()
			indexes = [self.lw.row(s) for s in self.lw.selectedItems()]
			for i in range(len(indexes)):
				if indexes[i] == -1:
					indexes[i] = None
			return result == 1, indexes

	dialog = _get_items_dialog(gui,items,title,subtitle,flag_select_multiple)
	success,value = dialog.exec_()
	return success,value

def get_datasetname_hdf5(gui,fname, dataset_name):
	from ..pysmd import find_datasets_in_hdf5
	datasets = find_datasets_in_hdf5(fname)
	ds = ['%s %s'%(datasets[i][0],str(datasets[i][1])) for i in range(len(datasets))]
	success,indexes = get_items_dialog(gui,ds,"Select one Dataset for {}".format(dataset_name),fname,False)
	if success:
		return success,datasets[indexes[0]][0]
	else:
		return success,''

def load_interactive(gui):
	from PyQt5.QtWidgets import QInputDialog

	fnames = get_filenames(gui)
	if len(fnames) == 0:
		logger.info('No files to load')
		return

	holding_pen = []
	for i in range(len(fnames)):
		logger.info('Trying to load %s'%(fnames[i]))

		smd_group_keys = gui.maven.io.find_smds_in_hdf5(fnames[i])
		if len(smd_group_keys) == 0:
			logger.info('No SMDs detected in %s'%(fnames[i]))
			continue
		elif len(smd_group_keys) == 1:
			to_load = smd_group_keys
		else:
			success,indexes = get_items_dialog(gui,smd_group_keys,"Select SMD Group(s)",fnames[i],True)
			to_load = [smd_group_keys[ii] for ii in indexes]

		for group in to_load:
			try:
				logger.info('Trying to load smd %s/%s'%(fnames[i],group))
				smd = gui.maven.io.load_smd_hdf5(fnames[i],group)
				if gui.maven.prefs['io.force_double']:
					smd.raw = smd.raw.astype('double')
			except Exception as e:
				logger.error('Failed to load smd %s/%s\n%s'%(fnames[i],group,str(e)))
				continue
			try:
				logger.info('Trying to load tMAVEN %s/%s'%(fnames[i],group))
				tmaven = gui.maven.io.load_tmaven_hdf5(fnames[i],group)
			except Exception as e:
				logger.error('Failed to load tMAVEN %s/%s\n%s'%(fnames[i],group,str(e)))
				tmaven = None
			holding_pen.append([smd,tmaven])

	for hpi in holding_pen:
		gui.maven.io.add_data(*hpi)
	del holding_pen
	gui.maven.emit_data_update()


def load_raw_hdf5dataset(gui):
	from PyQt5.QtWidgets import QFileDialog
	fname = QFileDialog.getOpenFileName(gui,'Choose one HDF5 file to load dataset from','./')[0]
	if fname == "":
		logger.info('No file to load')
		return

	try:
		logger.info('Trying to load %s'%(fname))
		success,dataset_name = get_datasetname_hdf5(gui,fname, "Raw traces")

		gui.maven.io.load_raw_hdf5(fname,dataset_name)

	except Exception as e:
		logging.error('failed to load %s\n%s'%(fname,str(e)))
		return

def load_raw_text(gui):
	from PyQt5.QtWidgets import QFileDialog
	fname = QFileDialog.getOpenFileName(gui,'Choose .dat ASCII file to load dataset from','./')[0]
	if fname == "":
		logger.info('No file to load')
		return

	try:
		logger.info('Trying to load %s'%(fname))
		skiprows = gui.maven.prefs['io.skiprows']
		delimiter = str(gui.maven.prefs['io.delimiter'])
		gui.maven.io.load_raw_txt(fname,skiprows,delimiter)

	except Exception as e:
		logging.error('failed to load %s\n%s'%(fname,str(e)))
		return

def load_raw_numpy(gui):
	from PyQt5.QtWidgets import QFileDialog
	fname = QFileDialog.getOpenFileName(gui,'Choose .npy binary file to load dataset from','./')[0]
	if fname == "":
		logger.info('No file to load')
		return

	try:
		logger.info('Trying to load %s'%(fname))
		gui.maven.io.load_smd_numpy(fname)
	except Exception as e:
		logging.error('failed to load %s\n%s'%(fname,str(e)))
		return

def load_raw_spartan(gui):
	from PyQt5.QtWidgets import QFileDialog
	fname = QFileDialog.getOpenFileName(gui,'Choose .traces spartan file to load dataset from','./')[0]
	if fname == "":
		logger.info('No file to load')
		return

	try:
		logger.info('Trying to load %s'%(fname))
		gui.maven.io.load_spartan(fname)
	except Exception as e:
		logging.error('failed to load %s\n%s'%(fname,str(e)))
		return

def load_tmaven_dataset_hdf5(gui,d_name):
	from PyQt5.QtWidgets import QFileDialog
	fname = QFileDialog.getOpenFileName(gui,'Choose one HDF5 file to load dataset from','./')[0]
	if fname == "":
		logger.info('No file to load')
		return

	try:
		logger.info('Trying to load %s'%(fname))

		if d_name == "all":
			success1,class_name = get_datasetname_hdf5(gui,fname, "Classes")
			success2,pre_name = get_datasetname_hdf5(gui,fname, "Pre-time")
			success3,post_name = get_datasetname_hdf5(gui,fname, "Post-time")
		elif d_name == "class":
			success1,class_name = get_datasetname_hdf5(gui,fname, "Classes")
			success2,pre_name = (True, None)
			success3,post_name = (True, None)
		elif d_name == "pre-post":
			success1,class_name = (True, None)
			success2,pre_name = get_datasetname_hdf5(gui,fname, "Pre-time")
			success3,post_name = get_datasetname_hdf5(gui,fname, "Post-time")

		if success1 and success2 and success3:
			datasets = [class_name, pre_name, post_name]

		gui.maven.io.load_tmaven_all_hdf5(fname,datasets)

	except Exception as e:
		logging.error('failed to load %s\n%s'%(fname,str(e)))
		return

def load_tmaven_dataset_txt(gui,d_name):
	from PyQt5.QtWidgets import QFileDialog
	fname = QFileDialog.getOpenFileName(gui,'Choose .dat ASCII file to load from','./')[0]
	if fname == "":
		logger.info('No file to load')
		return

	try:
		logger.info('Trying to load %s'%(fname))
		skiprows = gui.maven.prefs['io.skiprows']
		delimiter = gui.maven.prefs['io.delimiter']
		gui.maven.io.load_tmaven_all_txt(fname,skiprows,delimiter,d_name)

	except Exception as e:
		logging.error('failed to load %s\n%s'%(fname,str(e)))
		return



def export_smd(gui):
	from PyQt5.QtWidgets import QFileDialog,QInputDialog
	oname = QFileDialog.getSaveFileName(gui, 'Export Data', '_.hdf5','*.hdf5')[0]
	if not oname == "":
		try:
			with h5py.File(oname,'r') as f:
				keys = list(f.keys())
		except:
			keys = []
		gname = QInputDialog.getText(gui,'SMD Group Name','Enter SMD group name. Already existing is {}'.format(keys), text='dataset')[0]
		if not gname == "":
			gui.maven.io.export_smd_hdf5(oname,gname)
		gui.maven.io.export_tmaven_hdf5(oname,gname)


def export_numpy(gui):
	from PyQt5.QtWidgets import QFileDialog
	oname = QFileDialog.getSaveFileName(gui, 'Export data (numpy)', '_.npy','*.npy')[0]
	if oname != "":
		gui.maven.io.export_raw_numpy(oname)
		cname = '{}_classes{}'.format(*os.path.splitext(oname))
		gui.maven.io.export_class_numpy(cname)


def export_text(gui):
	from PyQt5.QtWidgets import QFileDialog
	oname = QFileDialog.getSaveFileName(gui, 'Export classes (txt)', '_.txt','*.txt')[0]
	if oname != "":
		# gui.maven.io.export_raw_txt(oname)
		# cname = '{}_classes{}'.format(*os.path.splitext(oname))
		gui.maven.io.export_class_txt(oname)
