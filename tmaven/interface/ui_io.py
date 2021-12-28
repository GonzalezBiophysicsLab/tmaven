import logging
logger = logging.getLogger(__name__)
import h5py
import os


def build_menu(gui):
	from PyQt5.QtWidgets import QMenu, QAction

	menu_load = QMenu('Load')
	menu_save = QMenu('Export')

	### Load
	menu_load.addAction('Load Data',lambda : load_interactive(gui),shortcut='Ctrl+O')
	# menu_load.addAction('Make Fake Data',lambda : load_fake_data(gui))

	### save
	menu_save.addAction('Save SMD',lambda : export_smd(gui),shortcut='Ctrl+S')
	menu_save.addAction('Save numpy',lambda : export_numpy(gui))
	menu_save.addAction('Save classes (txt)',lambda : export_txt(gui))

	from ..interface.stylesheet import ss_qmenu
	[m.setStyleSheet(ss_qmenu) for m in [menu_load,menu_save]]
	return menu_load,menu_save

def get_filenames(gui):
	from PyQt5.QtWidgets import QFileDialog
	fnames = QFileDialog.getOpenFileNames(gui,'Choose one or more files to load data','./')[0]
	return fnames

def load_interactive(gui):
	from PyQt5.QtWidgets import QInputDialog

	fnames = get_filenames(gui)
	if len(fnames) == 0:
		logger.info('No files to load')
		return

	order = gui.maven.prefs['io.axis_order']
	missing = gui.maven.prefs['io.missing_axis']
	decollate = gui.maven.prefs['io.decollate']
	decollate_axis = gui.maven.prefs['io.decollate_axis']

	holding_pen = []
	for i in range(len(fnames)):
		logger.info('Trying to load %s'%(fnames[i]))
		if fnames[i].endswith('.hdf5') or fnames[i].endswith('.hdf') or fnames[i].endswith('.smd'):
			smd_keys = []
			group_keys = {}
			all_keys = []
			with h5py.File(fnames[i],'r') as f:
				## explore hdf5 file
				all_keys = list(f.keys())
				for k in all_keys:
					if type(f[k]) is h5py._hl.group.Group:
						if 'format' in f[k].attrs.keys():
							if f[k].attrs['format'] in ['smd','SMD','Smd']:
								smd_keys.append(k)
						group_keys[k] = list(f[k].keys())

			## It's a full-fledged SMD file
			if len(smd_keys) > 0:
				from PyQt5.QtWidgets import QInputDialog
				group,success = QInputDialog().getItem(gui, "Choose SMD Group", "Choose SMD Group Name:", smd_keys, 0, False)
				if success:
					success,smd = gui.maven.io.load_smd(fnames[i],group)
					if success:
						holding_pen.append(smd)
					continue
				else:
					return

			## It's maybe a labeled smd-like file
			elif len(list(group_keys.keys())) > 0:
				group,success = QInputDialog().getItem(gui, "Choose HDF5 Group", "Group Name:", group_keys, 0, False)
				if success:
					dataset,success = QInputDialog().getItem(gui, "Choose Dataset in {}".format(group), "Dataset Name:", group_keys[group], 0, False)
					if success:
						holding_pen.append(gui.maven.io.load_group_hdf5(fnames[i],group,dataset,order,missing,decollate,decollate_axis))
						continue
					else:
						return
				else:
					return

			## maybe they failed the group pick from hdf5 and want a toplevel dataset
			dataset,success = QInputDialog().getItem(gui, "Choose Top-level Dataset", "Dataset Name:", all_keys, 0, False)
			if success:
				holding_pen.append(gui.maven.io.load_group_hdf5(fnames[i],None,dataset,order,missing,decollate,decollate_axis))
				continue
			else:
				return

		## it's a numpy binary file
		if fnames[i].endswith('.npy'):
			holding_pen.append(gui.maven.io.load_numpy(fnames[i],order,missing,decollate,decollate_axis))
			continue
		## maybe it is a text file
		else:
			skiprows = gui.maven.prefs['io.skiprows']
			delimiter = gui.maven.prefs['io.delimiter']
			holding_pen.append(gui.maven.io.load_txt(fnames[i],skiprows,delimiter,order,missing,decollate,decollate_axis))
			continue

	for smd in holding_pen:
		gui.maven.io.add_data(smd,None)
	del holding_pen
	gui.maven.emit_data_update()


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
			gui.maven.io.export_smd(oname,gname)


def export_numpy(gui):
	from PyQt5.QtWidgets import QFileDialog
	oname = QFileDialog.getSaveFileName(gui, 'Export data (numpy)', '_.npy','*.npy')[0]
	if oname != "":
		gui.maven.io.export_raw_numpy(oname)
		cname = '{}_classes{}'.format(*os.path.splitext(oname))
		gui.maven.io.export_class_numpy(cname)


def export_txt(gui):
	from PyQt5.QtWidgets import QFileDialog
	oname = QFileDialog.getSaveFileName(gui, 'Export classes (txt)', '_.txt','*.txt')[0]
	if oname != "":
		# gui.maven.io.export_raw_txt(oname)
		# cname = '{}_classes{}'.format(*os.path.splitext(oname))
		gui.maven.io.export_class_txt(oname)
