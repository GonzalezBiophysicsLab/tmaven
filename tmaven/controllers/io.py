import numpy as np
import h5py
import time
import os
import logging
logger = logging.getLogger(__name__)
# from ..maven import check_mavengui_present

default_prefs = {
	'io.skiprows':0,
	'io.delimiter':r'\t',
	'io.axis_order':[0,1,2],
	'io.missing_axis':2,
	'io.decollate':1,
	'io.decollate_axis':1,
}

from ..pysmd import smd_container, concatenate_smds, save_smd_in_hdf5, load_smd_in_hdf5, find_smds_in_hdf5

class controller_io(object):
	''' Handles input/output of data

	Parameters
	# ----------
	# gui : plotter_gui
	# 	instance of the `fret_plot` GUI

	Attributes
	----------
	# menu_load : QMenu
	# 	menu for options under 'File >> Load'
	# 		* load data
	# 		* load HMM
	# menu_save : QMenu
	# 	menu for options under 'File >> Export'
	# 		* save SMD
	# 		* save old HDF5
	# action_clear : QAction
	# 	clear data
	'''

	def __init__(self,maven):
		self.maven = maven
		self.maven.prefs.add_dictionary(default_prefs)

	def add_data(self,new_smd,new_tmaven=None):
		if self.maven.smd.raw.shape == (0,0,0):
			self.maven.smd = new_smd
			self.maven.data.initialize_tmaven_params()
		else:
			self.maven.smd = concatenate_smd(self.maven.smd,new_smd[0])
			self.maven.data.update_tmaven_params()
		if not new_tmaven is None:
			classes, pre_list, post_list = new_tmaven
			self.maven.data.classes[-classes.size:] = classes.copy()
			self.maven.data.pre_list[-classes.size:] = pre_list.copy()
			self.maven.data.post_list[-classes.size:] = post_list.copy()
		self.process_data_change()

	def process_data_change(self):
		self.maven.data.update_tmaven_params()
		self.maven.emit_data_update()

	def clear_data(self):
		self.maven.smd = smd_container()
		self.process_data_change()

	def blank_smd(self):
		return smd_container()

	def load_smd_hdf5(self,fname,gname):
		try:
			success,smd = load_smd_in_hdf5(fname,gname)
			if success:
				return smd
		except Exception as e:
			logger.error('smdload failed {}/{}\nerror:{}'.format(fname,gname,str(e)))
		raise Exception('smdload was not successful {}/{}'.format(fname,gname))

	def load_smd_group_hdf5(self,fname,group,dataset,order,missing,decollate,decollate_axis):
		try:
			classkey = None
			classes = None
			with h5py.File(fname,'r') as f:
				if group is None:
					g = f
				else:
					g = f[group]
				d = g[dataset][:]

				for k in g.keys():
					if k.lower().startswith('class'):
						classkey = k
						classes = g[classkey][:]
						logger.info('found classes in {}'.format(classkey))
			logger.info('loaded group hdf5 {} with key {}/{}.'.format(fname, group, dataset))
			d,success = self.fix_decollate(d,decollate,decollate_axis)
			if not success:
				logger.error('group hdf5 load failed {} {}/{} because of bad decollating'.format(fname,group,dataset))
				return None
			d,success = self.fix_missing_dimensions(d,missing)
			if not success:
				logger.error('group hdf5 load failed {} {}/{} because of bad missing dimension'.format(fname,group,dataset))
				return None
			d,success = self.fix_axis_order(d,order)
			if not success:
				logger.error('group hdf5 load failed {} {}/{} because of bad ordering'.format(fname,group,dataset))
				return None

			smd = self.blank_smd()
			smd.initialize_data(d)
			if group is None:
				smd.source_names[0] = '{}:{}'.format(fname,dataset)
			else:
				smd.source_names[0] = '{}:{}/{}'.format(fname,group,dataset)
			if not classes is None:
				if d.shape[0] == classes.size:
					smd.classes = classes
			return smd
		except Exception as e:
			logger.error('group hdf5 load failed {}\n{}'.format(fname,str(e)))

	def load_smd_txt(self,fname,skiprows,delimiter,order,missing,decollate,decollate_axis):
		try:
			d = np.loadtxt(fname,skiprows=skiprows,delimiter=delimiter)
			logger.info('loaded txt {}. skiprows {}, delimiter {}'.format(fname, skiprows, delimiter))
			d,success = self.fix_decollate(d,decollate,decollate_axis)
			if not success:
				logger.error('txt load failed {} {}/{} because of bad decollating'.format(fname,group,dataset))
				return None
			d,success = self.fix_missing_dimensions(d,missing)
			if not success:
				logger.error('txt load failed {} because of bad missing dimension'.format(fname))
				return
			d,success = self.fix_axis_order(d,order)
			if not success:
				logger.error('txt load failed {} because of bad ordering'.format(fname))
				return
			smd = self.blank_smd()
			smd.initialize_data(d)
			smd.source_names[0] = '{}'.format(fname)
			return smd
		except Exception as e:
			logger.error('txt load failed {}\n{}'.format(fname,str(e)))

	def load_smd_numpy(self,fname,order,missing,decollate,decollate_axis):
		try:
			d = np.load(fname)
			logger.info('loaded np {}'.format(fname))
			d,success = self.fix_decollate(d,decollate,decollate_axis)
			if not success:
				logger.error('txt load failed {} {}/{} because of bad decollating'.format(fname,group,dataset))
				return None
			d,success = self.fix_missing_dimensions(d,missing)
			if not success:
				logger.error('np load failed {} because of bad missing dimension'.format(fname))
				return
			d,success = self.fix_axis_order(d,order)
			if not success:
				logger.error('np load failed {} because of bad ordering'.format(fname))
				return
			smd = self.blank_smd()
			smd.initialize_data(d)
			smd.source_names[0] = '{}'.format(fname)
			return smd
		except Exception as e:
			logger.error('np load failed {}\n{}'.format(fname,str(e)))

	def load_tmaven_hdf5(self,fname,gname):
		success = False
		try:
			with h5py.File(fname,'r') as f:
				g = f[gname]
				if 'tMAVEN' in g:
					gd = g['tMAVEN']
					format = gd.attrs['format']
					if format == 'tMAVEN':
						date_modified = gd.attrs['date_modified']
						classes = gd['classes'][:]
						pre_list = gd['pre_list'][:]
						post_list = gd['post_list'][:]
						success = True
			if success:
				logger.info("Loaded tMAVEN from HDF5 file: %s in group: %s"%(fname,gname))
				return classes.copy(), pre_list.copy(), post_list.copy()
		except Exception as e:
			logger.error("Error load tMAVEN from HDF5 file%s"%(str(e)))
		logger.error("Failed to load tMAVEN from HDF5 file: %s in group: %s"%(fname,gname))
		return None

	def fix_missing_dimensions(self,d,missing):
		success = False
		if d.ndim == 1:
			d = d.reshape((1,d.size,1))
			logger.info('added dimensions 0 and 2')
			success = True
		elif d.ndim == 2:
			if missing == 2:
				d = d.reshape((d.shape[0],d.shape[1],1))
				logger.info('added dimension 2')
				success = True
			elif missing == 0:
				d = d.reshape((1,d.shape[0],d.shape[1]))
				logger.info('added dimension 0')
				success = True
		else:
			success = True
		return d,success

	def fix_axis_order(self,d,order):
		if order == [0,1,2]:
			logger.info('axis already ordered properly')
			return d,True
		if not 0 in order and not 1 in order and not 2 in order and len(order) == 3:
			logger.error('axis order {} does not make sense. Should be a list and have 0 1 and 2'.fomat(order))
			return d,False
		d = np.moveaxis(d,order,[0,1,2])
		logger.info('moved axes from {} to [0,1,2]'.format(order))
		return d,True

	def fix_decollate(self,d,n,axis):
		if type(n) != int:
			return d,False
		if n == 1:
			return d,True
		elif n > 1:
			if axis == 0:
				return np.array([d[i::n] for i in range(n)]),True
			elif axis == 1:
				return np.array([d[:,i::n] for i in range(n)]),True
		return d,False

	def export_tmaven_hdf5(self,oname,gname):
		if self.maven.smd.nmol == 0:
			return
		if oname == "" or gname == "":
			return
		mask = self.maven.data.get_toggled_mask()
		try:
			with h5py.File(oname,'a') as f:
				g = f[gname]
				if 'tMAVEN' in g:
					del g['tMAVEN']
					f.flush()
				gd = g.create_group('tMAVEN')
				gd.attrs['format'] = 'tMAVEN'
				gd.attrs['date_modified'] = time.ctime()
				gd.create_dataset('classes',data=self.classes[mask].astype('int64'),dtype='int64',compression='gzip')
				gd.create_dataset('pre_list',data=self.pre_list[mask].astype('int64'),dtype='int64',compression='gzip')
				gd.create_dataset('post_list',data=self.post_list[mask].astype('int64'),dtype='int64',compression='gzip')
				f.flush()
				f.close()
				logger.info("Export tMAVEN to HDF5 file: %s in group: %s"%(oname,gname))
		except Exception as e:
			logger.error("Failed to export tMAVEN to HDF5 file: %s in group: %s\n%s"%(oname,gname,str(e)))

	def export_smd_hdf5(self,oname,gname):
		'''  Save data function

		Parameters
		----------
		oname : string
			filename
		gname : string
			hdf5 group name -- i.e. experiment name

		Notes
		-----
		Format is documented in `fret_plot/data/smd.py` and `fret_plot/data/fret.py`

		'''
		if not self.maven.smd.nmol == 0:
			if not oname == "" and not gname == "":
				mask = self.maven.data.get_toggled_mask()
				try:
					success = save_smd_in_hdf5(self.maven.smd,oname,gname,overwrite=True,mask=mask)
					if success:
						logger.info("Exported data %s %s"%(oname,gname))
						return
					logger.info("Failed to export SMD %s %s"%(oname,gname))
				except Exception as e:
					logger.error("Failed to export SMD %s %s\n%s"%(oname,gname,str(e)))

		logger.error('There was a problem trying to export the traces')

	def export_raw_numpy(self,oname):
		if not self.maven.smd.nmol == 0:
			if not oname == "":
				mask = self.maven.data.get_toggled_mask()
				success = np.save(oname,self.maven.smd.raw[mask])
				logger.info("Exported data %s"%(oname))
				return
		logger.error('There was a problem trying to export the traces')

	def export_class_numpy(self,oname):
		if not self.maven.smd.nmol == 0:
			if not oname == "":
				mask = self.maven.data.get_toggled_mask()
				success = np.save(oname,self.maven.data.classes[mask])
				logger.info("Exported data %s"%(oname))
				return
		logger.error('There was a problem trying to export the traces')

	def export_class_txt(self,oname):
		if not self.maven.smd.nmol == 0:
			if not oname == "":
				mask = self.maven.data.get_toggled_mask()
				success = np.savetxt(oname,self.maven.data.classes[mask],delimiter=self.maven.prefs['io.delimiter'])
				logger.info("Exported data %s"%(oname))
				return
		logger.error('There was a problem trying to export the traces')
