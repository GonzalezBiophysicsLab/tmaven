import numpy as np
import h5py
import time
import os
import logging
logger = logging.getLogger(__name__)
# from ..maven import check_mavengui_present

default_prefs = {
	'io.datadims':2,
	'io.skiprows':0,
	'io.delimiter':r'\t',
	'io.axis_order':[0,1,2],
	'io.missing_axis':2,
	'io.decollate':1,
	'io.decollate_axis':1,
	'io.force_double':True,
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
		self.find_smds_in_hdf5 = find_smds_in_hdf5

	def add_data(self,new_smd,new_tmaven=None):
		if not new_smd is None:
			if self.maven.smd.raw.shape == (0,0,0):
				self.maven.smd = new_smd
				self.maven.data.initialize_tmaven_params()
			else:
				self.maven.smd = concatenate_smds(self.maven.smd,new_smd)
				self.maven.data.update_tmaven_params()
		if not new_tmaven is None:
			classes, pre_list, post_list = new_tmaven
			if not classes is None:
				self.maven.data.classes[-classes.size:] = classes.copy()
			if not pre_list is None:
				self.maven.data.pre_list[-pre_list.size:] = pre_list.copy()
			if not post_list is None:
				self.maven.data.post_list[-post_list.size:] = post_list.copy()
		self.process_data_change()

	def process_data_change(self):
		self.maven.data.update_tmaven_params()
		self.maven.emit_data_update()

	def clear_data(self):
		self.maven.smd = smd_container()
		self.process_data_change()

	def blank_smd(self):
		return smd_container()

	def load_smdtmaven_hdf5(self,fname,gname):
		smd = self.load_smd_hdf5(fname,gname)
		if self.maven.prefs['io.force_double']:
			smd.raw = smd.raw.astype('double')
		tmaven = self.load_tmaven_hdf5(fname,gname)
		self.add_data(smd,tmaven)
		self.maven.emit_data_update()

	def load_smd_hdf5(self,fname,gname):
		try:
			success,smd = load_smd_in_hdf5(fname,gname)
			if success:
				return smd
		except Exception as e:
			logger.error('smdload failed {}/{}\nerror:{}'.format(fname,gname,str(e)))
		raise Exception('smdload was not successful {}/{}'.format(fname,gname))

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

	def load_raw_hdf5(self,fname,dataset):
		success = False
		try:
			with h5py.File(fname,'r') as f:
				dat = f[dataset][:]
			logger.info('loaded HDF5 {}. dataset {}'.format(fname, dataset))
			smd = self.convert_to_smd(dat,dataset)
			smd.source_names[0] = '{}:{}'.format(fname,dataset)
			self.add_data(smd, None)
			self.maven.emit_data_update()
		except Exception as e:
			logging.error('failed to load %s\n%s'%(fname,str(e)))
			return

	def load_raw_txt(self,fname,skiprows,delimiter):
		try:
			if delimiter == r'\t':
				delimiter = '\t'
			d = np.loadtxt(fname,skiprows=skiprows,delimiter=delimiter)
			logger.info('loaded txt {}. skiprows {}, delimiter {}'.format(fname, skiprows, delimiter))
			smd = self.convert_to_smd(d,fname)
			smd.source_names[0] = '{}'.format(fname)
			self.add_data(smd, None)
			self.maven.emit_data_update()
		except Exception as e:
			logger.error('txt load failed {}\n{}'.format(fname,str(e)))

	def load_raw_numpy(self,fname):
		try:
			d = np.load(fname)
			logger.info('loaded np {}'.format(fname))
			smd = self.convert_to_smd(d,fname)
			smd.source_names[0] = '{}'.format(fname)
			self.add_data(smd, None)
			self.maven.emit_data_update()
		except Exception as e:
			logger.error('np load failed {}\n{}'.format(fname,str(e)))

	def load_spartan(self, fname):
		from .io_special.io_spartan import read_spartan

		try:
			chNames,time,channels,metadata = read_spartan(fname)
			d = channels[:2,:,:]
			self.maven.prefs['io.axis_order'] = [1,2,0]
			smd = self.convert_to_smd(d,fname) #[1,2,0]
			print(smd.raw.shape)
			smd.source_names[0] = '{}'.format(fname)
			smd.time = time
			smd.chNames = chNames[:2]
			smd.__dict__.update(metadata)
			self.add_data(smd, None)
			self.maven.emit_data_update()

		except Exception as e:
			logger.error('spartan load failed {}\n{}'.format(fname,str(e)))

	def load_tmaven_all_hdf5(self, fname, datasets):
		try:
			temp = []
			with h5py.File(fname,'r') as f:
				for i in datasets:
					if not i is None:
						temp.append(f[i][:].astype('int'))
					else:
						temp.append(None)

			tmaven = (temp[0], temp[1],temp[2])
			self.add_data(None, tmaven)
			self.maven.emit_data_update()

		except Exception as e:
			logging.error('failed to load %s\n%s'%(fname,str(e)))
			return

	def load_tmaven_all_txt(self, fname, skiprows, delimiter, d_name):
		temp = np.loadtxt(fname,skiprows=skiprows,delimiter=delimiter)

		if d_name == "all":
			classes = temp[:, 0].astype('int')
			pre_list = temp[:, 1].astype('int')
			post_list = temp[:, 2].astype('int')
		elif d_name == "class":
			classes = temp[:, 0].astype('int')
			pre_list = None
			post_list = None
		elif d_name == "pre-post":
			classes = None
			pre_list = temp[:, 1].astype('int')
			post_list = temp[:, 2].astype('int')

		tmaven = (classes, pre_list, post_list)
		self.add_data(None, tmaven)
		self.maven.emit_data_update()

	def convert_to_smd(self,d,dname):
		#datadims = self.maven.prefs['io.datadims']
		order = self.maven.prefs['io.axis_order']
		missing = self.maven.prefs['io.missing_axis']
		decollate = self.maven.prefs['io.decollate']
		decollate_axis = self.maven.prefs['io.decollate_axis']


		d,success = self.fix_decollate(d,decollate,decollate_axis)
		if not success:
			logger.error('Dataset load failed {} because of bad decollating'.format(dname))
			return None
		d,success = self.fix_missing_dimensions(d,missing)
		if not success:
			logger.error('Dataset load failed {} because of bad missing dimension'.format(dname))
			return None
		d,success = self.fix_axis_order(d,order)
		if not success:
			logger.error('Dataset load failed {} because of bad ordering'.format(dname))
			return None

		smd = self.blank_smd()
		smd.initialize_data(d)
		return smd

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
		elif d.ndim == 4:
			#this is a hard-coded solution for vbscope hdf5 files
			d = d[:,:,:,0]
			logger.info('removed extra dimension')
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
		mask = self.maven.selection.get_toggled_mask()
		try:
			with h5py.File(oname,'a') as f:
				g = f[gname]
				if 'tMAVEN' in g:
					del g['tMAVEN']
					f.flush()
				gd = g.create_group('tMAVEN')
				gd.attrs['format'] = 'tMAVEN'
				gd.attrs['date_modified'] = time.ctime()
				gd.create_dataset('classes',data=self.maven.data.classes[mask].astype('int64'),dtype='int64',compression='gzip')
				gd.create_dataset('pre_list',data=self.maven.data.pre_list[mask].astype('int64'),dtype='int64',compression='gzip')
				gd.create_dataset('post_list',data=self.maven.data.post_list[mask].astype('int64'),dtype='int64',compression='gzip')
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
				mask = self.maven.selection.get_toggled_mask()
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
				mask = self.maven.selection.get_toggled_mask()
				success = np.save(oname,self.maven.smd.raw[mask])
				logger.info("Exported data %s"%(oname))
				return
		logger.error('There was a problem trying to export the traces')

	def export_class_numpy(self,oname):
		if not self.maven.smd.nmol == 0:
			if not oname == "":
				mask = self.maven.selection.get_toggled_mask()
				success = np.save(oname,self.maven.data.classes[mask])
				logger.info("Exported data %s"%(oname))
				return
		logger.error('There was a problem trying to export the traces')

	def export_class_txt(self,oname):
		if not self.maven.smd.nmol == 0:
			if not oname == "":
				mask = self.maven.selection.get_toggled_mask()
				success = np.savetxt(oname,self.maven.data.classes[mask],delimiter=self.maven.prefs['io.delimiter'])
				logger.info("Exported data %s"%(oname))
				return
		logger.error('There was a problem trying to export the traces')
