import os
import h5py
import time
import numpy as np
import logging
logger = logging.getLogger(__name__)

class smd_container(object):
	'''
	SMD (v2) class for single-molecule data
	=======================================
	This SMD class is the object representation of an SMD (v2) update to SMD.
	Data format is NxTx{Data...}
		- 100 molecules, 1000 timepoints, 16 recording channels = (100,1000,16)
		- 18 molecules, 1 timepoints, 4k pixel image = (18,1,64,64)
	'''
	def __init__(self):
		super().__init__()
		self.initialize_data()

	def __getattr__(self,name):
		if name == 'nmol':
			return self.raw.shape[0]
		elif name in ['ntime','nt']:
			return self.raw.shape[1]
		elif name in ['ncolors','ncolor']:
			return self.raw.shape[2]
		elif name == 'ndata':
			return self.raw.shape[2:]
		else:
			return object.__getattribute__(self, name)

	def initialize_data(self,rawdata=np.array(()).reshape((0,0,0))):
		'''
		Initialize all data in this `smd_container`

		Parameters
		----------
		rawdata : np.ndarray (int/float/double)
			There do not even have to be any raw data (i.e., it can be shape (0,0,0)). (NTD)

		Notes
		-----
		This function is used a lot to erase the container
		'''

		self.raw = rawdata
		self.source_index = np.zeros(self.nmol,dtype='int64')
		self.source_names = ['dataset']
		self.source_dicts = [{'date added':time.ctime()}]
		self.smd_dict = {'date_created':time.ctime(),'date_modified':time.ctime(),'description':'','format':'SMD'}

	def order(self,neworder):
		''' Reorder traces

		Reorders raw data, and index and source lists.  Can be used to reorder or remove traces from `container_fret`

		Parameters
		----------
		neworder : np.ndarray (bool/int)
			* if neworder is bool, then traces are just kept/removed.
			* if neworder is int, then data will be reordered.

		Notes
		-----
		Be careful dtype of neworder
		'''

		self.raw = self.raw[neworder]
		self.source_index = self.source_index[neworder]


def concatenate_smds(smd1,smd2):
	''' Combine two smd_containers into one other container. smd1 takes priority

	Parameters
	----------
	smd1 : `smd_container`
	smd2 : `smd_container`

	Returns
	-------
	out : new_container
		the concatenated data in a new smd object
	'''

	if smd1.ndata != smd2.ndata:
		raise Exception('Concatenate failure: Data dimensions are different shapes')

	nmol = smd1.nmol + smd2.nmol
	maxt = np.max((smd1.ntime,smd2.ntime))
	increment = len(smd1.source_names)

	out = smd_container()
	out.initialize_data(np.zeros((nmol,maxt,*smd1.ndata)))

	## 1
	out.raw[:smd1.nmol,:smd1.ntime] = smd1.raw.copy()
	out.source_index[:smd1.nmol] = smd1.source_index.copy()

	## 2
	out.raw[smd1.nmol:,:smd2.ntime] = smd2.raw.copy()
	out.source_index[smd1.nmol:] = smd2.source_index.copy() + increment

	out.source_names = smd1.source_names + smd2.source_names
	out.source_dicts = smd1.source_dicts + smd2.source_dicts
	out.smd_dict = {**smd2.smd_dict, **smd1.smd_dict}

	return out

def save_smd_in_hdf5(smd,file_name,group_name,overwrite=True,mask=None):
	''' Save a `smd_container` into an HDF5 file

	Parameters
	----------
	smd : smd_container
		data to be saved
	fname : str
		file name for the hdf5 file
	gname : str
		group name for the dataset to be entered in the hdf5 file
	overwrite : bool
		flag for whether to overwrite group in hdf5 file if it exists
	mask : None or ndarray(nmol,bool)
		bool of whether to include that molecule in smd in the hdf5 file

	Returns
	-------
	success : bool

	'''
	with h5py.File(file_name,'a') as f:
		if group_name in f:
			if overwrite:
				del f[group_name]
				f.flush()
			else:
				f.close()
				return False

		if mask is None:
			mask = np.ones(smd.nmol,dtype='bool')

		#### smd metadata
		g = f.create_group(group_name)
		g.attrs['format'] = smd.smd_dict['format']
		g.attrs['date_created'] = smd.smd_dict['date_created']
		smd.smd_dict['date_modified'] = time.ctime()
		g.attrs['date_modified'] = smd.smd_dict['date_modified']

		#### data
		gd = g.create_group('data')
		gd.attrs['description'] = smd.smd_dict['description']
		gd.create_dataset('raw',data=smd.raw[mask],compression='gzip')
		gd.create_dataset('source_index',data=smd.source_index[mask].astype(int),dtype='int64',compression='gzip')

		#### sources
		gs = g.create_group('sources')
		gs.attrs['source_list'] = str(smd.source_names)
		for i in range(len(smd.source_names)):
			# source = smd.source_names[i]
			source = '{}'.format(i)
			sd = smd.source_dicts[i]
			gsi = gs.create_group(source)
			for key in sd.keys():
				gsi.attrs[key] = sd[key]
			gsi.attrs['source_name'] = smd.source_names[i]

		f.flush()
		f.close()
		return True

def load_smd_in_hdf5(file_name,group_name):
	'''loads SMD/HDF5 file into `smd_container`

	Parameters
	----------
	fname : str
		file name for the hdf5 file
	gname : str
		group name for the dataset entry in the hdf5 file

	Returns
	-------
	success : bool
	out : smd_container
		the data you want

	'''
	from os.path import isfile
	from ast import literal_eval ## for safety instead of eval

	smd = smd_container()
	success = False

	if not isfile(file_name):
		logger.info('%s is not a file'%(file_name))
		return success,smd

	with h5py.File(file_name,'r') as f:
		if group_name in f:
			g = f[group_name]
			if g.attrs['format'] in ['smd','SMD','Smd']:
				## get data
				gd = g['data']
				smd.initialize_data(gd['raw'][:])

				for key in g.attrs:
					logger.info('attr: %s%s'%(key,g.attrs[key]))
					smd.smd_dict[key] = g.attrs[key]

				## protect against poorly formatted files
				if 'source' in gd.keys():
					smd.source_index = gd['source_index'][:].astype('int64')
				for key in gd.attrs:
					logger.info('attr: %s%s'%(key,gd.attrs[key]))
					smd.smd_dict[key] = gd.attrs[key]

				gs = g['sources']
				try:
					## everything needs to be in the correct order
					## so we are relying on source index
					smd.source_names = []
					smd.source_dicts = []
					order = np.argsort([int(k) for k in list(gs.keys())])
					keys = np.array(list(gs.keys()))[order]
					for key in keys:
						gsi = gs[key]
						dgsi = {}
						for akey in gsi.attrs.keys():
							dgsi[akey] = gsi.attrs[akey]
							logger.info('attr: %s%s'%(key,gsi.attrs[key]))
						smd.source_dicts.append(dgsi)
						smd.source_names.append(dgsi['source_name'])
				except Exception as e:
					keys = list(gs.keys())
					smd.source_dicts = [{} for _ in range(len(keys))]
					smd.source_names = literal_eval(gs.attrs['source_list'])
				success = True
			else:
				logger.info('%s is not smd format'%(file_name))
		f.close()
	return success,smd

def find_smds_in_hdf5(file_name):
	''' Search a (possibly) HDF5 file for all smd format entries '''
	out = []
	def visit_func(name, node) :
		if isinstance(node, h5py.Group) :
			if 'format' in node.attrs.keys():
				if node.attrs['format'].lower() == 'smd':
					out.append(node.name)
	try:
		with h5py.File(file_name,'r') as f:
			f.visititems(visit_func)
	except:
		pass
	return out
