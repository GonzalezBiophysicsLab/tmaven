import numpy as np
import logging
logger = logging.getLogger(__name__)


default_prefs = {
}

class controller_data(object):
	''' Handles tMAVEN data parameters

	Parameters
	===================
	LINKS TO SMD
	------------
	.nmol
	.ntime,.nt
	.ncolors,.ncolor
	.ndata
	.raw

	HELD LOCALLY
	------------
	.corrected
	.classes
	.data_index
	.pre_list
	.post_list
	.flag_ons

	.idealized_ran
	.idealized

	'''

	def __init__(self,maven):
		self.maven = maven
		self.maven.prefs.add_dictionary(default_prefs)
		self.initialize_tmaven_params()

	def initialize_tmaven_params(self):
		self.corrected = self.raw.copy()
		self.classes  = np.zeros(self.nmol,dtype='int64')
		self.data_index = np.arange(self.nmol,dtype='int64')
		self.idealized_ran = None
		self.idealized = None
		self.pre_list = np.zeros(self.nmol,dtype='int')
		self.post_list = np.zeros(self.nmol,dtype='int') + self.ntime
		self.flag_ons = np.ones(self.nmol,dtype='bool')

	def update_tmaven_params(self):
		## when molecules have been added to the smd but the tMAVEN preferences aren't updated yet
		nmol0 = self.corrected.shape[0]
		if nmol0 == self.nmol:
			return
		elif nmol0 < self.nmol:
			corrected = self.raw.copy()
			classes = np.zeros(self.nmol,dtype='int64')
			data_index = np.arange(self.nmol,dtype='int64')
			pre_list = np.zeros(self.nmol,dtype='int')
			post_list = np.zeros(self.nmol,dtype='int') + self.ntime
			flag_ons = np.ones(self.nmol,dtype='bool')

			corrected[:nmol0,:self.corrected.shape[1]] = self.corrected.copy()
			classes[:nmol0] = self.classes.copy()
			data_index[:nmol0] = self.data_index.copy()
			pre_list[:nmol0] = self.pre_list.copy()
			post_list[:nmol0] = self.post_list.copy()
			flag_ons[:nmol0] = self.flag_ons.copy()

			self.corrected = corrected
			self.classes = classes
			self.data_index = data_index
			self.pre_list = pre_list
			self.post_list = post_list
			self.flag_ons = flag_ons
		else: ## something went really wrong...
			self.initialize_tmaven_params()

	def order(self,neworder):
		''' Reorder traces
		Parameters
		----------
		neworder : np.ndarray (bool/int)
			* if neworder is bool, then traces are just kept/removed.
			* if neworder is int, then data will be reordered.

		Notes
		-----
		Be careful dtype of neworder
		'''
		self.maven.smd.order(neworder)

		self.corrected = self.corrected[neworder]
		self.classes = self.classes[neworder]
		self.data_index = self.data_index[neworder]
		self.pre_list = self.pre_list[neworder]
		self.post_list = self.post_list[neworder]
		self.flag_ons = self.flag_ons[neworder]

		## I think there are probably some issues with this. eg bool vs int neworder etc.
		if not self.idealized_ran is None:
			idealized_ran = []
			for iri in self.idealized_ran:
				idealized_ran.append(np.argmax(iri == neworder))
			self.idealized_ran = idealized_ran

	def __getattr__(self,name):
		if name == 'raw':
			return self.maven.smd.raw
		elif name == 'nmol':
			return self.maven.smd.nmol
		elif name in ['ntime','nt']:
			return self.maven.smd.nt
		elif name in ['ncolors','ncolor']:
			return self.maven.smd.ncolor
		elif name == 'ndata':
			return self.maven.smd.ndata
		else:
			return object.__getattribute__(self, name)
