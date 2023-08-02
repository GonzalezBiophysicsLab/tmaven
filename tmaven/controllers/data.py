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

	def split_trace(self,ind,loc):
		self.maven.smd.raw = np.concatenate([self.maven.smd.raw[:ind],self.maven.smd.raw[[ind,]].copy(),self.maven.smd.raw[[ind,]].copy(),self.maven.smd.raw[ind+1:]],axis=0)
		# self.maven.smd.raw[ind,loc:] *= np.nan
		# self.maven.smd.raw[ind+1,:loc] *= np.nan
		self.corrected = np.concatenate([self.corrected[:ind],self.corrected[[ind,]].copy(),self.corrected[[ind,]].copy(),self.corrected[ind+1:]],axis=0)
		# self.corrected[ind,loc:] *= np.nan
		# self.corrected[ind+1,:loc] *= np.nan
		self.classes = np.concatenate([self.classes[:ind],self.classes[[ind,]].copy(),self.classes[[ind,]].copy(),self.classes[ind+1:]],axis=0)
		self.data_index = np.concatenate([self.data_index[:ind],self.data_index[[ind,]].copy(),self.data_index[[ind,]].copy(),self.data_index[ind+1:]],axis=0)
		self.pre_list = np.concatenate([self.pre_list[:ind],self.pre_list[[ind,]].copy(),self.post_list[[ind,]].copy(),self.pre_list[ind+1:]],axis=0)
		self.post_list = np.concatenate([self.post_list[:ind],self.post_list[[ind,]].copy(),self.post_list[[ind,]].copy()*0+self.maven.data.nt,self.post_list[ind+1:]],axis=0)
		self.flag_ons = np.concatenate([self.flag_ons[:ind],self.flag_ons[[ind,]].copy(),self.flag_ons[[ind,]].copy(),self.flag_ons[ind+1:]],axis=0)
		self.maven.smd.source_index = np.concatenate([self.maven.smd.source_index[:ind],self.maven.smd.source_index[[ind,]].copy(),self.maven.smd.source_index[[ind,]].copy(),self.maven.smd.source_index[ind+1:]],axis=0)
		self.maven.io.process_data_change()
		print(self.raw.shape,self.nmol)

	def split_trace(self,ind,loc):
		### split ind in place 
		self.maven.smd.raw = np.concatenate([self.maven.smd.raw[:ind],self.maven.smd.raw[[ind,]].copy(),self.maven.smd.raw[[ind,]].copy(),self.maven.smd.raw[ind+1:]],axis=0)
		# self.maven.smd.raw[ind,loc:] *= np.nan
		# self.maven.smd.raw[ind+1,:loc] *= np.nan
		self.corrected = np.concatenate([self.corrected[:ind],self.corrected[[ind,]].copy(),self.corrected[[ind,]].copy(),self.corrected[ind+1:]],axis=0)
		# self.corrected[ind,loc:] *= np.nan
		# self.corrected[ind+1,:loc] *= np.nan
		self.classes = np.concatenate([self.classes[:ind],self.classes[[ind,]].copy(),self.classes[[ind,]].copy(),self.classes[ind+1:]],axis=0)
		self.data_index = np.concatenate([self.data_index[:ind],self.data_index[[ind,]].copy(),self.data_index[[ind,]].copy(),self.data_index[ind+1:]],axis=0)
		self.pre_list = np.concatenate([self.pre_list[:ind],self.pre_list[[ind,]].copy(),self.post_list[[ind,]].copy(),self.pre_list[ind+1:]],axis=0)
		self.post_list = np.concatenate([self.post_list[:ind],self.post_list[[ind,]].copy(),self.post_list[[ind,]].copy()*0+self.maven.data.nt,self.post_list[ind+1:]],axis=0)
		self.flag_ons = np.concatenate([self.flag_ons[:ind],self.flag_ons[[ind,]].copy(),self.flag_ons[[ind,]].copy(),self.flag_ons[ind+1:]],axis=0)
		self.maven.smd.source_index = np.concatenate([self.maven.smd.source_index[:ind],self.maven.smd.source_index[[ind,]].copy(),self.maven.smd.source_index[[ind,]].copy(),self.maven.smd.source_index[ind+1:]],axis=0)
		self.maven.io.process_data_change()
		logger.info('Split trace %d at frame %d'%(self.data_index[ind],loc))
		
	def combine_traces(self,ind1,ind2):
		## combine traces into ind1, then delete ind2
		self.maven.smd.raw[ind1] = self.maven.smd.raw[ind1].copy()
		# self.maven.smd.raw[ind1] = np.nanmean([self.maven.smd.raw[ind1],self.maven.smd.raw[ind2]],axis=0)
		self.maven.smd.source_index[ind1] = self.maven.smd.source_index[ind1]
		self.corrected[ind1] = self.corrected[ind1].copy()
		# self.corrected[ind1] = np.nanmean([self.corrected[ind1],self.corrected[ind2]],axis=0)
		self.classes[ind1] = self.classes[ind1]
		self.data_index[ind1] = self.data_index[ind1]
		self.pre_list[ind1] = 0#np.min(self.pre_list[[ind1,ind2]])
		self.post_list[ind1] = self.nt#np.max(self.post_list[[ind1,ind2]])
		self.flag_ons[ind1] = np.product(self.flag_ons[[ind1,ind2]])
		
		self.maven.smd.raw = np.delete(self.maven.smd.raw,ind2,axis=0)
		self.maven.smd.source_index = np.delete(self.maven.smd.source_index,ind2,axis=0)
		self.corrected = np.delete(self.corrected,ind2,axis=0)
		self.classes = np.delete(self.classes,ind2,axis=0)
		self.data_index = np.delete(self.data_index,ind2,axis=0)
		self.pre_list = np.delete(self.pre_list,ind2,axis=0)
		self.post_list = np.delete(self.post_list,ind2,axis=0)
		self.flag_ons = np.delete(self.flag_ons,ind2,axis=0)
		
		self.maven.io.process_data_change()
		logger.info('combined traces %d and %d'%(ind1,ind2))
	
	def collect_trace(self,ind):
		keep = self.data_index == self.data_index[ind]
		if keep.sum() < 2:
			logger.info('only one instance of %d(%d)'%(ind,self.data_index[ind]))
			return ind
		inds = np.sort(np.nonzero(keep)[0])
		for i in range(1,inds.size):
			self.combine_traces(inds[0],inds[inds.size-i])
		return inds[0]


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
