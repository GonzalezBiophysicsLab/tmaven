import numpy as np
import logging
logger = logging.getLogger(__name__)
import types
import h5py as h
from .model_container import model_container

default_prefs = {
	'modeler.nrestarts':4,
	'modeler.converge':1e-10,
	'modeler.maxiters':1000,
	'modeler.vbgmm.prior.beta':0.25,
	'modeler.vbgmm.prior.a':0.1,
	'modeler.vbgmm.prior.b':0.01,
	'modeler.vbgmm.prior.pi':1,
	'modeler.vbhmm.prior.beta':0.25,
	'modeler.vbhmm.prior.a':2.5,
	'modeler.vbhmm.prior.b':0.01,
	'modeler.vbhmm.prior.alpha':1.,
	'modeler.vbhmm.prior.pi':1,
	'modeler.vbconhmm.prior.beta':0.25,
	'modeler.vbconhmm.prior.a':2.5,
	'modeler.vbconhmm.prior.b':0.01,
	'modeler.vbconhmm.prior.alpha':1.,
	'modeler.vbconhmm.prior.pi':1.,
	'modeler.ebhmm.prior.beta':0.25,
	'modeler.ebhmm.prior.a':2.5,
	'modeler.ebhmm.prior.b':0.01,
	'modeler.ebhmm.prior.alpha':1.,
	'modeler.ebhmm.prior.pi':1.,
	'modeler.hhmm.tolerance': 1e-4,
    'modeler.hhmm.maxiters':100,
    'modeler.hhmm.restarts':2
}

def export_dict_to_group(h_group, dicty, attributes=[]):
	for k in dicty.keys():
		if k in attributes:
			pass
		#print(k)
		elif not dicty[k] is None:
			if isinstance(dicty[k], dict):
				hh_group = h_group.create_group(k)
				export_dict_to_group(hh_group, dicty[k])
			elif np.isscalar(dicty[k]):
				h_group.create_dataset(k,data=dicty[k])
			elif not isinstance(dicty[k],types.FunctionType):
				h_group.create_dataset(k,data=dicty[k],compression='gzip')


def load_group_to_dict(h_group):
	dicty = {}
	for key, item in h_group.items():
		if isinstance(item, h._hl.dataset.Dataset):
			dicty[key] = item[()]
		elif isinstance(item, h._hl.group.Group):
			dicty[key] = load_group_to_dict(item)
	return dicty

class controller_modeler(object):
	''' Handles modeling data

	* several .get_* functions to help setup models
	* several model specific io functions (save,load)
	* direct modeling functions (should basically not know about a GUI/maven. input is all relevant variables etc.). Individual modes will get the relevant data together and then call these functions which will dump the result into .result


	### TODO:
		cache compile in a separate thread fxns that need to be numba'd?
	'''

	def __init__(self,maven):
		super().__init__()
		self.maven = maven
		self.maven.prefs.add_dictionary(default_prefs)
		self.cached_functions = {}
		self.model_container = model_container ## for easy access to blank containers elsewhere

		self.models = []
		self._active_model_index = None
		#### you can get and set maven.modeler.model and maven.modeler.idealized to get the active model
		## set self.model = None or to a model_container. Or use self.set_model with an int index to get those in self.models
		## self.idealized will grab the idealized data in the active model OR if it's the wrong size or non-existant, will provide a NaN array of shape maven.data.raw
		## setting self.idealized will overwrite the idealized data on the active model... or do nothing.

	def emit_model_update(self):
		pass

	def __getattr__(self,name):
		if name == 'model':
			if self._active_model_index is None:
				return None
			elif self._active_model_index < len(self.models):
				return self.models[self._active_model_index]
			else:
				return None
		elif name == 'idealized':
			if not self.model.idealized is None:
				if self.maven.data.corrected.shape == self.model.idealized.shape:
					return self.model.idealized
			return np.zeros_like(self.maven.data.corrected)+np.nan
		else:
			return object.__getattribute__(self, name)

	def __setattr__(self, name, value):
		if name == 'model':
			if value is None:
				self._active_model_index = None
			elif not (type(value) is model_container):
				return
			else:
				if not value in self.models:
					self.models.append(value)
				self._active_model_index = self.models.index(value)
		elif name == 'idealized':
			if not self.model is None:
				self.model.idealized = value
		else:
			self.__dict__[name] = value

	def set_model(self,i):
		n = len(self.models)
		if i is None:
			self._active_model_index = None
			self.maven.emit_data_update()
		elif n > 0 and i < n:
			self._active_model_index = i
			self.maven.emit_data_update()

	def remove_models(self,indexes):
		if not type(indexes) is list:
			logger.info('No models removed, because {} is not a list'.format(indexes))
			return

		if self._active_model_index is None:
			current = None
		elif self._active_model_index in indexes:
			current = None
		else:
			current = self.models[self._active_model_index]
		self.models = [self.models[i] for i in range(len(self.models)) if not i in indexes]
		if not current is None:
			self._active_model_index = self.models.index(current)
		self.maven.emit_data_update()

	def add_models(self,models):
		if not type(indexes) is list:
			logger.info('No models added, because {} is not a list'.format(models))
			return
		self.models.append(models)
		self._active_model_index = len(self.models)-1
		self.maven.emit_data_update()

	def export_result_to_hdf5(self,fn):
		import time

		result = self.model
		print(result.type)
		model_dict = result.convert_to_dict()
		print(result.type)


		with h.File(fn,'a') as f:

			if 'model' in f:
				del f['model']
				f.flush()

			g = f.create_group('model')

			attributes = ['type', 'time_made','rate_type']

			for atts in attributes:
				g.attrs[atts] = model_dict[atts]

			g.attrs['time_modified'] = time.ctime()

			export_dict_to_group(g, model_dict, attributes)

			f.flush()
			f.close()

		logger.info('saved result in {}'.format(fn))
		print(result)
		return True

	def load_result_from_hdf5(self,fn):
		from os.path import isfile

		success = False
		result = None

		if not isfile(fn):
			logger.info('%s is not a file'%(fn))
			return success,result

		with h.File(fn,'r') as f:
			#try:
			if 1:
				g = f['model']

				result_dict = load_group_to_dict(g)

				for key, item in g.attrs.items():
					result_dict[key] = item

				type = result_dict['type']

				from .model_container import model_container

				result = model_container(type = type)

				result.convert_from_dict(result_dict)

				if result.type in ["threshold","Threshold"]:
					result.idealize = lambda : self.idealize_fret_threshold(result)
				elif result.type in ["kmeans","k-means","Kmeans"]:
					result.idealize = lambda : self.idealize_fret_kmeans(result)
				elif result.type in ["ml GMM","ML GMM","vb GMM","VB GMM"]:
					result.idealize = lambda : self.idealize_fret_gmm(result)
				elif result.type in ["VB Consensus HMM",'vb Consensus HMM']:
					result.idealize = lambda : self.idealize_fret_threshold(result)

				success = True
				self.model = result

				logger.info('loaded result from {}'.format(fn))
				self.make_report(result)
				self.maven.emit_data_update()

			#except:
			else:
				logger.info('%s is not kmd format'%(fn))

			f.close()

		return success,result

	def update_idealization(self):
		try:
			self.model.idealize(self.maven)
			logger.info('Updated idealization for active model')
		except Exception as e:
			logger.info('Failed to update idealization for active model\n{}'.format(e))
		self.maven.emit_data_update()

	def get_traces(self):
		checked = self.maven.data.flag_ons
		return checked

	def get_fret_traces(self):
		keep = self.get_traces()
		if keep.sum() == 0:
			logger.info('Failed to get traces')
			return False,keep,[]
		pre,post = self.get_prepost()
		notshort = post-pre > 2
		keep = keep[notshort]
		y = [self.maven.calc_fret(i)[pre[i]:post[i],1].astype('double').flatten() for i in range(keep.size) if keep[i]]
		y = self.clip_traces(y,-1,2)
		return True,keep,y

	def get_prepost(self):
		pre = self.maven.data.pre_list
		post = self.maven.data.post_list
		return pre,post

	def get_model_descriptions(self):
		items = [m.description() for m in self.models]
		try:
			items[self._active_model_index] = '>>> '+items[self._active_model_index]
		except:
			pass
		return items

	def get_survival_dwells(self, state):
		result = self.model
		dwells_model = result.dwells
		from .dwells import survival
		tau, survival_norm = survival(np.array(dwells_model[str(state)]))
		return tau, survival_norm


	def make_report(self,model):
		#model_dict = model.__dict__

		#for i in ['idealize','idealized','ran']:
			#model_dict.pop(i)

		type = model.type
		s = '\nModel type = {}\n'.format(type)
		#model_dict.pop('type')

		N = len(model.ran)
		s += 'Model ran on {} traces\n'.format(N)

		nstates = model.nstates
		s += 'nstates = {}\n'.format(nstates)
		#model_dict.pop('nstates')

		mean = model.mean
		s += 'means = {}\n'.format(mean)
		#model_dict.pop('mean')

		var = model.var
		s += 'vars = {}\n'.format(var)
		#model_dict.pop('var')

		frac = model.frac
		s += 'fracs = {}\n'.format(frac)
		#model_dict.pop('nstates')

		if not model.tmatrix is None:
			tmatrix = model.tmatrix
			norm_tmatrix = model.norm_tmatrix
			s += 'tmatrix = \n{}\n'.format(tmatrix)
			s += 'tmatrix normalized = \n{}\n'.format(norm_tmatrix)

		if not model.rates is None:
			rate_type = model.rate_type
			rates = model.rates
			s += 'Rate type = {}\n'.format(rate_type)
			s += 'Rates = \n{}\n'.format(rates)

		logger.info(s)

	def clip_traces(self,y,low=-1,high=2):
		np.random.seed(666)
		for i in range(len(y)):
			yy = y[i]
			## Clip traces and redistribute randomly
			bad = np.bitwise_or((yy < -1.),np.bitwise_or((yy > 2),np.isnan(yy)))
			yy[bad] = np.random.uniform(low=low,high=high,size=int(bad.sum()))
			y[i] = yy
		import time
		np.random.seed(int(time.time()*1000)%(2**32-1))
		return y

	def recast_rs(self,result):
		data = self.maven.calc_fret()[:,:,1]
		N,T = data.shape
		rs = result.r
		result.r = np.zeros((N,T,result.nstates)) + np.nan
		for i in range(N):
			ii = result.ran[i]
			pre = self.maven.data.pre_list[ii]
			post = self.maven.data.post_list[ii]
			result.r[ii,pre:post] = rs[i]

	def cached_vbgmm(self,y,priors,nstates,maxiters,converge,nrestarts,ncpu):
		''' Run individual VB GMM
		priors is a 1D np.ndarray of eg [beta,a,b,alpha] used for all states
		'''
		if not 'VB GMM' in self.cached_functions:
			logger.info('Caching VB GMM...')
			from .gmm_vb import vb_em_gmm,vb_em_gmm_parallel
			self.cached_functions['VB GMM'] = vb_em_gmm_parallel

		result = self.cached_functions['VB GMM'](y,nstates,maxiters=maxiters,threshold=converge,nrestarts=nrestarts,prior_strengths=priors,ncpu=ncpu)
		return result

	def cached_mlgmm(self,y,nstates,maxiters,converge,nrestarts,ncpu):
		''' Run individual ML GMM '''
		if not 'ML GMM' in self.cached_functions:
			logger.info('Caching ML GMM...')
			from .gmm_ml import ml_em_gmm,ml_em_gmm_parallel
			self.cached_functions['ML GMM'] = ml_em_gmm_parallel

		# ## has outlier class -- remove last point
		result = self.cached_functions['ML GMM'](y,nstates,maxiters=maxiters,threshold=converge,nrestarts=nrestarts,ncpu=ncpu)
		# result.r = result.r[:,:-1]
		# result.mu = result.mu[:-1]
		# result.var = result.var[:-1]
		# result.ppi = result.ppi[:-1]
		return result

	def cached_vbconhmm(self,y,priors,nstates,maxiters,converge,nrestarts,ncpu):
		''' Run consensus vb hmm
		y should be a list?
		priors is a 1D np.ndarray of eg [beta,a,b,pi,alpha] used for all states
		'''
		if not 'VB CON HMM' in self.cached_functions:
			logger.info('Caching VB CON HMM...')
			from .hmm_vb_consensus import consensus_vb_em_hmm, consensus_vb_em_hmm_parallel
			self.cached_functions['VB CON HMM'] = consensus_vb_em_hmm_parallel

		result = self.cached_functions['VB CON HMM'](y,nstates,maxiters=maxiters,threshold=converge,nrestarts=nrestarts,priors=priors,ncpu=ncpu,init_kmeans=True)
		return result

	def cached_vbhmm(self,y,priors,nstates,maxiters,converge,nrestarts,ncpu):
		''' Run individual VB HMM
		priors is a 1D np.ndarray of eg [beta,a,b,pi,alpha] used for all states
		'''
		if not 'VB HMM' in self.cached_functions:
			logger.info('Caching VB HMM...')
			from .hmm_vb import vb_em_hmm,vb_em_hmm_parallel
			self.cached_functions['VB HMM'] = vb_em_hmm_parallel

		result = self.cached_functions['VB HMM'](y,nstates,maxiters=maxiters,threshold=converge,nrestarts=nrestarts,priors=priors,ncpu=ncpu,init_kmeans=True)
		return result

	def cached_ebhmm(self,y,priors,nstates,maxiters,converge,nrestarts,ncpu):
		''' Run ensemble EB HMM
		priors is a 1D np.ndarray of eg [beta,a,b,pi,alpha] used for all states
		'''
		if not 'EB HMM' in self.cached_functions:
			logger.info('Caching EB HMM...')
			from .hmm_eb import eb_em_hmm
			self.cached_functions['EB HMM'] = eb_em_hmm

		result = self.cached_functions['EB HMM'](y,nstates,maxiters=maxiters,threshold=converge,nrestarts=nrestarts,priors=priors,ncpu=ncpu,init_kmeans=True)
		return result

	def cached_mlhmm(self,y,nstates,maxiters,converge,nrestarts,ncpu):
		''' Run individual ML HMM '''
		if not 'ML HMM' in self.cached_functions:
			logger.info('Caching ML HMM...')
			from .hmm_ml import ml_em_hmm, ml_em_hmm_parallel
			self.cached_functions['ML HMM'] = ml_em_hmm_parallel

		result = self.cached_functions['ML HMM'](y,nstates,maxiters=maxiters,threshold=converge,nrestarts=nrestarts,ncpu=ncpu)
		return result

	def cached_threshold(self,y,threshold):
		''' Idealize an np.ndarray with a threshold into classes 0 (<) and 1 (>)'''

		#nstates = 2
		yclass = (y > threshold).astype('int')
		mean = np.array([y[yclass==j].mean() for j in [0,1]]).astype('double')
		var = np.array([y[yclass==j].var() for j in [0,1]]).astype('double')
		frac = np.array([(yclass==j).sum() for j in [0,1]]).astype('double')
		frac /= frac.sum()

		result = model_container(type='threshold',
								 nstates=2,mean=mean,var=var,frac=frac,
								 threshold=threshold)
		return result

	def cached_kmeans(self,y,nstates):
		''' Cluster a 1D np.ndarray using K-means. Will flatten if not 1d'''
		if not 'kmeans' in self.cached_functions:
			logger.info('Caching K-means...')
			from .kmeans import kmeans
			self.cached_functions['kmeans'] = kmeans

		result = self.cached_functions['kmeans'](y,nstates)
		return result
	
	def cache_vmphhmm(self,y,depth_vec,prod_states,max_iter,tol,restarts,guess):
		''' Run vmp hhmm'''
		if not 'hHMM' in self.cached_functions:
			logger.info('Caching hHMM...')
			from .hhmm_vmp import vmp_hhmm
			self.cached_functions['hHMM'] = vmp_hhmm

		result = vmp_hhmm(y,depth_vec,prod_states,max_iter,tol,restarts,guess)
		return result


	def cache_exponential_fxn(self,type):
		from .dwells import optimize_single_surv, optimize_double_surv, optimize_triple_surv, optimize_stretch_surv
		if type == 'Single Exponential':
			self.cached_functions[type] = optimize_single_surv
		if type == 'Double Exponential':
			self.cached_functions[type] = optimize_double_surv
		if type == 'Triple Exponential':
			self.cached_functions[type] = optimize_triple_surv
		if type == 'Stretched Exponential':
			self.cached_functions[type] = optimize_stretch_surv

	def run_fret_kmeans(self,nstates):
		success, keep, y = self.get_fret_traces()
		if not success:
			return

		result = self.cached_kmeans(np.concatenate(y),nstates)
		result.ran = np.nonzero(keep)[0].tolist()
		result.idealize = lambda : self.idealize_fret_kmeans(result)
		result.idealize()
		self.recast_rs(result)
		self.model = result
		self.make_report(result)
		self.maven.emit_data_update()

	def run_fret_vbgmm(self,nstates):
		success, keep, y = self.get_fret_traces()
		if not success:
			return

		maxiters = self.maven.prefs['modeler.maxiters']
		converge = self.maven.prefs['modeler.converge']
		nrestarts = self.maven.prefs['modeler.nrestarts']
		ncpu = self.maven.prefs['ncpu']
		priors = np.array([self.maven.prefs[sss] for sss in ['modeler.vbgmm.prior.beta','modeler.vbgmm.prior.a','modeler.vbgmm.prior.b','modeler.vbgmm.prior.pi']])

		result = self.cached_vbgmm(np.concatenate(y),priors,nstates,maxiters,converge,nrestarts,ncpu)
		result.ran = np.nonzero(keep)[0].tolist()
		result.idealize = lambda : self.idealize_fret_gmm(result)
		result.idealize()
		self.recast_rs(result)
		self.model = result
		self.make_report(result)
		self.maven.emit_data_update()

	def run_fret_vbgmm_modelselection(self,nstates_min,nstates_max):
		success, keep, y = self.get_fret_traces()
		if not success:
			return

		maxiters = self.maven.prefs['modeler.maxiters']
		converge = self.maven.prefs['modeler.converge']
		nrestarts = self.maven.prefs['modeler.nrestarts']
		ncpu = self.maven.prefs['ncpu']
		priors = np.array([self.maven.prefs[sss] for sss in ['modeler.vbgmm.prior.beta','modeler.vbgmm.prior.a','modeler.vbgmm.prior.b','modeler.vbgmm.prior.pi']])

		results = []
		for nstates in range(nstates_min,nstates_max+1):
			result = self.cached_vbgmm(np.concatenate(y),priors,nstates,maxiters,converge,nrestarts,ncpu)
			result.ran = np.nonzero(keep)[0].tolist()
			result.idealize = lambda : self.idealize_fret_gmm(result)
			result.idealize()
			self.recast_rs(result)
			results.append(result)

		elbos = np.array([ri.likelihood[-1,0] for ri in results])
		modelmax = np.argmax(elbos)
		logger.info('vbgmm - best elbo: %f, nstates=%d'%(elbos[modelmax],results[modelmax].nstates))
		for i in range(len(results)):
			self.models.append(results[i])
			if i == modelmax:
				self._active_model_index = len(self.models)-1
				self.make_report(results[i])

		#self.model = result
		self.maven.emit_data_update()

	def run_fret_mlgmm(self,nstates):
		success, keep, y = self.get_fret_traces()
		if not success:
			return

		maxiters = self.maven.prefs['modeler.maxiters']
		converge = self.maven.prefs['modeler.converge']
		nrestarts = self.maven.prefs['modeler.nrestarts']
		ncpu = self.maven.prefs['ncpu']

		result = self.cached_mlgmm(np.concatenate(y),nstates,maxiters,converge,nrestarts,ncpu)
		result.ran = np.nonzero(keep)[0].tolist()
		result.idealize = lambda : self.idealize_fret_gmm(result)
		result.idealize()
		self.recast_rs(result)
		self.model = result
		self.make_report(result)
		self.maven.emit_data_update()

	def run_fret_mlhmm(self,nstates):
		success,keep,y = self.get_fret_traces()
		if not success:
			return

		maxiters = self.maven.prefs['modeler.maxiters']
		converge = self.maven.prefs['modeler.converge']
		nrestarts = self.maven.prefs['modeler.nrestarts']
		ncpu = self.maven.prefs['ncpu']

		from .fxns.hmm import viterbi
		from .model_container import trace_model_container

		mu = np.ones(nstates)*np.nan
		var = mu.copy()
		frac = mu.copy()
		result = self.model_container(type='ml HMM',nstates=nstates,mean=mu,var =var,frac=frac)
		data = self.maven.calc_fret()[:,:,1]
		result.idealized = np.zeros_like(data) + np.nan
		trace_level = {}
		result.ran = np.nonzero(keep)[0].tolist()

		for i in range(len(y)):
			yi = y[i].astype('double')
			r = self.cached_mlhmm(yi,nstates,maxiters,converge,nrestarts,ncpu)
			ii = result.ran[i]
			pre = self.maven.data.pre_list[ii]
			post = self.maven.data.post_list[ii]
			result.idealized[ii,pre:post] = r.mean[viterbi(yi,r.mean,r.var,r.tmatrix,r.frac).astype('int')]
			trace_level_inst = trace_model_container(r, ii)
			trace_level_inst.idealized = result.idealized[ii]
			trace_level[str(ii)] = trace_level_inst

		result.trace_level = trace_level
		self.model = result
		self.make_report(result)
		self.maven.emit_data_update()

	def run_fret_kmeans_mlhmm(self,nstates):
		success, keep, y = self.get_fret_traces()
		if not success:
			return

		maxiters = self.maven.prefs['modeler.maxiters']
		converge = self.maven.prefs['modeler.converge']
		nrestarts = self.maven.prefs['modeler.nrestarts']
		ncpu = self.maven.prefs['ncpu']

		from .fxns.hmm import viterbi
		from .model_container import trace_model_container

		data = self.maven.calc_fret()[:,:,1]
		idealized = np.zeros_like(data) + np.nan
		ran = np.nonzero(keep)[0].tolist()

		trace_level = {}

		for i in range(len(y)):
			yi = y[i].astype('double')
			r = self.cached_mlhmm(yi,nstates,maxiters,converge,nrestarts,ncpu)
			ii = ran[i]
			pre = self.maven.data.pre_list[ii]
			post = self.maven.data.post_list[ii]
			vit = r.mean[viterbi(yi,r.mean,r.var,r.tmatrix,r.frac).astype('int')]
			idealized[ii,pre:post] = vit
			trace_level_inst = trace_model_container(r, ii)
			trace_level_inst.idealized = idealized[ii]
			trace_level[str(ii)] = trace_level_inst

		vits = np.concatenate(idealized)
		vits = vits[np.isfinite(vits)]
		result = self.cached_kmeans(vits,nstates)
		result.trace_level = trace_level
		result.type = "kmeans + ml HMM"
		result.ran = ran

		result.rate_type = "Transition Matrix"
		from .fxns.hmm import compose_tmatrix, normalize_tmatrix,convert_tmatrix 
		tmatrix = compose_tmatrix(y,result)
		result.tmatrix = tmatrix
		result.norm_tmatrix = normalize_tmatrix(tmatrix)
		result.rates = convert_tmatrix(tmatrix)

		result.idealize = lambda : self.idealize_fret_kmeans_viterbi(result,idealized)
		result.idealize()
		self.recast_rs(result)
		self.model = result
		self.make_report(result)
		self.maven.emit_data_update()

	def run_fret_mlhmm_one(self,nstates):
		success,keep,y = self.get_fret_traces()
		if not success:
			return
			
		if keep.sum() != 1:
			#Warning "Error: Run all, Apply all","You have more than one molecule turned on")
			return
		maxiters = self.maven.prefs['modeler.maxiters']
		converge = self.maven.prefs['modeler.converge']
		nrestarts = self.maven.prefs['modeler.nrestarts']
		ncpu = self.maven.prefs['ncpu']

		result = self.cached_mlhmm(y[0].astype('double'),nstates,maxiters,converge,nrestarts,ncpu)
		result.ran = [np.nonzero(keep)[0]]
		result.idealize = lambda : self.idealize_fret_hmm(result)
		result.idealize()
		self.model = result
		self.maven.emit_data_update()

	def run_fret_threshold(self,threshold):
		success,keep,y = self.get_fret_traces()
		if not success:
			return
			
		result = self.cached_threshold(np.concatenate(y),threshold)
		result.ran = np.nonzero(keep)[0].tolist()
		result.idealize = lambda : self.idealize_fret_threshold(result)
		result.idealize()
		self.model = result
		self.make_report(result)
		self.maven.emit_data_update()

	def run_fret_vbconhmm(self,nstates):
		success,keep,y = self.get_fret_traces()
		if not success:
			return

		maxiters = self.maven.prefs['modeler.maxiters']
		converge = self.maven.prefs['modeler.converge']
		nrestarts = self.maven.prefs['modeler.nrestarts']
		ncpu = self.maven.prefs['ncpu']

		mu_prior = np.percentile(np.concatenate(y),np.linspace(0,100,nstates+2))[1:-1]
		beta_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.vbconhmm.prior.beta']
		a_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.vbconhmm.prior.a']
		b_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.vbconhmm.prior.b']
		pi_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.vbconhmm.prior.pi']
		tm_prior = np.ones((nstates,nstates))*self.maven.prefs['modeler.vbconhmm.prior.alpha']

		priors = [mu_prior, beta_prior, a_prior, b_prior, pi_prior, tm_prior]

		result = self.cached_vbconhmm(y,priors,nstates,maxiters,converge,nrestarts,ncpu)

		result.ran = np.nonzero(keep)[0].tolist()
		result.idealize = lambda : self.idealize_fret_hmm(result)
		result.idealize()
		self.recast_rs(result)

		self.model = result
		self.make_report(result)
		self.maven.emit_data_update()

	def run_fret_vbconhmm_modelselection(self,nstates_min,nstates_max):
		if nstates_min > nstates_max:
			logger.info('nstates min > max')
			return

		success,keep,y = self.get_fret_traces()
		if not success:
			return

		maxiters = self.maven.prefs['modeler.maxiters']
		converge = self.maven.prefs['modeler.converge']
		nrestarts = self.maven.prefs['modeler.nrestarts']
		ncpu = self.maven.prefs['ncpu']

		data = self.maven.calc_fret()[:,:,1]
		results = []
		for nstates in range(nstates_min,nstates_max+1):
			mu_prior = np.percentile(np.concatenate(y),np.linspace(0,100,nstates+2))[1:-1]
			beta_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.vbconhmm.prior.beta']
			a_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.vbconhmm.prior.a']
			b_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.vbconhmm.prior.b']
			pi_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.vbconhmm.prior.pi']
			tm_prior = np.ones((nstates,nstates))*self.maven.prefs['modeler.vbconhmm.prior.alpha']

			priors = [mu_prior, beta_prior, a_prior, b_prior, pi_prior, tm_prior]

			result = self.cached_vbconhmm(y,priors,nstates,maxiters,converge,nrestarts,ncpu)
			result.idealized = np.zeros_like(data) + np.nan
			result.ran = np.nonzero(keep)[0].tolist()
			result.idealize = lambda : self.idealize_fret_hmm(result)
			result.idealize()
			self.recast_rs(result)

			results.append(result)

		elbos = np.array([ri.likelihood[-1,0] for ri in results])
		modelmax = np.argmax(elbos)
		logger.info('vbconsensus hmm - best elbo: %f, nstates=%d'%(elbos[modelmax],results[modelmax].nstates))
		for i in range(len(results)):
			self.models.append(results[i])
			if i == modelmax:
				self._active_model_index = len(self.models)-1
				self.make_report(results[i])
		self.maven.emit_data_update()

	def run_fret_threshold_vbconhmm(self,nstates, threshold):
		success,keep,y = self.get_fret_traces()
		if not success:
			return

		maxiters = self.maven.prefs['modeler.maxiters']
		converge = self.maven.prefs['modeler.converge']
		nrestarts = self.maven.prefs['modeler.nrestarts']
		ncpu = self.maven.prefs['ncpu']

		mu_prior = np.percentile(np.concatenate(y),np.linspace(0,100,nstates+2))[1:-1]
		beta_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.vbconhmm.prior.beta']
		a_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.vbconhmm.prior.a']
		b_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.vbconhmm.prior.b']
		pi_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.vbconhmm.prior.pi']
		tm_prior = np.ones((nstates,nstates))*self.maven.prefs['modeler.vbconhmm.prior.alpha']

		priors = [mu_prior, beta_prior, a_prior, b_prior, pi_prior, tm_prior]

		result = self.cached_vbconhmm(y,priors,nstates,maxiters,converge,nrestarts,ncpu)

		result.ran = np.nonzero(keep)[0].tolist()
		result.idealize = lambda : self.idealize_fret_hmm(result)
		result.idealize()
		self.recast_rs(result)

		con_result = result
		idealized = result.idealized.copy()
		vits = np.concatenate(idealized)
		vits = vits[np.isfinite(vits)]
		result = self.cached_threshold(vits,threshold)
		result.consensusmodel = con_result
		result.type = "threshold + vbConsensus"
		result.ran = np.nonzero(keep)[0].tolist()
		result.idealize = lambda : self.idealize_fret_threshold_viterbi(result,idealized)
		result.idealize()
		self.model = result
		self.make_report(result)
		self.maven.emit_data_update()


	def run_fret_vbhmm(self,nstates):
		success,keep,y = self.get_fret_traces()
		if not success:
			return

		maxiters = self.maven.prefs['modeler.maxiters']
		converge = self.maven.prefs['modeler.converge']
		nrestarts = self.maven.prefs['modeler.nrestarts']
		ncpu = self.maven.prefs['ncpu']

		mu_prior = np.percentile(np.concatenate(y),np.linspace(0,100,nstates+2))[1:-1]
		beta_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.vbhmm.prior.beta']
		a_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.vbhmm.prior.a']
		b_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.vbhmm.prior.b']
		pi_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.vbhmm.prior.pi']
		tm_prior = np.ones((nstates,nstates))*self.maven.prefs['modeler.vbhmm.prior.alpha']

		priors = [mu_prior, beta_prior, a_prior, b_prior, pi_prior, tm_prior]

		from .fxns.hmm import viterbi
		from .model_container import trace_model_container

		mu = np.ones(nstates)*np.nan
		var = mu.copy()
		frac = mu.copy()
		result = self.model_container(type='vb HMM',nstates=nstates,mean=mu,var =var,frac=frac)

		data = self.maven.calc_fret()[:,:,1]
		result.idealized = np.zeros_like(data) + np.nan
		trace_level = {}
		result.ran = np.nonzero(keep)[0].tolist()

		for i in range(len(y)):
			yi = y[i].astype('double')
			r = self.cached_vbhmm(yi,priors,nstates,maxiters,converge,nrestarts,ncpu)
			ii = result.ran[i]
			pre = self.maven.data.pre_list[ii]
			post = self.maven.data.post_list[ii]
			result.idealized[ii,pre:post] = r.mean[viterbi(yi,r.mean,r.var,r.tmatrix,r.frac).astype('int')]
			trace_level_inst = trace_model_container(r, ii)
			trace_level_inst.idealized = result.idealized[ii]
			trace_level[str(ii)] = trace_level_inst

		result.trace_level = trace_level
		self.model = result
		self.make_report(result)
		self.maven.emit_data_update()

	def run_fret_vbhmm_modelselection(self,nstates_min,nstates_max):
		if nstates_min > nstates_max:
			logger.info('nstates min > max')
			return

		success,keep,y = self.get_fret_traces()
		if not success:
			return

		maxiters = self.maven.prefs['modeler.maxiters']
		converge = self.maven.prefs['modeler.converge']
		nrestarts = self.maven.prefs['modeler.nrestarts']
		ncpu = self.maven.prefs['ncpu']


		from .fxns.hmm import viterbi
		from .model_container import trace_model_container

		mu = np.ones(nstates_max)*np.nan
		var = mu.copy()
		frac = mu.copy()
		result = self.model_container(type='vb HMM_model selection',nstates=nstates_max,mean=mu,var =var,frac=frac)

		data = self.maven.calc_fret()[:,:,1]
		result.idealized = np.zeros_like(data) + np.nan
		trace_level = {}
		result.ran = np.nonzero(keep)[0].tolist()
		y_flat = np.concatenate(y)

		for i in range(len(y)):
			results = []
			yi = y[i].astype('double')
			for nstates in range(nstates_min,nstates_max+1):
				mu_prior = np.percentile(y_flat,np.linspace(0,100,nstates+2))[1:-1]
				beta_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.vbhmm.prior.beta']
				a_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.vbhmm.prior.a']
				b_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.vbhmm.prior.b']
				pi_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.vbhmm.prior.pi']
				tm_prior = np.ones((nstates,nstates))*self.maven.prefs['modeler.vbhmm.prior.alpha']

				priors = [mu_prior, beta_prior, a_prior, b_prior, pi_prior, tm_prior]

				results.append(self.cached_vbhmm(yi,priors,nstates,maxiters,converge,nrestarts,ncpu))

			elbos = np.array([ri.likelihood[-1,0] for ri in results])
			modelmax = np.argmax(elbos)
			r = results[modelmax]
			ii = result.ran[i]
			pre = self.maven.data.pre_list[ii]
			post = self.maven.data.post_list[ii]
			result.idealized[ii,pre:post] = r.mean[viterbi(yi,r.mean,r.var,r.tmatrix,r.frac).astype('int')]
			trace_level_inst = trace_model_container(r, ii)
			trace_level_inst.idealized = result.idealized[ii]
			trace_level[str(ii)] = trace_level_inst

		result.trace_level = trace_level
		self.model = result
		self.make_report(result)
		self.maven.emit_data_update()

	def run_fret_kmeans_vbhmm(self,nstates):
		success, keep, y = self.get_fret_traces()
		if not success:
			return

		maxiters = self.maven.prefs['modeler.maxiters']
		converge = self.maven.prefs['modeler.converge']
		nrestarts = self.maven.prefs['modeler.nrestarts']
		ncpu = self.maven.prefs['ncpu']

		from .fxns.hmm import viterbi
		from .model_container import trace_model_container

		data = self.maven.calc_fret()[:,:,1]
		idealized = np.zeros_like(data) + np.nan
		ran = np.nonzero(keep)[0].tolist()

		trace_level = {}

		y_flat = np.concatenate(y)

		for i in range(len(y)):
			yi = y[i].astype('double')
			results = []
			for k in range(1,nstates+1):
				mu_prior = np.percentile(y_flat,np.linspace(0,100,k+2))[1:-1]
				beta_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.vbhmm.prior.beta']
				a_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.vbhmm.prior.a']
				b_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.vbhmm.prior.b']
				pi_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.vbhmm.prior.pi']
				tm_prior = np.ones((k,k))*self.maven.prefs['modeler.vbhmm.prior.alpha']

				priors = [mu_prior, beta_prior, a_prior, b_prior, pi_prior, tm_prior]
				results.append(self.cached_vbhmm(yi,priors,k,maxiters,converge,nrestarts,ncpu))

			elbos = np.array([ri.likelihood[-1,0] for ri in results])
			modelmax = np.argmax(elbos)
			r = results[modelmax]
			ii = ran[i]
			pre = self.maven.data.pre_list[ii]
			post = self.maven.data.post_list[ii]

			vit = r.mean[viterbi(yi,r.mean,r.var,r.norm_tmatrix,r.frac).astype('int')]
			idealized[ii,pre:post] = vit
			trace_level_inst = trace_model_container(r, ii)
			trace_level_inst.idealized = idealized[ii]
			trace_level[str(ii)] = trace_level_inst

		vits = np.concatenate(idealized)
		vits = vits[np.isfinite(vits)]
		result = self.cached_kmeans(vits,nstates)
		result.trace_level = trace_level
		result.type = "kmeans + vb HMM"
		result.ran = ran

		result.rate_type = "Transition Matrix"
		from .fxns.hmm import compose_tmatrix, normalize_tmatrix,convert_tmatrix 
		tmatrix = compose_tmatrix(y,result)
		result.tmatrix = tmatrix
		result.norm_tmatrix = normalize_tmatrix(tmatrix)
		result.rates = convert_tmatrix(tmatrix)
		
		result.idealize = lambda : self.idealize_fret_kmeans_viterbi(result,idealized)
		result.idealize()
		self.recast_rs(result)
		self.model = result
		self.make_report(result)
		self.maven.emit_data_update()

	def run_fret_vbgmm_vbhmm(self,nstates):
		success, keep, y = self.get_fret_traces()
		if not success:
			return

		maxiters = self.maven.prefs['modeler.maxiters']
		converge = self.maven.prefs['modeler.converge']
		nrestarts = self.maven.prefs['modeler.nrestarts']
		ncpu = self.maven.prefs['ncpu']

		from .fxns.hmm import viterbi
		from .model_container import trace_model_container

		data = self.maven.calc_fret()[:,:,1]
		idealized = np.zeros_like(data) + np.nan
		ran = np.nonzero(keep)[0].tolist()

		trace_level = {}
		y_flat = np.concatenate(y)

		for i in range(len(y)):
			yi = y[i].astype('double')
			results = []
			for k in range(1,nstates+1):
				mu_prior = np.percentile(y_flat,np.linspace(0,100,k+2))[1:-1]
				beta_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.vbhmm.prior.beta']
				a_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.vbhmm.prior.a']
				b_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.vbhmm.prior.b']
				pi_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.vbhmm.prior.pi']
				tm_prior = np.ones((k,k))*self.maven.prefs['modeler.vbhmm.prior.alpha']

				priors = [mu_prior, beta_prior, a_prior, b_prior, pi_prior, tm_prior]
				results.append(self.cached_vbhmm(yi,priors,k,maxiters,converge,nrestarts,ncpu))

			elbos = np.array([ri.likelihood[-1,0] for ri in results])
			modelmax = np.argmax(elbos)
			r = results[modelmax]
			ii = ran[i]
			pre = self.maven.data.pre_list[ii]
			post = self.maven.data.post_list[ii]

			vit = r.mean[viterbi(yi,r.mean,r.var,r.tmatrix,r.frac).astype('int')]
			idealized[ii,pre:post] = vit
			trace_level_inst = trace_model_container(r, ii)
			trace_level_inst.idealized = idealized[ii]
			trace_level[str(ii)] = trace_level_inst

		vits = np.concatenate(idealized)
		vits = vits[np.isfinite(vits)]
		priors = np.array([self.maven.prefs[sss] for sss in ['modeler.vbgmm.prior.beta','modeler.vbgmm.prior.a','modeler.vbgmm.prior.b','modeler.vbgmm.prior.pi']])

		result = self.cached_vbgmm(vits,priors,nstates,maxiters,converge,nrestarts,ncpu)
		result.trace_level = trace_level
		result.type = "vb GMM + vb HMM"
		result.ran = ran

		result.rate_type = "Transition Matrix"
		from .fxns.hmm import compose_tmatrix, normalize_tmatrix,convert_tmatrix 
		tmatrix = compose_tmatrix(y,result)
		result.tmatrix = tmatrix
		result.norm_tmatrix = normalize_tmatrix(tmatrix)
		result.rates = convert_tmatrix(tmatrix)

		result.idealize = lambda : self.idealize_fret_gmm_viterbi(result,idealized)
		result.idealize()

		viterbi_var = result.var
		#var = (result.r*((y_flat[:,None] - result.mean[None,:])**2)).sum(0)/(result.r).sum()
		var = np.zeros_like(result.mean)
		for i,state in enumerate(np.argmax(result.r, axis = 1)):
			var[state] += (y_flat[i] - result.mean[state])**2

		var /= result.r.sum()

		# print(viterbi_var,var)
		result.var = var
		result.viterbi_var = viterbi_var
		self.recast_rs(result)

		self.model = result
		self.make_report(result)
		self.maven.emit_data_update()

	def run_fret_vbgmm_vbhmm_modelselection(self,nstates_min,nstates_max):
		success, keep, y = self.get_fret_traces()
		if not success:
			return

		maxiters = self.maven.prefs['modeler.maxiters']
		converge = self.maven.prefs['modeler.converge']
		nrestarts = self.maven.prefs['modeler.nrestarts']
		ncpu = self.maven.prefs['ncpu']

		from .fxns.hmm import viterbi
		from .model_container import trace_model_container

		data = self.maven.calc_fret()[:,:,1]
		idealized = np.zeros_like(data) + np.nan
		ran = np.nonzero(keep)[0].tolist()
		results_ens = []
		y_flat = np.concatenate(y)


		for nstates in range(nstates_min,nstates_max+1):
			trace_level = {}

			for i in range(len(y)):
				yi = y[i].astype('double')
				results = []
				for k in range(1,nstates+1):
					mu_prior = np.percentile(y_flat,np.linspace(0,100,k+2))[1:-1]
					beta_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.vbhmm.prior.beta']
					a_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.vbhmm.prior.a']
					b_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.vbhmm.prior.b']
					pi_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.vbhmm.prior.pi']
					tm_prior = np.ones((k,k))*self.maven.prefs['modeler.vbhmm.prior.alpha']

					priors = [mu_prior, beta_prior, a_prior, b_prior, pi_prior, tm_prior]
					results.append(self.cached_vbhmm(yi,priors,k,maxiters,converge,nrestarts,ncpu))

				elbos = np.array([ri.likelihood[-1,0] for ri in results])
				modelmax = np.argmax(elbos)
				r = results[modelmax]
				ii = ran[i]
				pre = self.maven.data.pre_list[ii]
				post = self.maven.data.post_list[ii]

				vit = r.mean[viterbi(yi,r.mean,r.var,r.tmatrix,r.frac).astype('int')]
				idealized[ii,pre:post] = vit
				trace_level_inst = trace_model_container(r, ii)
				trace_level_inst.idealized = idealized[ii]
				trace_level[str(ii)] = trace_level_inst

			vits = np.concatenate(idealized)
			vits = vits[np.isfinite(vits)]
			priors = np.array([self.maven.prefs[sss] for sss in ['modeler.vbgmm.prior.beta','modeler.vbgmm.prior.a','modeler.vbgmm.prior.b','modeler.vbgmm.prior.pi']])

			result = self.cached_vbgmm(vits,priors,nstates,maxiters,converge,nrestarts,ncpu)
			result.trace_level = trace_level
			result.type = "vb GMM + vb HMM"
			result.ran = ran

			result.rate_type = "Transition Matrix"
			from .fxns.hmm import compose_tmatrix, normalize_tmatrix,convert_tmatrix 
			tmatrix = compose_tmatrix(y,result)
			result.tmatrix = tmatrix
			result.norm_tmatrix = normalize_tmatrix(tmatrix)
			result.rates = convert_tmatrix(tmatrix)

			result.idealize = lambda : self.idealize_fret_gmm_viterbi(result,idealized)
			result.idealize()
			self.recast_rs(result)
			results_ens.append(result)

		elbos = np.array([ri.likelihood[-1,0] for ri in results_ens])
		modelmax = np.argmax(elbos)
		logger.info('VB HMM->VB GMM - best elbo: %f, nstates=%d'%(elbos[modelmax],results_ens[modelmax].nstates))
		for i in range(len(results_ens)):
			self.models.append(results_ens[i])
			if i == modelmax:
				self._active_model_index = len(self.models)-1
				self.make_report(results_ens[i])
		self.maven.emit_data_update()


	def run_fret_threshold_vbhmm(self,nstates,threshold):
		success, keep, y = self.get_fret_traces()
		if not success:
			return

		maxiters = self.maven.prefs['modeler.maxiters']
		converge = self.maven.prefs['modeler.converge']
		nrestarts = self.maven.prefs['modeler.nrestarts']
		ncpu = self.maven.prefs['ncpu']

		from .fxns.hmm import viterbi
		from .model_container import trace_model_container

		data = self.maven.calc_fret()[:,:,1]
		idealized = np.zeros_like(data) + np.nan
		ran = np.nonzero(keep)[0].tolist()

		trace_level = {}

		y_flat = np.concatenate(y)

		for i in range(len(y)):
			yi = y[i].astype('double')
			results = []
			for k in range(1,nstates+1):
				mu_prior = np.percentile(y_flat,np.linspace(0,100,k+2))[1:-1]
				beta_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.vbhmm.prior.beta']
				a_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.vbhmm.prior.a']
				b_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.vbhmm.prior.b']
				pi_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.vbhmm.prior.pi']
				tm_prior = np.ones((k,k))*self.maven.prefs['modeler.vbhmm.prior.alpha']

				priors = [mu_prior, beta_prior, a_prior, b_prior, pi_prior, tm_prior]
				results.append(self.cached_vbhmm(yi,priors,k,maxiters,converge,nrestarts,ncpu))

			elbos = np.array([ri.likelihood[-1,0] for ri in results])
			modelmax = np.argmax(elbos)
			r = results[modelmax]
			ii = ran[i]
			pre = self.maven.data.pre_list[ii]
			post = self.maven.data.post_list[ii]

			vit = r.mean[viterbi(yi,r.mean,r.var,r.tmatrix,r.frac).astype('int')]
			idealized[ii,pre:post] = vit
			trace_level_inst = trace_model_container(r, ii)
			trace_level_inst.idealized = idealized[ii]
			trace_level[str(ii)] = trace_level_inst

		vits = np.concatenate(idealized)
		vits = vits[np.isfinite(vits)]
		result = self.cached_threshold(vits,threshold)
		result.trace_level = trace_level
		result.type = "threshold + vb HMM"
		result.ran = ran
		result.idealize = lambda : self.idealize_fret_threshold_viterbi(result,idealized)
		result.idealize()
		self.model = result
		self.make_report(result)
		self.maven.emit_data_update()


	def run_fret_ebhmm(self,nstates):
		success,keep,y = self.get_fret_traces()
		if not success:
			return

		maxiters = self.maven.prefs['modeler.maxiters']
		converge = self.maven.prefs['modeler.converge']
		nrestarts = self.maven.prefs['modeler.nrestarts']
		ncpu = self.maven.prefs['ncpu']

		mu_prior = np.percentile(np.concatenate(y),np.linspace(0,100,nstates+2))[1:-1]
		beta_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.ebhmm.prior.beta']
		a_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.ebhmm.prior.a']
		b_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.ebhmm.prior.b']
		pi_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.ebhmm.prior.pi']
		tm_prior = np.ones((nstates,nstates))*self.maven.prefs['modeler.ebhmm.prior.alpha']

		priors = [mu_prior, beta_prior, a_prior, b_prior, pi_prior, tm_prior]

		result, vbs = self.cached_ebhmm(y,priors,nstates,maxiters,converge,nrestarts,ncpu)
		result.ran = np.nonzero(keep)[0].tolist()

		data = self.maven.calc_fret()[:,:,1]
		idealized = np.zeros_like(data) + np.nan
		chain = np.zeros_like(idealized).astype('int')

		trace_level = {}
		rs = []

		from .fxns.hmm import viterbi
		from .model_container import trace_model_container

		for i in range(len(y)):
			yi = y[i].astype('double')
			r = vbs[i]
			rs.append(r.r)
			ii = result.ran[i]
			pre = self.maven.data.pre_list[ii]
			post = self.maven.data.post_list[ii]

			idealpath = viterbi(yi,r.mean,r.var,r.tmatrix,r.frac).astype('int')
			vit = r.mean[idealpath]
			idealized[ii,pre:post] = vit
			chain[ii,pre:post] = idealpath.copy()
			trace_level_inst = trace_model_container(r, ii)
			trace_level_inst.idealized = idealized[ii]
			trace_level_inst.chain = chain[ii]
			trace_level[str(ii)] = trace_level_inst

		result.trace_level = trace_level
		result.r = rs
		self.recast_rs(result)
		result.idealized = idealized
		result.chain = chain
		self.model = result
		self.make_report(result)
		self.maven.emit_data_update()

	def run_fret_ebhmm_modelselection(self,nstates_min,nstates_max):
		if nstates_min > nstates_max:
			logger.info('nstates min > max')
			return

		success,keep,y = self.get_fret_traces()
		if not success:
			return

		maxiters = self.maven.prefs['modeler.maxiters']
		converge = self.maven.prefs['modeler.converge']
		nrestarts = self.maven.prefs['modeler.nrestarts']
		ncpu = self.maven.prefs['ncpu']

		data = self.maven.calc_fret()[:,:,1]
		results = []

		from .fxns.hmm import viterbi
		from .model_container import trace_model_container

		for nstates in range(nstates_min,nstates_max+1):
			mu_prior = np.percentile(np.concatenate(y),np.linspace(0,100,nstates+2))[1:-1]
			beta_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.ebhmm.prior.beta']
			a_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.ebhmm.prior.a']
			b_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.ebhmm.prior.b']
			pi_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.ebhmm.prior.pi']
			tm_prior = np.ones((nstates,nstates))*self.maven.prefs['modeler.ebhmm.prior.alpha']

			priors = [mu_prior, beta_prior, a_prior, b_prior, pi_prior, tm_prior]

			result, vbs = self.cached_ebhmm(y,priors,nstates,maxiters,converge,nrestarts,ncpu)
			result.ran = np.nonzero(keep)[0].tolist()

			data = self.maven.calc_fret()[:,:,1]
			idealized = np.zeros_like(data) + np.nan
			chain = np.zeros_like(idealized).astype('int')

			trace_level = {}
			rs = []


			for i in range(len(y)):
				yi = y[i].astype('double')
				r = vbs[i]
				rs.append(r.r)
				ii = result.ran[i]
				pre = self.maven.data.pre_list[ii]
				post = self.maven.data.post_list[ii]

				idealpath = viterbi(yi,r.mean,r.var,r.tmatrix,r.frac).astype('int')
				vit = r.mean[idealpath]
				idealized[ii,pre:post] = vit
				chain[ii,pre:post] = idealpath.copy()
				trace_level_inst = trace_model_container(r, ii)
				trace_level_inst.idealized = idealized[ii]
				trace_level_inst.chain = chain[ii]
				trace_level[str(ii)] = trace_level_inst

			result.trace_level = trace_level
			result.idealized = idealized
			result.chain = chain
			result.r = rs
			self.recast_rs(result)
			results.append(result)

		elbos = np.array([ri.likelihood[-1] for ri in results])
		modelmax = np.argmax(elbos)
		logger.info('EB HMM - best elbo: %f, nstates=%d'%(elbos[modelmax],results[modelmax].nstates))
		for i in range(len(results)):
			self.models.append(results[i])
			if i == modelmax:
				self._active_model_index = len(self.models)-1
				self.make_report(results[i])
		self.maven.emit_data_update()

	def run_fret_hhmm(self):
		success,keep,y = self.get_fret_traces()
		if not success:
			return
	
		max_iter = self.maven.prefs['modeler.hhmm.maxiters']
		tol = self.maven.prefs['modeler.hhmm.tolerance']
		restarts = self.maven.prefs['modeler.hhmm.restarts']
		guess = np.array([0., 1.])
		depth_vec = [2]
		prod_states = 2


		post_mu, post_beta, post_a, post_b, post_pi, post_tm, post_exit,likelihood = self.cached_vmphhmm(y,
																							depth_vec,prod_states,
																							max_iter,tol,
																							restarts,guess)

		#hardcoded to 4 state
		flat_tm = np.zeros((4,4),dtype=np.float64)
		#result = [post_mu, post_beta, post_a, post_b, post_pi, post_tm, post_exit]

		new_tm = []
		for level in post_tm:
			new_level = []
			for tm in level:
				newt = tm.T.copy()
				for i in range(newt.shape[0]):
					newt[i] /= newt[i].sum()
				new_level.append(newt)
			new_tm.append(new_level)

		flat_tm[:2,:2] = new_tm[1][0]
		flat_tm[-2:,-2:] = new_tm[1][1]
		flat_tm[:2,-2:] = post_exit[1][0].reshape(2,1)@post_pi[1][1].reshape(1,2)*new_tm[0][0][0,1]
		flat_tm[-2:,:2] = post_exit[1][1].reshape(2,1)@post_pi[1][0].reshape(1,2)*new_tm[0][0][1,0]

		#print(flat_tm)
		mu = np.zeros(4)
		mu[:2] = post_mu.flatten()
		mu[-2:] = post_mu.flatten()
		var = np.zeros(4)
		var[:2] = (post_b/post_a).flatten()
		var[-2:] = (post_b/post_a).flatten()

		upper_pi = (post_exit[0][0]*post_pi[0][0]).flatten()
		pi =  np.zeros(4)
		pi[:2] = upper_pi*post_pi[1][0]*post_exit[1][1].flatten()
		pi[-2:] = upper_pi*post_pi[1][1]*post_exit[1][0].flatten()
		pi /= pi.sum()

		for i in range(flat_tm.shape[0]):
			flat_tm[i] /= flat_tm[i].sum()

		from .model_container import model_container

		result = model_container(type='hierarchical HMM',
								nstates = 4,mean=mu,var=var,frac=pi,
								tmatrix=flat_tm,
								likelihood=likelihood,
								a=post_a,b=post_b,beta=post_beta, h_pi=post_pi, h_tm=post_tm, h_exit=post_exit)

		return result

	def run_fret_vbhmm_one(self,nstates):
		success,keep,y = self.get_fret_traces()
		if not success:
			return
		if keep.sum() != 1:
			## WARNING:"Error: Run all, Apply all","You have more than one molecule turned on"
			return
		maxiters = self.maven.prefs['modeler.maxiters']
		converge = self.maven.prefs['modeler.converge']
		nrestarts = self.maven.prefs['modeler.nrestarts']
		ncpu = self.maven.prefs['ncpu']

		mu_prior = np.percentile(np.concatenate(y),np.linspace(0,100,nstates+2))[1:-1]
		beta_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.vbhmm.prior.beta']
		a_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.vbhmm.prior.a']
		b_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.vbhmm.prior.b']
		pi_prior = np.ones_like(mu_prior)*self.maven.prefs['modeler.vbhmm.prior.pi']
		tm_prior = np.ones((nstates,nstates))*self.maven.prefs['modeler.vbhmm.prior.alpha']

		priors = [mu_prior, beta_prior, a_prior, b_prior, pi_prior, tm_prior]

		result = self.cached_vbhmm(y[0].astype('double'),priors,nstates,maxiters,converge,nrestarts,ncpu)

		result.ran = [np.nonzero(keep)[0]]
		result.idealize = lambda : self.idealize_fret_hmm(result)
		result.idealize()
		self.model = result
		self.make_report(result)
		self.maven.emit_data_update()

	def run_fret_dwell_analysis(self, type, state, fix_A = False):
		tau,surv = self.get_survival_dwells(state)

		if not type in self.cached_functions:
			self.cache_exponential_fxn(type)

		rates = self.cached_functions[type](tau,surv,fix_A)

		model = self.model

		if model.rate_type in ["Transition Matrix", "N/A"]:
			model.rate_type = "Dwell Analysis"
			model.rates = {}

		model.rates[state] = {}
		model.rates[state]['ks'] = rates[0]
		model.rates[state]['As'] = rates[-1]

		if type == "Stretched Exponential":
			model.rates[state]["betas"] = rates[1]
		elif "betas" in model.rates:
			model.rates.pop('betas')

		self.model = model

	def run_fret_tmatrix(self):
		model = self.model

		if not model.tmatrix is None:
			from .fxns.hmm import convert_tmatrix
			model.rates = convert_tmatrix(model.tmatrix)
			model.rate_type = "Transition Matrix"
		else:
			return


	def idealize_fret_gmm(self,result):
		data = self.maven.calc_fret()[:,:,1]
		result.idealized = np.zeros_like(data) + np.nan
		result.chain = np.zeros_like(result.idealized).astype('int')

		for ii in result.ran:
			pre = self.maven.data.pre_list[ii]
			post = self.maven.data.post_list[ii]
			prob = 1./np.sqrt(2.*np.pi*result.var[None,None,:])*np.exp(-.5/result.var[None,None,:]*(data[ii,pre:post,None]-result.mean[None,None,:])**2.)
			prob /= prob.sum(2)[:,:,None]
			prob *= result.frac[None,None,:]
			idealpath = np.argmax(prob,axis=2).astype('int64')
			result.chain[ii, pre:post] = idealpath.copy()
			result.idealized[ii, pre:post] = result.mean[idealpath]

	def idealize_fret_kmeans(self,result):
		data = self.maven.calc_fret()[:,:,1]
		result.idealized = np.zeros_like(data) + np.nan
		result.chain = np.zeros_like(result.idealized).astype('int')

		for ii in result.ran:
			#ii = result.ran[i]
			pre = self.maven.data.pre_list[ii]
			post = self.maven.data.post_list[ii]
			idealpath = np.abs(data[ii,pre:post,None] - result.mean[None,:]).argmin(1).astype('int64')
			result.chain[ii, pre:post] = idealpath.copy()
			result.idealized[ii, pre:post] = result.mean[idealpath]

	def idealize_fret_threshold(self,result):
		data = self.maven.calc_fret()[:,:,1]
		threshold = result.threshold
		result.idealized = np.zeros_like(data) + np.nan
		result.chain = np.zeros_like(result.idealized).astype('int')
		for ii in result.ran:
			pre = self.maven.data.pre_list[ii]
			post = self.maven.data.post_list[ii]
			idealpath = (data[ii,pre:post] > threshold).astype('int')
			result.chain[ii,pre:post] = idealpath.copy()
			result.idealized[ii,pre:post] = result.mean[idealpath]

	def idealize_fret_kmeans_viterbi(self,result,vit):
		result.idealized = np.zeros_like(vit) + np.nan
		result.chain = np.zeros_like(result.idealized).astype('int')
		for ii in result.ran:
			pre = self.maven.data.pre_list[ii]
			post = self.maven.data.post_list[ii]
			idealpath = np.abs(vit[ii,pre:post,None] - result.mean[None,:]).argmin(1).astype('int64')
			result.chain[ii,pre:post] = idealpath.copy()
			result.idealized[ii, pre:post] = result.mean[idealpath]

	def idealize_fret_gmm_viterbi(self,result, vit):
		result.idealized = np.zeros_like(vit) + np.nan
		result.chain = np.zeros_like(result.idealized).astype('int')

		for ii in result.ran:
			pre = self.maven.data.pre_list[ii]
			post = self.maven.data.post_list[ii]
			prob = 1./np.sqrt(2.*np.pi*result.var[None,None,:])*np.exp(-.5/result.var[None,None,:]*(vit[ii,pre:post,None]-result.mean[None,None,:])**2.)
			prob /= prob.sum(2)[:,:,None]
			prob *= result.frac[None,None,:]
			idealpath = np.argmax(prob,axis=2).astype('int64')
			result.chain[ii,pre:post] = idealpath.copy()
			result.idealized[ii, pre:post] = result.mean[idealpath]

	def idealize_fret_threshold_viterbi(self,result,vit):
		result.idealized = np.zeros_like(vit) + np.nan
		result.chain = np.zeros_like(result.idealized).astype('int')
		threshold = result.threshold

		for ii in result.ran:
			pre = self.maven.data.pre_list[ii]
			post = self.maven.data.post_list[ii]
			idealpath = (vit[ii,pre:post] > threshold).astype('int')
			result.chain[ii,pre:post] = idealpath.copy()
			result.idealized[ii,pre:post] = result.mean[idealpath]

	def idealize_fret_hmm(self,result):
		from .fxns.hmm import viterbi
		data = self.maven.calc_fret()[:,:,1].astype('double')
		result.idealized = np.zeros_like(data) + np.nan
		result.chain = np.zeros_like(result.idealized).astype('int')
		pre = self.maven.data.pre_list
		post = self.maven.data.post_list
		for i in range(data.shape[0]):
			if post[i]-pre[i]>=2:
				result.chain[i,pre[i]:post[i]] = viterbi(data[i,pre[i]:post[i]],result.mean,result.var,result.norm_tmatrix,result.frac).astype('int')
				result.idealized[i,pre[i]:post[i]] = result.mean[result.chain[i,pre[i]:post[i]]]
