import numpy as np
import logging
logger = logging.getLogger(__name__)

default_prefs = {
	'modeler.nrestarts':4,
	'modeler.converge':1e-10,
	'modeler.maxiters':1000,
	'modeler.vbhmm.prior.beta':0.25,
	'modeler.vbhmm.prior.a':2.5,
	'modeler.vbhmm.prior.b':0.01,
	'modeler.vbhmm.prior.alpha':1.,
	'modeler.vbhmm.prior.pi':1,
	'modeler.vbconhmm.prior.beta':0.25,
	'modeler.vbconhmm.prior.a':2.5,
	'modeler.vbconhmm.prior.b':0.01,
	'modeler.vbconhmm.prior.alpha':1.,
	'modeler.vbconhmm.prior.pi':1.
}

class controller_modeler(object):
	''' Handles modeling data

	* several .get_* functions to help setup models
	* several model specific io functions (save,load)
	* direct modeling functions (should basically not know about a GUI/maven. input is all relevant variables etc.). Individual modes will get the relevant data together and then call these functions which will dump the result into .result


	### TODO:
		cache compile in a separate thread fxns that need to be numba'd?
		remove ??? * .result - is None or holds a model result
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

	def export_result(self,result,fn):
		import pickle
		with open(fn, 'wb') as f:
			pickle.dump(result,f)
		logger.info('saved result in {}'.format(fn))

	def load_result(self,fn):
		import pickle
		with open(fn, 'rb') as f:
			result = pickle.load(f)
		logger.info('loaded result from {}'.format(fn))
		return result

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

	def clip_traces(self,y,low=-1,high=2):
		for i in range(len(y)):
			yy = y[i]
			## Clip traces and redistribute randomly
			bad = np.bitwise_or((yy < -1.),np.bitwise_or((yy > 2),np.isnan(yy)))
			yy[bad] = np.random.uniform(low=low,high=high,size=int(bad.sum()))
			y[i] = yy
		return y

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
		result = self.cached_functions['ML GMM'](y,nstates+1,maxiters=maxiters,threshold=converge,nrestarts=nrestarts,ncpu=ncpu)
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

		result = self.cached_functions['VB CON HMM'](y,nstates,maxiters=maxiters,threshold=converge,nrestarts=nrestarts,prior_strengths=priors,ncpu=ncpu)
		return result

	def cached_vbhmm(self,y,priors,nstates,maxiters,converge,nrestarts,ncpu):
		''' Run individual VB HMM
		priors is a 1D np.ndarray of eg [beta,a,b,pi,alpha] used for all states
		'''
		if not 'VB HMM' in self.cached_functions:
			logger.info('Caching VB HMM...')
			from .hmm_vb import vb_em_hmm,vb_em_hmm_parallel
			self.cached_functions['VB HMM'] = vb_em_hmm_parallel

		result = self.cached_functions['VB HMM'](y,nstates,maxiters=maxiters,threshold=converge,nrestarts=nrestarts,prior_strengths=priors,ncpu=ncpu)
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
		result = model_container('threshold')
		result.threshold = threshold
		yclass = (y > threshold).astype('int')
		result.mu = np.array([y[yclass==j].mean() for j in [0,1]]).astype('double')
		result.var = np.array([y[yclass==j].var() for j in [0,1]]).astype('double')
		result.ppi = np.array([(yclass==j).sum() for j in [0,1]]).astype('double')
		result.ppi /= result.ppi.sum()
		return result

	def cached_kmeans(self,y,nstates):
		''' Cluster a 1D np.ndarray using K-means. Will flatten if not 1d'''
		if not 'kmeans' in self.cached_functions:
			logger.info('Caching K-means...')
			from .kmeans import kmeans
			self.cached_functions['kmeans'] = kmeans

		result = self.cached_functions['kmeans'](y,nstates)
		return result

	def run_fret_kmeans(self,nstates):
		success, keep, y = self.get_fret_traces()
		if not success:
			return

		result = self.cached_kmeans(np.concatenate(y),nstates)
		result.ran = np.nonzero(keep)[0].tolist()
		result.idealize = lambda : self.idealize_fret_kmeans(result)
		result.idealize()
		self.model = result
		self.maven.emit_data_update()

	def run_fret_vbgmm(self,nstates):
		success, keep, y = self.get_fret_traces()
		if not success:
			return

		maxiters = self.maven.prefs['modeler.maxiters']
		converge = self.maven.prefs['modeler.converge']
		nrestarts = self.maven.prefs['modeler.nrestarts']
		ncpu = self.maven.prefs['ncpu']
		priors = np.array([self.maven.prefs[sss] for sss in ['modeler.vbhmm.prior.beta','modeler.vbhmm.prior.a','modeler.vbhmm.prior.b','modeler.vbhmm.prior.pi']])

		result = self.cached_vbgmm(np.concatenate(y),priors,nstates,maxiters,converge,nrestarts,ncpu)
		result.ran = np.nonzero(keep)[0].tolist()
		result.idealize = lambda : self.idealize_fret_gmm(result)
		result.idealize()
		self.model = result
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
		self.model = result
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
		result = self.model_container('ml HMM')
		data = self.maven.calc_fret()[:,:,1]
		result.idealized = np.zeros_like(data) + np.nan
		result.hmms = []
		result.ran = np.nonzero(keep)[0].tolist()

		for i in range(len(y)):
			yi = y[i].astype('double')
			result.hmms.append(self.cached_mlhmm(yi,nstates,maxiters,converge,nrestarts,ncpu))
			ri = result.hmms[-1]
			ii = result.ran[i]
			pre = self.maven.data.pre_list[ii]
			post = self.maven.data.post_list[ii]
			result.idealized[ii,pre:post] = ri.mu[viterbi(yi,ri.mu,ri.var,ri.tmatrix,ri.ppi).astype('int')]
		self.model = result
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
		hmms = []
		data = self.maven.calc_fret()[:,:,1]
		vit = []
		idealized = np.zeros_like(data) + np.nan
		ran = np.nonzero(keep)[0].tolist()

		for i in range(len(y)):
			yi = y[i].astype('double')
			hmms.append(self.cached_mlhmm(yi,nstates,maxiters,converge,nrestarts,ncpu))
			ri = hmms[-1]
			ii = ran[i]
			pre = self.maven.data.pre_list[ii]
			post = self.maven.data.post_list[ii]
			vit.append(ri.mu[viterbi(yi,ri.mu,ri.var,ri.tmatrix,ri.ppi).astype('int')])
			idealized[ii,pre:post] = vit[-1]

		result = self.cached_kmeans(np.concatenate(vit),nstates)
		result.hmms = hmms
		result.type = "kmeans + ml HMM"
		result.ran = ran
		result.idealize = lambda : self.idealize_fret_kmeans_viterbi(result,idealized)
		result.idealize()
		self.model = result
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

		result = self.cached_threshold(np.concatenate(y),threshold)
		result.idealize = lambda : self.idealize_fret_threshold(result)
		result.idealize()
		self.model = result
		self.maven.emit_data_update()

	def run_fret_vbconhmm(self,nstates):
		success,keep,y = self.get_fret_traces()
		if not success:
			logger.info('failed to get traces')
			return

		maxiters = self.maven.prefs['modeler.maxiters']
		converge = self.maven.prefs['modeler.converge']
		nrestarts = self.maven.prefs['modeler.nrestarts']
		ncpu = self.maven.prefs['ncpu']
		priors = np.array([self.maven.prefs[sss] for sss in ['modeler.vbconhmm.prior.beta',
						'modeler.vbconhmm.prior.a','modeler.vbconhmm.prior.b','modeler.vbconhmm.prior.pi',
						'modeler.vbconhmm.prior.alpha']])

		from .fxns.hmm import viterbi
		result = self.cached_vbconhmm(y,priors,nstates,maxiters,converge,nrestarts,ncpu)

		data = self.maven.calc_fret()[:,:,1]
		result.idealized = np.zeros_like(data) + np.nan
		result.ran = np.nonzero(keep)[0].tolist()
		result.idealize = lambda : self.idealize_fret_hmm(result)
		result.idealize()
		self.model = result
		self.maven.emit_data_update()

	def run_fret_vbconhmm_modelselection(self,nstates_min,nstates_max):
		if nstates_min > nstates_max:
			logger.info('nstates min > max')
			return

		success,keep,y = self.get_fret_traces()
		if not success:
			logger.info('failed to get traces')
			return

		maxiters = self.maven.prefs['modeler.maxiters']
		converge = self.maven.prefs['modeler.converge']
		nrestarts = self.maven.prefs['modeler.nrestarts']
		ncpu = self.maven.prefs['ncpu']
		priors = np.array([self.maven.prefs[sss] for sss in ['modeler.vbconhmm.prior.beta',
						'modeler.vbconhmm.prior.a','modeler.vbconhmm.prior.b','modeler.vbconhmm.prior.pi',
						'modeler.vbconhmm.prior.alpha']])

		from .fxns.hmm import viterbi
		data = self.maven.calc_fret()[:,:,1]
		results = []
		for nstates in range(nstates_min,nstates_max+1):
			result = self.cached_vbconhmm(y,priors,nstates,maxiters,converge,nrestarts,ncpu)
			result.idealized = np.zeros_like(data) + np.nan
			result.ran = np.nonzero(keep)[0].tolist()
			result.idealize = lambda : self.idealize_fret_hmm(result)
			result.idealize()
			results.append(result)

		elbos = np.array([ri.elbo[-1,0] for ri in results])
		modelmax = np.argmax(elbos)
		logger.info('vbconsensus hmm - best elbo: %f, nstates=%d'%(elbos[modelmax],results[modelmax].mu.size))
		for i in range(len(results)):
			self.models.append(results[i])
			if i == modelmax:
				self._active_model_index = len(self.models)-1
		self.maven.emit_data_update()

	def run_fret_vbhmm(self,nstates):
		success,keep,y = self.get_fret_traces()
		if not success:
			return

		maxiters = self.maven.prefs['modeler.maxiters']
		converge = self.maven.prefs['modeler.converge']
		nrestarts = self.maven.prefs['modeler.nrestarts']
		ncpu = self.maven.prefs['ncpu']
		priors = np.array([self.maven.prefs[sss] for sss in ['modeler.vbhmm.prior.beta',
						'modeler.vbhmm.prior.a','modeler.vbhmm.prior.b',
						'modeler.vbhmm.prior.pi','modeler.vbhmm.prior.alpha']])

		from .fxns.hmm import viterbi
		result = self.model_container('vb HMM')
		data = self.maven.calc_fret()[:,:,1]
		result.idealized = np.zeros_like(data) + np.nan
		result.hmms = []
		result.ran = np.nonzero(keep)[0].tolist()

		for i in range(len(y)):
			yi = y[i].astype('double')
			result.hmms.append(self.cached_vbhmm(yi,priors,nstates,maxiters,converge,nrestarts,ncpu))
			r = result.hmms[-1]
			ii = result.ran[i]
			pre = self.maven.data.pre_list[ii]
			post = self.maven.data.post_list[ii]
			result.idealized[ii,pre:post] = r.mu[viterbi(yi,r.mu,r.var,r.tmatrix,r.ppi).astype('int')]
		self.model = result
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
		priors = np.array([self.maven.prefs[sss] for sss in ['modeler.vbhmm.prior.beta',
						'modeler.vbhmm.prior.a','modeler.vbhmm.prior.b',
						'modeler.vbhmm.prior.pi','modeler.vbhmm.prior.alpha']])

		from .fxns.hmm import viterbi
		result = self.model_container('vb HMM')
		data = self.maven.calc_fret()[:,:,1]
		result.idealized = np.zeros_like(data) + np.nan
		result.hmms = []
		result.ran = np.nonzero(keep)[0].tolist()

		for i in range(len(y)):
			results = []
			yi = y[i].astype('double')
			for nstates in range(nstates_min,nstates_max+1):
				results.append(self.cached_vbhmm(yi,priors,nstates,maxiters,converge,nrestarts,ncpu))
			elbos = np.array([ri.elbo[-1,0] for ri in results])
			modelmax = np.argmax(elbos)
			r = results[modelmax]
			result.hmms.append(r)
			ii = result.ran[i]
			pre = self.maven.data.pre_list[ii]
			post = self.maven.data.post_list[ii]
			result.idealized[ii,pre:post] = r.mu[viterbi(yi,r.mu,r.var,r.tmatrix,r.ppi).astype('int')]

		self.model = result
		self.maven.emit_data_update()

	def run_fret_kmeans_vbhmm(self,nstates):
		success, keep, y = self.get_fret_traces()
		if not success:
			return

		maxiters = self.maven.prefs['modeler.maxiters']
		converge = self.maven.prefs['modeler.converge']
		nrestarts = self.maven.prefs['modeler.nrestarts']
		ncpu = self.maven.prefs['ncpu']
		priors = np.array([self.maven.prefs[sss] for sss in ['modeler.vbhmm.prior.beta',
						'modeler.vbhmm.prior.a','modeler.vbhmm.prior.b',
						'modeler.vbhmm.prior.pi','modeler.vbhmm.prior.alpha']])

		from .fxns.hmm import viterbi
		hmms = []
		data = self.maven.calc_fret()[:,:,1]
		vit = []
		idealized = np.zeros_like(data) + np.nan
		ran = np.nonzero(keep)[0].tolist()

		for i in range(len(y)):
			yi = y[i].astype('double')
			hmms.append(self.cached_vbhmm(yi,priors,nstates,maxiters,converge,nrestarts,ncpu))
			ri = hmms[-1]
			ii = ran[i]
			pre = self.maven.data.pre_list[ii]
			post = self.maven.data.post_list[ii]
			vit.append(ri.mu[viterbi(yi,ri.mu,ri.var,ri.tmatrix,ri.ppi).astype('int')])
			idealized[ii,pre:post] = vit[-1]

		result = self.cached_kmeans(np.concatenate(vit),nstates)
		result.hmms = hmms
		result.type = "kmeans + vb HMM"
		result.ran = ran
		result.idealize = lambda : self.idealize_fret_kmeans_viterbi(result,idealized)
		result.idealize()
		self.model = result
		self.maven.emit_data_update()

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
		priors = np.array([self.maven.prefs[sss] for sss in ['modeler.vbhmm.prior.beta',
						'modeler.vbhmm.prior.a','modeler.vbhmm.prior.b',
						'modeler.vbhmm.prior.pi','modeler.vbhmm.prior.alpha']])

		result = self.cached_vbhmm(y[0].astype('double'),priors,nstates,maxiters,converge,nrestarts,ncpu)

		result.ran = [np.nonzero(keep)[0]]
		result.idealize = lambda : self.idealize_fret_hmm(result)
		result.idealize()
		self.model = result
		self.maven.emit_data_update()

	def idealize_fret_gmm(self,result):
		data = self.maven.calc_fret()[:,:,1]
		result.idealized = np.zeros_like(data) + np.nan

		prob = 1./np.sqrt(2.*np.pi*result.var[None,None,:])*np.exp(-.5/result.var[None,None,:]*(data[:,:,None]-result.mu[None,None,:])**2.)
		prob /= prob.sum(2)[:,:,None]
		prob *= result.ppi[None,None,:]
		idealpath = np.argmax(prob,axis=2).astype('int64')
		result.idealized = result.mu[idealpath]

	def idealize_fret_kmeans(self,result):
		data = self.maven.calc_fret()[:,:,1]
		result.idealized = np.zeros_like(data) + np.nan
		for ii in result.ran:
			#ii = result.ran[i]
			pre = self.maven.data.pre_list[ii]
			post = self.maven.data.post_list[ii]
			idealpath = np.abs(data[ii,pre:post,None] - result.mu[None,:]).argmin(1).astype('int64')
			result.idealized[ii, pre:post] = result.mu[idealpath]

	def idealize_fret_threshold(self,result):
		data = self.maven.calc_fret()[:,:,1]

		# f2 = result.var[0]/result.var.sum()
		# threshold = result.mu[0] + (result.mu[1]-result.mu[0])*f2
		threshold = result.threshold

		result.idealized = np.zeros_like(data)
		for i in range(data.shape[0]):
			idealpath = (data[i] > threshold).astype('int')
			result.idealized[i] = result.mu[idealpath]

	def idealize_fret_kmeans_viterbi(self,result,vit):
		result.idealized = np.zeros_like(vit) + np.nan
		for ii in result.ran:
			#ii = result.ran[i]
			pre = self.maven.data.pre_list[ii]
			post = self.maven.data.post_list[ii]
			idealpath = np.abs(vit[ii,pre:post,None] - result.mu[None,:]).argmin(1).astype('int64')
			result.idealized[ii, pre:post] = result.mu[idealpath]

	def idealize_fret_hmm(self,result):
		from .fxns.hmm import viterbi
		data = self.maven.calc_fret()[:,:,1].astype('double')
		result.idealized = np.zeros_like(data) + np.nan
		pre = self.maven.data.pre_list
		post = self.maven.data.post_list
		for i in range(data.shape[0]):
			if post[i]-pre[i]>=2:
				result.idealized[i,pre[i]:post[i]] = result.mu[viterbi(data[i,pre[i]:post[i]],result.mu,result.var,result.tmatrix,result.ppi).astype('int')]


class model_container(object):
	'''
	A container object for the details of any (ensemble) model learnt from sm data.
	A model must contain:
		- type (str): Identifies the specific model used
		- ran (list/None): List of indices of molecules the model was run on or None
		- idealized (np.ndarray/None): representation of the model in dataspace -- shape should probably match maven.data.raw.shape

	Optional top level model information (populated depending on 'type'):
		These are all duck-typed. Check to make sure that the model type has one of these.

	.idealize is an overwritten function that should update .idealized all by itself (or just not do anything)
	'''
	def __init__(self, type, ran=[], idealized=None, **kwargs):
		self.type = type
		self.ran = ran
		self.idealized = idealized
		self.__dict__.update(kwargs)

		import time
		self.time_made = time.ctime()

	def description(self):
		mem = hex(id(self))
		try:
			t = '{} ({})'.format(self.type,str(self.nstates))
		except:
			t = self.type

		return '[{}] {} - {}'.format(mem,t,self.time_made)

	def idealize(self):
		pass
