import numpy as np
import logging
logger = logging.getLogger(__name__)
import h5py as h
from .model_container import model_container
from .modeler_io import export_dict_to_group, load_group_to_dict

default_prefs = """
[modeler]
converge = 1e-8
maxiters = 100
threshold = 0.5
threshold_jump_delta = 0.1
threshold_jump_n = 1
threshold_jump_sigma = 3
dtype = "FRET"
clip = True
clip_min = -1.
clip_max = 2.0

[modeler.nrestarts]
default = 5
min = 1

[modeler.dwells]
include_first = True
include_last = False
fix_norm = False

[modeler.kmeans.nstates]
default = 2
min = 1
max = 1000

[modeler.mlgmm]
nstates = 2

[modeler.mlhmm]
nstates = 2

[modeler.vbgmm]
nstates = 2
nstates_min = 1
nstates_max = 6
[modeler.vbgmm.prior]
beta = 0.25
a = 0.1
b = 0.01
pi = 1

[modeler.vbhmm]
nstates = 2
nstates_min = 1
nstates_max = 6
[modeler.vbhmm.prior]
beta = 0.25
a = 2.5
b = 0.01
alpha = 1.
pi = 1

[modeler.vbconhmm]
nstates = 2
nstates_min = 1
nstates_max = 6
[modeler.vbconhmm.prior]
beta = 0.25
a = 2.5
b = 0.01
alpha = 1.
pi = 1.

[modeler.ebhmm]
nstates = 2
nstates_min = 1
nstates_max = 6
[modeler.ebhmm.prior]
beta = 0.25
a = 2.5
b = 0.01
alpha = 1.
pi = 1.

[modeler.hhmm]
tolerance =  1e-4
maxiters = 100
restarts = 2

[modeler.biasd]
tau = 1.0
likelihood = 'Python'
nwalkers = 96
thin = 1000
steps = 200
filename = './biasd.hdf5'
[modeler.biasd.prior.e1]
type = 'Normal'
p1 = 0.0
p2 = 0.1
[modeler.biasd.prior.e2]
type = 'Normal'
p1 = 1.0
p2 = 0.1
[modeler.biasd.prior.sigma1]
type = 'Log-uniform'
p1 = 0.01
p2 = 1.0
[modeler.biasd.prior.sigma2]
type = 'Log-uniform'
p1 = 0.01
p2 = 1.0
[modeler.biasd.prior.k1]
type = 'Log-uniform'
p1 = 0.001
p2 = 1000.0
[modeler.biasd.prior.k2]
type = 'Log-uniform'
p1 = 0.001
p2 = 1000.0
"""

class controller_modeler(object):
	''' Handles modeling data

	* several .get_* functions to help setup models
	* several model specific io functions (save,load)
	* direct modeling functions (should basically not know about a GUI/maven. input is all relevant variables etc.). Individual modes will get the relevant data together and then call these functions which will dump the result into .result


	'''

	def __init__(self,maven):
		super().__init__()
		self.maven = maven
		self.maven.prefs.add(default_prefs)
		# self.cached_functions = {}
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

	
	'''
	-------------------------- Accessories --------------------------
	'''

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
		model_dict = result.convert_to_dict()

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
					result.idealize = lambda : self.idealize_threshold(result)
				elif result.type in ["kmeans","k-means","Kmeans"]:
					result.idealize = lambda : self.idealize_kmeans(result)
				elif result.type in ["ml GMM","ML GMM","vb GMM","VB GMM"]:
					result.idealize = lambda : self.idealize_gmm(result)
				elif result.type in ["VB Consensus HMM",'vb Consensus HMM']:
					result.idealize = lambda : self.idealize_threshold(result)

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

	def get_trace_keep(self,all=False):
		okay = np.ones(self.maven.data.nmol,dtype='bool')
		if not all:
			checked = self.maven.data.flag_ons
			longenough = self.maven.data.post_list-self.maven.data.pre_list >= 2 ## HMMs need more than 1
			okay = np.bitwise_and(checked,longenough)			
		return okay

	def get_data(self,dtype):
		if type(dtype) is str:
			if dtype.lower() == 'sum':
				return True,self.maven.data.corrected[:,:].sum(2).astype('double')
			elif dtype.lower().startswith('rel ') or dtype.lower().startswith('r'):
				ind = int(dtype[-1])
				return True,self.maven.calc_relative()[:,:,ind].astype('double')
			elif dtype.lower() == 'fret':
				return True,self.maven.calc_relative()[:,:,1].astype('double')
			elif int(dtype) in [0,1,2,3,4,5,6,7,8,9]:
				return True,self.maven.data.corrected[:,:,int(dtype)].astype('double')
		elif type(dtype) is int:
			if dtype in [0,1,2,3,4,5,6,7,8,9]:
				return True,self.maven.data.corrected[:,:,dtype].astype('double')
		return False,None

	def get_traces(self,dtype='sum',all=False):
		keep = self.get_trace_keep(all)
		if keep.sum() == 0:
			logger.info('Failed to get traces')
			return False,keep,[]
		pre,post = self.get_prepost()

		success,data = self.get_data(dtype)
		if not success:
			logger.info('Failed to get traces')
			return False,keep,[]
		
		y = []
		for i in range(keep.size):
			if keep[i]:
				y.append(data[i,pre[i]:post[i]])

		if self.maven.prefs['modeler.clip']:
			y = self.clip_traces(y,self.maven.prefs['modeler.clip_min'],self.maven.prefs['modeler.clip_max'])
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
	
	def get_priors(self, model_type, nstates = None, y_flat = None):

		prior_beta = self.maven.prefs[model_type + '.prior.beta']
		prior_a = self.maven.prefs[model_type + '.prior.a']
		prior_b = self.maven.prefs[model_type + '.prior.b']
		prior_pi = self.maven.prefs[model_type + '.prior.pi']

		if "gmm" in model_type:
			return np.array([prior_beta, prior_a, prior_b, prior_pi])
		elif "hmm" in model_type:
			mu_prior = np.percentile(y_flat,np.linspace(0,100,nstates+2))[1:-1]
			beta_prior = np.ones_like(mu_prior)*prior_beta
			a_prior = np.ones_like(mu_prior)*prior_a
			b_prior = np.ones_like(mu_prior)*prior_b
			pi_prior = np.ones_like(mu_prior)*prior_pi
			tm_prior = np.ones((nstates,nstates))*self.maven.prefs[model_type + '.prior.alpha']

			return [mu_prior, beta_prior, a_prior, b_prior, pi_prior, tm_prior]

	def get_model_specs(self):
		maxiters = self.maven.prefs['modeler.maxiters']
		converge = self.maven.prefs['modeler.converge']
		nrestarts = self.maven.prefs['modeler.nrestarts']
		ncpu = self.maven.prefs['ncpu']

		return maxiters, converge, nrestarts, ncpu

	def make_report(self,model):

		type = model.type
		s = '\nModel type = {}\n'.format(type)

		N = len(model.ran)
		s += 'Model ran on {} traces\n'.format(N)

		nstates = model.nstates
		s += 'nstates = {}\n'.format(nstates)

		mean = model.mean
		s += 'means = {}\n'.format(mean)

		var = model.var
		s += 'vars = {}\n'.format(var)

		frac = model.frac
		s += 'fracs = {}\n'.format(frac)

		if not model.tmatrix is None:
			tmatrix = model.tmatrix
			try:
				norm_tmatrix = model.norm_tmatrix
			except:
				from .fxns.hmm import normalize_tmatrix
				model.norm_tmatrix = normalize_tmatrix(tmatrix)
				norm_tmatrix = model.norm_tmatrix
			s += 'tmatrix = \n{}\n'.format(tmatrix)
			s += 'tmatrix normalized = \n{}\n'.format(norm_tmatrix)

		if not model.rates is None:
			s += 'Rate type = {}\n'.format(model.rate_type)
			s += 'Rates (frame^-1) = \n{}\n'.format(model.rates)
			try:
				s += 'Rates uncertainty (1 sigma; frame^-1) = \n{}\n'.format(model.rates_stddev)
			except:
				pass

		logger.info(s)
		return s
		

	def clip_traces(self,y,low=-1,high=2):
		np.random.seed(666)
		for i in range(len(y)):
			yy = y[i]
			## Clip traces and redistribute randomly
			bad = np.bitwise_or((yy < low),np.bitwise_or((yy > high),np.isnan(yy)))
			yy[bad] = np.random.uniform(low=low,high=high,size=int(bad.sum()))
			y[i] = yy
		import time
		np.random.seed(int(time.time()*1000)%(2**32-1))
		return y

	def recast_rs(self,result):
		N = self.maven.data.nmol
		T = self.maven.data.nt
		rs = result.r
		result.r = np.zeros((N,T,result.nstates)) + np.nan
		for i,ii in enumerate(result.ran):
			# print(i,ii)
			pre = self.maven.data.pre_list[ii]
			post = self.maven.data.post_list[ii]
			result.r[ii,pre:post] = rs[i]

	'''
	-------------------------- Models --------------------------
	'''

	def run_threshold(self):
		dtype = self.maven.prefs['modeler.dtype']
		threshold = self.maven.prefs['modeler.threshold']
		success,keep,y = self.get_traces(dtype)
		if not success:
			return
		
		from .threshold import calc_threshold
		result = calc_threshold(np.concatenate(y),threshold)
		result.dtype = dtype
		result.ran = np.nonzero(keep)[0].tolist()
		result.idealize = lambda : self.idealize_threshold(result)
		result.idealize()
		self.model = result
		self.make_report(result)
		self.maven.emit_data_update()
	
	def run_threshold_jump(self):
		dtype = self.maven.prefs['modeler.dtype']
		threshold_jump_delta = self.maven.prefs['modeler.threshold_jump_delta']
		threshold_jump_n = self.maven.prefs['modeler.threshold_jump_n']
		threshold_jump_sigma = self.maven.prefs['modeler.threshold_jump_sigma']
		success,keep,y = self.get_traces(dtype)
		if not success:
			return
		
		from .threshold import calc_threshold_jump
		result = calc_threshold_jump(np.concatenate(y),threshold_jump_delta,threshold_jump_n,threshold_jump_sigma)
		result.dtype = dtype
		result.ran = np.nonzero(keep)[0].tolist()
		result.idealize = lambda : self.idealize_threshold_jump(result)
		result.idealize()
		self.model = result
		self.make_report(result)
		self.maven.emit_data_update()

	'''
	######################### Mixtures ######################### 
	'''

	def run_kmeans(self):
		nstates = self.maven.prefs['modeler.kmeans.nstates']

		dtype = self.maven.prefs['modeler.dtype']
		success,keep,y = self.get_traces(dtype)
		if not success:
			return

		from .kmeans import kmeans
		result = kmeans(np.concatenate(y),nstates)
		result.dtype = dtype
		result.ran = np.nonzero(keep)[0].tolist()
		# print(result.ran)
		result.idealize = lambda : self.idealize_kmeans(result)
		result.idealize()
		self.recast_rs(result)
		self.model = result
		self.make_report(result)
		self.maven.emit_data_update()

	def run_mlgmm(self):
		nstates = self.maven.prefs['modeler.mlgmm.nstates']

		dtype = self.maven.prefs['modeler.dtype']
		success,keep,y = self.get_traces(dtype)
		if not success:
			return

		maxiters, converge, nrestarts, ncpu = self.get_model_specs()

		from .gmm_ml import ml_em_gmm,ml_em_gmm_parallel
		result = ml_em_gmm_parallel(np.concatenate(y),nstates,maxiters,converge,nrestarts,ncpu)
		result.ran = np.nonzero(keep)[0].tolist()
		result.dtype = dtype
		result.idealize = lambda : self.idealize_gmm(result)
		result.idealize()
		self.recast_rs(result)
		self.model = result
		self.make_report(result)
		self.maven.emit_data_update()

	def run_vbgmm(self):
		nstates = self.maven.prefs['modeler.vbgmm.nstates']

		dtype = self.maven.prefs['modeler.dtype']
		success,keep,y = self.get_traces(dtype)
		if not success:
			return

		maxiters, converge, nrestarts, ncpu = self.get_model_specs()
		priors = self.get_priors('modeler.vbgmm')

		from .gmm_vb import vb_em_gmm,vb_em_gmm_parallel
		result = vb_em_gmm_parallel(np.concatenate(y),nstates,maxiters,converge,nrestarts,priors,ncpu)
		result.dtype = dtype
		result.ran = np.nonzero(keep)[0].tolist()
		result.idealize = lambda : self.idealize_gmm(result)
		result.idealize()
		self.recast_rs(result)
		self.model = result
		self.make_report(result)
		self.maven.emit_data_update()

	def run_vbgmm_modelselection(self):
		nstates_min = self.maven.prefs['modeler.vbgmm.nstates_min']
		nstates_max = self.maven.prefs['modeler.vbgmm.nstates_max']

		dtype = self.maven.prefs['modeler.dtype']
		success,keep,y = self.get_traces(dtype)
		if not success:
			return

		maxiters, converge, nrestarts, ncpu = self.get_model_specs()
		priors = self.get_priors('modeler.vbgmm')

		from .gmm_vb import vb_em_gmm_parallel
		results = []
		for nstates in range(nstates_min,nstates_max+1):
			result = vb_em_gmm_parallel(np.concatenate(y),nstates,maxiters,converge,nrestarts,priors,ncpu)
			result.ran = np.nonzero(keep)[0].tolist()
			result.dtype = dtype
			result.idealize = lambda : self.idealize_gmm(result)
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

		self.maven.emit_data_update()


	'''
	######################### Composite HMMs ######################### 
	'''

	def run_kmeans_mlhmm(self):
		nstates = self.maven.prefs['modeler.mlhmm.nstates']

		dtype = self.maven.prefs['modeler.dtype']
		success,keep,y = self.get_traces(dtype)
		if not success:
			return
		
		specs = self.get_model_specs()

		from .hmm_ml import tracelevel_mlhmm

		trace_level, idealized, ran = tracelevel_mlhmm(y, keep, nstates, specs,
													   self.maven.data.nmol, self.maven.data.nt,
													   self.maven.data.pre_list, self.maven.data.post_list,
													   dtype)
		from .kmeans import kmeans
		vits = np.concatenate(idealized)
		vits = vits[np.isfinite(vits)]
		result = kmeans(vits,nstates)
		result.trace_level = trace_level
		result.type = "kmeans + ml HMM"
		result.ran = ran
		result.dtype = dtype

		result.rate_type = "Transition Matrix"
		from .fxns.hmm import compose_tmatrix, normalize_tmatrix,convert_tmatrix 
		tmatrix = compose_tmatrix(y,result)
		result.tmatrix = tmatrix
		result.norm_tmatrix = normalize_tmatrix(tmatrix)
		result.rates, self.rates_stddev = convert_tmatrix(tmatrix)

		result.idealize = lambda : self.idealize_kmeans_viterbi(result,idealized)
		result.idealize()
		self.recast_rs(result)
		self.model = result
		self.make_report(result)
		self.maven.emit_data_update()

	def run_kmeans_vbhmm(self):
		nstates = self.maven.prefs['modeler.vbhmm.nstates']

		dtype = self.maven.prefs['modeler.dtype']
		success,keep,y = self.get_traces(dtype)
		if not success:
			return
		
		from .hmm_vb import tracelevel_vbhmm_modelselection

		specs = self.get_model_specs()
		trace_level, idealized, ran = tracelevel_vbhmm_modelselection(y, keep, nstates, self.get_priors, specs,
																	self.maven.data.nmol, self.maven.data.nt,
																	self.maven.data.pre_list, self.maven.data.post_list,
														   			dtype)
		from .kmeans import kmeans
		vits = np.concatenate(idealized)
		vits = vits[np.isfinite(vits)]
		result = kmeans(vits,nstates)
		result.trace_level = trace_level
		result.type = "kmeans + vb HMM"
		result.ran = ran

		result.rate_type = "Transition Matrix"
		from .fxns.hmm import compose_tmatrix, normalize_tmatrix,convert_tmatrix 
		tmatrix = compose_tmatrix(y,result)
		result.tmatrix = tmatrix
		result.norm_tmatrix = normalize_tmatrix(tmatrix)
		result.rates, self.rates_stddev = convert_tmatrix(tmatrix)
		
		result.idealize = lambda : self.idealize_kmeans_viterbi(result,idealized)
		result.idealize()
		self.recast_rs(result)
		self.model = result
		self.make_report(result)
		self.maven.emit_data_update()


	def run_vbgmm_vbhmm(self):
		nstates = self.maven.prefs['modeler.vbhmm.nstates']

		dtype = self.maven.prefs['modeler.dtype']
		success,keep,y = self.get_traces(dtype)
		if not success:
			return

		from .hmm_vb import tracelevel_vbhmm_modelselection

		specs = self.get_model_specs()
		maxiters, converge, nrestarts, ncpu = specs
		y_flat = np.concatenate(y)

		trace_level, idealized, ran = tracelevel_vbhmm_modelselection(y, keep, nstates, self.get_priors, specs,
																	self.maven.data.nmol, self.maven.data.nt,
																	self.maven.data.pre_list, self.maven.data.post_list,
														   			dtype)
		vits = np.concatenate(idealized)
		vits = vits[np.isfinite(vits)]
		priors = self.get_priors('modeler.vbgmm')

		from .gmm_vb import vb_em_gmm_parallel

		result = vb_em_gmm_parallel(vits,nstates,maxiters,converge,nrestarts,priors=priors,ncpu=ncpu)
		result.trace_level = trace_level
		result.type = "vb GMM + vb HMM"
		result.dtype = dtype
		result.ran = ran

		result.rate_type = "Transition Matrix"
		from .fxns.hmm import compose_tmatrix, normalize_tmatrix,convert_tmatrix 
		tmatrix = compose_tmatrix(y,result)
		result.tmatrix = tmatrix
		result.norm_tmatrix = normalize_tmatrix(tmatrix)
		result.rates, self.rates_stddev = convert_tmatrix(tmatrix)

		result.idealize = lambda : self.idealize_gmm_viterbi(result,idealized)
		result.idealize()

		viterbi_var = result.var
		#var = (result.r*((y_flat[:,None] - result.mean[None,:])**2)).sum(0)/(result.r).sum()
		var = np.zeros_like(result.mean)
		for i,state in enumerate(np.argmax(result.r, axis = 1)):
			var[state] += (y_flat[i] - result.mean[state])**2

		var /= result.r.sum()

		result.var = var
		result.viterbi_var = viterbi_var
		self.recast_rs(result)

		self.model = result
		self.make_report(result)
		self.maven.emit_data_update()

	def run_vbgmm_vbhmm_modelselection(self):
		nstates_min = self.maven.prefs['modeler.vbhmm.nstates_min']
		nstates_max = self.maven.prefs['modeler.vbhmm.nstates_max']
		if nstates_min > nstates_max:
			logger.info('nstates min > max')
			return

		dtype = self.maven.prefs['modeler.dtype']
		success,keep,y = self.get_traces(dtype)
		if not success:
			return

		specs = self.get_model_specs()
		maxiters, converge, nrestarts, ncpu = specs
		results_ens = []
		y_flat = np.concatenate(y)

		from .hmm_vb import tracelevel_vbhmm_modelselection

		for nstates in range(nstates_min,nstates_max+1):
			trace_level, idealized, ran = tracelevel_vbhmm_modelselection(y, keep, nstates, self.get_priors, specs,
																		self.maven.data.nmol, self.maven.data.nt,
																		self.maven.data.pre_list, self.maven.data.post_list,
														   				dtype)

			vits = np.concatenate(idealized)
			vits = vits[np.isfinite(vits)]
			priors = self.get_priors('modeler.vbgmm')

			from .gmm_vb import vb_em_gmm_parallel
			result = vb_em_gmm_parallel(vits,nstates,maxiters,converge,nrestarts,priors=priors,ncpu=ncpu)
			result.trace_level = trace_level
			result.type = "vb GMM + vb HMM"
			result.dtype = dtype
			result.ran = ran

			result.rate_type = "Transition Matrix"
			from .fxns.hmm import compose_tmatrix, normalize_tmatrix,convert_tmatrix 
			tmatrix = compose_tmatrix(y,result)
			result.tmatrix = tmatrix
			result.norm_tmatrix = normalize_tmatrix(tmatrix)
			result.rates, self.rates_stddev = convert_tmatrix(tmatrix)

			result.idealize = lambda : self.idealize_gmm_viterbi(result,idealized)
			result.idealize()

			viterbi_var = result.var
			#var = (result.r*((y_flat[:,None] - result.mean[None,:])**2)).sum(0)/(result.r).sum()
			var = np.zeros_like(result.mean)
			for i,state in enumerate(np.argmax(result.r, axis = 1)):
				var[state] += (y_flat[i] - result.mean[state])**2

			var /= result.r.sum()

			result.var = var
			result.viterbi_var = viterbi_var
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


	def run_threshold_vbhmm(self):
		nstates = self.maven.prefs['modeler.vbhmm.nstates']
		threshold = self.maven.prefs['modeler.threshold']

		dtype = self.maven.prefs['modeler.dtype']
		success,keep,y = self.get_traces(dtype)
		if not success:
			return
		
		from .hmm_vb import tracelevel_vbhmm_modelselection

		specs = self.get_model_specs()
		trace_level, idealized, ran = tracelevel_vbhmm_modelselection(y, keep, nstates, self.get_priors, specs,
																	self.maven.data.nmol, self.maven.data.nt,
																	self.maven.data.pre_list, self.maven.data.post_list,
														   			dtype)
		vits = np.concatenate(idealized)
		vits = vits[np.isfinite(vits)]
		from .threshold import calc_threshold
		result = calc_threshold(vits,threshold)
		result.trace_level = trace_level
		result.type = "threshold + vb HMM"
		result.dtype = dtype
		result.ran = ran
		result.idealize = lambda : self.idealize_threshold_viterbi(result,idealized)
		result.idealize()
		self.model = result
		self.make_report(result)
		self.maven.emit_data_update()

	'''
	######################### Global HMMs ######################### 
	'''

	def run_vbconhmm(self):
		nstates = self.maven.prefs['modeler.vbconhmm.nstates']

		dtype = self.maven.prefs['modeler.dtype']
		success,keep,y = self.get_traces(dtype)
		if not success:
			return
		
		maxiters, converge, nrestarts, ncpu = self.get_model_specs()

		priors = self.get_priors('modeler.vbconhmm', nstates, np.concatenate(y))
		from .hmm_vb_consensus import consensus_vb_em_hmm_parallel
		result = consensus_vb_em_hmm_parallel(y,nstates,maxiters,converge,nrestarts,priors=priors,ncpu=ncpu)

		result.ran = np.nonzero(keep)[0].tolist()
		result.dtype = dtype
		result.idealize = lambda : self.idealize_hmm(result)
		result.idealize()
		self.recast_rs(result)

		self.model = result
		self.make_report(result)
		self.maven.emit_data_update()

	def run_vbconhmm_modelselection(self):
		nstates_min = self.maven.prefs['modeler.vbconhmm.nstates_min']
		nstates_max = self.maven.prefs['modeler.vbconhmm.nstates_max']
		if nstates_min > nstates_max:
			logger.info('nstates min > max')
			return

		dtype = self.maven.prefs['modeler.dtype']
		success,keep,y = self.get_traces(dtype)
		if not success:
			return
		
		maxiters, converge, nrestarts, ncpu = self.get_model_specs()

		nmol = self.maven.data.nmol
		nt = self.maven.data.nt
		from .hmm_vb_consensus import consensus_vb_em_hmm_parallel

		results = []
		for nstates in range(nstates_min,nstates_max+1):
			priors = self.get_priors('modeler.vbconhmm', nstates, np.concatenate(y))

			result = consensus_vb_em_hmm_parallel(y,nstates,maxiters,converge,nrestarts,priors=priors,ncpu=ncpu)
			result.idealized = np.zeros((nmol,nt)) + np.nan
			result.ran = np.nonzero(keep)[0].tolist()
			result.dtype = dtype
			result.idealize = lambda : self.idealize_hmm(result)
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

	def run_threshold_vbconhmm(self):
		nstates = self.maven.prefs['modeler.vbconhmm.nstates']
		threshold = self.maven.prefs['modeler.threshold']

		dtype = self.maven.prefs['modeler.dtype']
		success,keep,y = self.get_traces(dtype)
		if not success:
			return
		
		maxiters, converge, nrestarts, ncpu = self.get_model_specs()

		priors = self.get_priors('modeler.vbconhmm', nstates, np.concatenate(y))

		from .hmm_vb_consensus import consensus_vb_em_hmm_parallel

		result = consensus_vb_em_hmm_parallel(y,nstates,maxiters,converge,nrestarts,priors=priors,ncpu=ncpu)
		result.dtype = dtype
		result.ran = np.nonzero(keep)[0].tolist()
		result.idealize = lambda : self.idealize_hmm(result)
		result.idealize()
		
		self.recast_rs(result)

		con_result = result
		idealized = result.idealized.copy()
		vits = np.concatenate(idealized)
		vits = vits[np.isfinite(vits)]
		from .threshold import calc_threshold
		result = calc_threshold(vits,threshold)
		result.dtype = dtype
		result.threshold = threshold
		result.consensusmodel = con_result
		result.type = "threshold + vbConsensus"
		result.ran = np.nonzero(keep)[0].tolist()
		result.idealize = lambda : self.idealize_threshold_viterbi(result,idealized)
		result.idealize()
		self.model = result
		self.make_report(result)
		self.maven.emit_data_update()

	def run_ebhmm(self):
		nstates = self.maven.prefs['modeler.ebhmm.nstates']

		dtype = self.maven.prefs['modeler.dtype']
		success,keep,y = self.get_traces(dtype)
		if not success:
			return

		maxiters, converge, nrestarts, ncpu = self.get_model_specs()

		priors = self.get_priors('modeler.ebhmm', nstates, np.concatenate(y))

		from .hmm_eb import eb_em_hmm, trace_level_eb
		result, vbs = eb_em_hmm(y,nstates,maxiters,nrestarts,converge,priors=priors,ncpu=ncpu)
		result.ran = np.nonzero(keep)[0].tolist()
		result.dtype = dtype

		trace_level, rs, idealized, chain = trace_level_eb(y, vbs, result.ran, 
													 	   self.maven.data.pre_list, self.maven.data.post_list,
														   self.maven.data.nmol, self.maven.data.nt,
														   dtype)

		result.trace_level = trace_level
		result.r = rs
		self.recast_rs(result)
		result.idealized = idealized
		result.chain = chain

		self.model = result
		self.make_report(result)
		self.maven.emit_data_update()

	def run_ebhmm_modelselection(self):
		nstates_min = self.maven.prefs['modeler.ebhmm.nstates_min']
		nstates_max = self.maven.prefs['modeler.ebhmm.nstates_max']
		if nstates_min > nstates_max:
			logger.info('nstates min > max')
			return
		
		dtype = self.maven.prefs['modeler.dtype']
		success,keep,y = self.get_traces(dtype)
		if not success:
			return

		maxiters, converge, nrestarts, ncpu = self.get_model_specs()

		results = []

		from .hmm_eb import eb_em_hmm, trace_level_eb

		for nstates in range(nstates_min,nstates_max+1):
			priors = self.get_priors('modeler.ebhmm', nstates, np.concatenate(y))

			result, vbs = eb_em_hmm(y,nstates,maxiters,nrestarts,converge,priors=priors,ncpu=ncpu)
			result.ran = np.nonzero(keep)[0].tolist()
			result.dtype = dtype

			trace_level, rs, idealized, chain = trace_level_eb(y, vbs, result.ran, 
													  		   self.maven.data.pre_list, self.maven.data.post_list,
															   self.maven.data.nmol, self.maven.data.nt,
															   dtype)

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

	'''
	######################### Dwell analysis ######################### 
	'''
	
	def run_dwell_analysis(self, ftype, state, fix_A = False):
		tau,surv = self.get_survival_dwells(state)

		from .dwells import optimize_single_surv, optimize_double_surv, optimize_triple_surv, optimize_stretch_surv
		fxns = {
			'Single Exponential':optimize_single_surv,
			'Double Exponential':optimize_double_surv,
			'Triple Exponential':optimize_triple_surv,
			'Stretched Exponential':optimize_stretch_surv,
		}
		rates = fxns[ftype](tau,surv,fix_A)

		model = self.model

		if model.rate_type in ["Transition Matrix", "N/A"]:
			model.rate_type = "Dwell Analysis"
			model.rates = {}

		model.rates[state] = {}
		model.rates[state]['ks'] = rates[0]
		model.rates[state]['As'] = rates[-3]
		model.rates[state]['error'] = rates[-2]
		model.rates[state]['R2'] = rates[-1]

		if ftype == "Stretched Exponential":
			model.rates[state]["betas"] = rates[1]
		elif "betas" in model.rates:
			model.rates.pop('betas')

		self.model = model

	def run_tmatrix(self):
		model = self.model

		if not model.tmatrix is None:
			from .fxns.hmm import convert_tmatrix
			model.rates, model.rates_stddev = convert_tmatrix(model.tmatrix)
			model.rate_type = "Transition Matrix"
		else:
			return

	'''
	######################### Experimental ######################### 
	'''

	def run_hhmm(self):
		dtype = self.maven.prefs['modeler.dtype']
		success,keep,y = self.get_traces(dtype)
		if not success:
			return
	
		max_iter = self.maven.prefs['modeler.hhmm.maxiters']
		tol = self.maven.prefs['modeler.hhmm.tolerance']
		restarts = self.maven.prefs['modeler.hhmm.restarts']
		guess = np.array([0., 1.])
		depth_vec = [2]
		prod_states = 2

		from .hhmm_vmp import vmp_hhmm
		post_mu, post_beta, post_a, post_b, post_pi, post_tm, post_exit,likelihood = self.cached_vmphhmm(
			y,
			depth_vec,
			prod_states,
			max_iter,
			tol,
			restarts,
			guess
		)

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

		tree = {}
		for i in range(len(post_pi)):
			tree_level = {}
			tree_level['pi'] = np.array(post_pi[i])
			tree_level['tm'] = np.array(post_tm[i])
			tree_level['exit'] = np.array(post_exit[i])
			tree['d={}'.format(i+2)] = tree_level

		from .model_container import model_container

		result = model_container(type='hierarchical HMM',
			nstates = 4,mean=mu,var=var,frac=pi,
			tmatrix=flat_tm,
			likelihood=likelihood,
			a=post_a,b=post_b,beta=post_beta, tree=tree)
		result.dtype = dtype
		result.ran = [np.nonzero(keep)[0]]
		self.model = result
		self.make_report(result)
		self.maven.emit_data_update()


	def run_biasd_safeimports(self):
		try:
			import biasd as b

			import tempfile
			import shutil
			import emcee
			import time
			import h5py
			import os
			return b,tempfile,shutil,emcee,time,h5py,os
		except:
			print('You need BIASD installed to run BIASD')
			print('see https://github.com/ckinzthompson/biasd')
			return None,None,None,None,None,None,None

	def run_biasd_checkfname(self):
		import os
		import h5py
		fname = self.maven.prefs['modeler.biasd.filename']
		if not os.path.exists(fname):
			return False,""
		else:
			with h5py.File(fname,'r') as f:
				if not 'biasd mcmc' in f:
					return False,""
		return True,fname

	def run_biasd_fithistogram(self):
		import matplotlib.pyplot as plt
		b,tempfile,shutil,emcee,time,h5py,os = self.run_biasd_safeimports()
		if b is None:
			return
		b.likelihood.use_python_numba_ll_2sigma()

		dtype = self.maven.prefs['modeler.dtype']
		success,keep,y = self.get_traces(dtype)
		if not success:
			return
		data = np.concatenate(y)
		
		tau = float(self.maven.prefs['modeler.biasd.tau'])
		
		dmin,dmax = np.percentile(data,[20,80])
		std = np.sqrt(np.var(data)/10.)
		guess = np.array((dmin,dmax,std,std,1./tau,1./tau))
		theta,covars= b.histogram.fit_histogram(data,tau,guess)
	
		xmin = theta[0]-3*theta[2]
		xmax = theta[1]+3*theta[3]
		nbins = 201
		x = np.linspace(xmin,xmax,nbins*10)
		y = np.exp(b.likelihood.nosum_log_likelihood(theta,x,tau))
		
		fig,ax = plt.subplots(1,dpi=100)
		nbins = 201
		ax.hist(data,bins=nbins,range=(data.min(),data.max()),density=True,alpha=.8,histtype='step',color='tab:blue')
		ax.plot(x,y,color='tab:red',alpha=.8)

		ax.set_ylabel('Probability Density',fontsize=10)
		ax.set_xlabel('Signal',fontsize=10)
		fig.tight_layout()
		plt.show()

	def run_biasd_loadinfo(self):
		b,tempfile,shutil,emcee,time,h5py,os = self.run_biasd_safeimports()
		if b is None:
			return
		
		success,fname = self.run_biasd_checkfname()
		if not success:
			return

		## Load in the initializations
		with h5py.File(fname,'r') as f:
			if not 'biasd mcmc' in f:
				return
			g = f['biasd mcmc']
			tau = float(g.attrs['tau'])
			pstring = g.attrs['prior']
			prior = pstring.split('\n')
			plabels = [priori.split(',')[0] for priori in prior]
			ptypes = [priori.split(',')[1] for priori in prior]
			pparam1 = [float(priori.split(',')[2]) for priori in prior]
			pparam2 = [float(priori.split(',')[3]) for priori in prior]
			nwalkers = int(g.attrs["nwalkers"])
			# chain = g['chain'][:]

		self.maven.prefs.__setitem__('modeler.biasd.tau',tau,quiet=True)	
		self.maven.prefs.__setitem__('modeler.biasd.nwalkers',nwalkers,quiet=True)	

		for i in range(len(ptypes)):
			if ptypes[i] == ' normal':
				stype = 'Normal'
			elif ptypes[i] == ' uniform':
				stype = 'Uniform'
			elif ptypes[i] == ' log uniform':
				stype = 'Log-uniform'
			self.maven.prefs.__setitem__(f'modeler.biasd.prior.{plabels[i]}.type',stype,quiet=True)	
			self.maven.prefs.__setitem__(f'modeler.biasd.prior.{plabels[i]}.p1',pparam1[i],quiet=True)	
			self.maven.prefs.__setitem__(f'modeler.biasd.prior.{plabels[i]}.p2',pparam2[i],quiet=True)	
		self.maven.prefs.emit_changed()
	
	def run_biasd_assembleprior(self):
		b,tempfile,shutil,emcee,time,h5py,os = self.run_biasd_safeimports()
		if b is None:
			return

		labels = ['e1','e2','sigma1','sigma2','k1','k2']
		pdists = []
		for key in labels:
			stype = self.maven.prefs[f'modeler.biasd.prior.{key}.type']
			p1 = self.maven.prefs[f'modeler.biasd.prior.{key}.p1']
			p2 = self.maven.prefs[f'modeler.biasd.prior.{key}.p2']
			if stype == 'Normal':
				pdists.append(b.distributions.normal(p1,p2))
			elif stype == 'Uniform':
				pdists.append(b.distributions.uniform(p1,p2))
			elif stype == 'Log-uniform':
				pdists.append(b.distributions.loguniform(p1,p2))
		prior = b.distributions.collection_standard_2sigma(*pdists)
		return prior

	def run_biasd_setupfile(self):
		b,tempfile,shutil,emcee,time,h5py,os = self.run_biasd_safeimports()
		if b is None:
			return

		ndim = 6
		prior = self.run_biasd_assembleprior()
		nwalkers = int(self.maven.prefs['modeler.biasd.nwalkers'])
		tau = float(self.maven.prefs['modeler.biasd.tau'])
		fname = self.maven.prefs['modeler.biasd.filename']

		dtype = self.maven.prefs['modeler.dtype']
		success,keep,y = self.get_traces(dtype)
		if not success:
			return
		data = np.concatenate(y)

		with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as temp_file: ## avoid weird dropbox lock issues
			temp_file_path = temp_file.name
		with emcee.backends.HDFBackend(temp_file_path,name='biasd mcmc') as backend:
			backend.reset(nwalkers, ndim)
			sampler, positions = b.mcmc.setup(data,prior,tau,nwalkers,backend=backend)
			assert np.all(positions[:,2:] > 0)
			# sampler.run_mcmc(positions, 0, progress=False) ## commit the initializations to file????
		del backend

		with h5py.File(temp_file_path,'a') as f: 	## Write Metadata
			g = f[f'biasd mcmc']
			g.create_dataset('data',data=data,compression='gzip',compression_opts=9)
			g.create_dataset('p0',data=positions,compression='gzip',compression_opts=9)
			g.attrs['tau'] = tau
			g.attrs['nwalkers'] = nwalkers
			g.attrs['ndim'] = ndim
			pstring = '\n'.join([f'{label}, {prior.parameters[label].name}, {prior.parameters[label].parameters[0]}, {prior.parameters[label].parameters[1]}' for label in prior.labels])
			g.attrs['prior'] = pstring
			f.flush()
		shutil.move(temp_file_path,fname) ## avoid weird dropbox lock issues
		if os.path.exists(temp_file_path):
			os.remove(temp_file_path)
		
		self.run_biasd_loadinfo() ## test 
		print(f'Created {fname}')

	def run_biasd_mcmc(self, stochastic=False):
		b,tempfile,shutil,emcee,time,h5py,os = self.run_biasd_safeimports()
		if b is None:
			return
		b.likelihood.use_python_numba_ll_2sigma()
		
		success,fname = self.run_biasd_checkfname()
		if not success:
			return
	
		## transfer information from file into preferences
		self.run_biasd_loadinfo()
		
		ndim = 6
		nwalkers = int(self.maven.prefs['modeler.biasd.nwalkers'])
		tau = float(self.maven.prefs['modeler.biasd.tau'])
		fname = self.maven.prefs['modeler.biasd.filename']
		thin = self.maven.prefs['modeler.biasd.thin']
		steps = self.maven.prefs['modeler.biasd.steps']
		prior = self.run_biasd_assembleprior()
		
		with h5py.File(fname,'r') as f:
			g = f['biasd mcmc']
			data = g['data'][:]
			p0 = g['p0'][:]

		## possibly thin the data....
		if stochastic:
			rng = np.random.default_rng(int(time.time()))
			data = rng.choice(data,size=np.min((thin,data.size)),replace=False)

		## Setup & run the sampler
		## Avoid weird lock issues with backed up file storage (eg dropbox) by working in a tempfile.
		## Also, tempfile means you don't overwrite the original yet until everything is complete...
		with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as temp_file: ## avoid weird dropbox lock issues
			temp_file_path = temp_file.name
		shutil.copy(fname,temp_file_path) ## avoid weird dropbox lock issues

		with emcee.backends.HDFBackend(temp_file_path,name='biasd mcmc') as backend:
			sampler,_ = b.mcmc.setup(data,prior,tau,nwalkers,backend=backend)
			sampler.run_mcmc(p0, steps, progress=True) ## Production
			p0 = sampler.get_last_sample().coords.copy()
		del backend
		with h5py.File(temp_file_path,'a') as f:
			g = f['biasd mcmc']
			g['p0'][...] = p0
			f.flush()
		shutil.move(temp_file_path,fname) ## avoid weird dropbox lock issues

	def run_biasd_randomizep0(self,justdead=False):
		b,tempfile,shutil,emcee,time,h5py,os = self.run_biasd_safeimports()
		if b is None:
			return
		success,fname = self.run_biasd_checkfname()
		if not success:
			return 
		with h5py.File(fname,'r') as f:
				g = f['biasd mcmc']
				accepted = g['accepted'][:]
				p0 = g['p0'][:]
		
		if justdead:
			remove = accepted < np.max(accepted)//10
			keep = np.bitwise_not(remove)
		else:
			remove = accepted >= 0
			keep = accepted >= 0

		p0[:,2:] = np.log(p0[:,2:])  ## do any magnitude parameters in log-space to avoid negative values...
		mean = np.mean(p0[keep],axis=0)
		cov = np.cov(p0[keep].T)
		rng = np.random.default_rng(int(time.time()))
		p1 = rng.multivariate_normal(mean,cov,p0.shape[0])
		p0[remove] = p1[remove].copy()
		p0[:,2:] = np.exp(p0[:,2:])

		with h5py.File(fname,'a') as f:
			g = f['biasd mcmc']
			g['p0'][...] = p0
			f.flush()

	def run_biasd_analyze(self):
		b,tempfile,shutil,emcee,time,h5py,os = self.run_biasd_safeimports()
		if b is None:
			return
		import matplotlib.pyplot as plt
		b.likelihood.use_python_numba_ll_2sigma()

		success,fname = self.run_biasd_checkfname()
		if not success:
			return

		self.run_biasd_loadinfo()
		prior = self.run_biasd_assembleprior()
		tau = self.maven.prefs['modeler.biasd.tau']

		## load data and chain
		with h5py.File(fname,'r') as f:
			g = f['biasd mcmc']
			data = g['data'][:]
			lnp = g['log_prob'][:]
			chain = g['chain'][:]
			nt,nwalkers,ndim = chain.shape
		
		# print(nt,nwalkers,ndim)
		if nt < 1:
			return
		
		## plot parameter evolutions
		# fig,ax = plt.subplots(ndim)
		# for i in range(ndim):
		# 	for j in range(nwalkers):
		# 		ax[i].plot(chain[:,j,i],color='k',alpha=.15)
		# 		ax[i].set_ylabel(labels[i])
		# plt.show()

		## plot log prob evolution 
		fig,ax = plt.subplots(1,2,figsize=(10,6),dpi=100)
		lnpmax = lnp.max()
		for j in range(nwalkers):
			# ax[0].plot(lnp[:,j],color='k',alpha=.05)
			ax[0].plot(np.abs(lnp[:,j]-lnpmax)/lnpmax,color='k',alpha=.3)
		# ax[1].set_ylim(,1)
		ax[0].set_yscale('log')
		ax[0].set_xlabel('MCMC Step')
		ax[0].set_ylabel('Rel. ln Posterior')

		## show likelihood of MAP solution over histograms of full data
		indbest = np.where(lnp==lnp.max())
		best = chain[indbest[0][-1],indbest[1][-1]]

		# flatchain = chain[-1]
		# mean = flatchain.mean(0)
		# cov = np.cov(flatchain.T)
		# for ii in range(len(labels)):
		# 	print(f'{labels[ii]}: {mean[ii]:.4f} {best[ii]:.4f}')#+/- {np.sqrt(cov[ii,ii]):.4f}')
		# print(np.any(chain[:,:,2:]<=0.))
		# print('\n')

		nbins = 201
		xx = np.linspace(data.min(),data.max(),10*nbins)
		ax[1].hist(data,range=(xx.min(),xx.max()),bins=nbins,histtype='step',density=True,color='k',lw=1.5)
		for j in range(nwalkers):
			params = chain[-1,j]
			yy = np.exp(b.likelihood.nosum_log_likelihood(params,xx,tau))
			ax[1].plot(xx,yy,color='tab:blue',alpha=.3,zorder=1)
		params = best
		yy = np.exp(b.likelihood.nosum_log_likelihood(params,xx,tau))
		ax[1].plot(xx,yy,color='tab:red',lw=1.5)
		ax[1].set_title(r'$\tau=$ %.3f s'%(tau))
		ax[1].set_xlabel(r'Signal')
		ax[1].set_ylabel(r'Probability Density')
		# ax[1].set_xlim(0,1)
		plt.tight_layout()
		plt.show()


	'''
	-------------------------- Idealization --------------------------
	'''

	def idealize_threshold(self,result):
		success,data = self.get_data(result.dtype)
		if not success:
			return
		threshold = result.threshold
		result.idealized = np.zeros_like(data) + np.nan
		result.chain = np.zeros_like(result.idealized).astype('int')
		for ii in result.ran:
			pre = self.maven.data.pre_list[ii]
			post = self.maven.data.post_list[ii]
			idealpath = (data[ii,pre:post] > threshold).astype('int')
			result.chain[ii,pre:post] = idealpath.copy()
			result.idealized[ii,pre:post] = result.mean[idealpath]

	def idealize_threshold_jump(self,result):
		from .threshold import ideal_threshold_jump
		success,data = self.get_data(result.dtype)
		if not success:
			return
		result.idealized = np.zeros_like(data) + np.nan
		result.chain = np.zeros_like(result.idealized).astype('int')
		for ii in result.ran:
			pre = self.maven.data.pre_list[ii]
			post = self.maven.data.post_list[ii]
			chain,ideal = ideal_threshold_jump(data[ii],pre,post,result.threshold_jump_delta,result.threshold_jump_n,result.threshold_jump_sigma)
			result.chain[ii] = chain.copy()
			result.idealized[ii] = ideal.copy()

	def idealize_kmeans(self,result):
		success,data = self.get_data(result.dtype)
		if not success:
			return
		result.idealized = np.zeros_like(data) + np.nan
		result.chain = np.zeros_like(result.idealized).astype('int')

		for ii in result.ran:
			pre = self.maven.data.pre_list[ii]
			post = self.maven.data.post_list[ii]
			idealpath = np.abs(data[ii,pre:post,None] - result.mean[None,:]).argmin(1).astype('int64')
			result.chain[ii, pre:post] = idealpath.copy()
			result.idealized[ii, pre:post] = result.mean[idealpath]

	def idealize_gmm(self,result):
		success,data = self.get_data(result.dtype)
		if not success:
			return
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

	def idealize_kmeans_viterbi(self,result,vit):
		result.idealized = np.zeros_like(vit) + np.nan
		result.chain = np.zeros_like(result.idealized).astype('int')
		for ii in result.ran:
			pre = self.maven.data.pre_list[ii]
			post = self.maven.data.post_list[ii]
			idealpath = np.abs(vit[ii,pre:post,None] - result.mean[None,:]).argmin(1).astype('int64')
			result.chain[ii,pre:post] = idealpath.copy()
			result.idealized[ii, pre:post] = result.mean[idealpath]

	def idealize_gmm_viterbi(self,result, vit):
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

	def idealize_threshold_viterbi(self,result,vit):
		result.idealized = np.zeros_like(vit) + np.nan
		result.chain = np.zeros_like(result.idealized).astype('int')
		threshold = result.threshold

		for ii in result.ran:
			pre = self.maven.data.pre_list[ii]
			post = self.maven.data.post_list[ii]
			idealpath = (vit[ii,pre:post] > threshold).astype('int')
			result.chain[ii,pre:post] = idealpath.copy()
			result.idealized[ii,pre:post] = result.mean[idealpath]

	def idealize_hmm(self,result):
		from .fxns.hmm import viterbi

		success,data = self.get_data(result.dtype)
		if not success:
			return
		
		result.idealized = np.zeros_like(data) + np.nan
		result.chain = np.zeros_like(result.idealized).astype('int')
		pre = self.maven.data.pre_list
		post = self.maven.data.post_list
		for i in range(len(data)):
			if post[i]-pre[i]>=2:
				result.chain[i,pre[i]:post[i]] = viterbi(data[i][pre[i]:post[i]],result.mean,result.var,result.norm_tmatrix,result.frac).astype('int')
				result.idealized[i,pre[i]:post[i]] = result.mean[result.chain[i,pre[i]:post[i]]]


	'''
	def run_mlhmm(self):
		nstates = self.maven.prefs['modeler.mlhmm.nstates']

		dtype = self.maven.prefs['modeler.dtype']
		success,keep,y = self.get_traces(dtype)
		if not success:
			return

		maxiters, converge, nrestarts, ncpu = self.get_model_specs()

		from .fxns.hmm import viterbi
		from .model_container import trace_model_container
		from .hmm_ml import ml_em_hmm_parallel

		mu = np.ones(nstates)*np.nan
		var = mu.copy()
		frac = mu.copy()
		result = self.model_container(type='ml HMM',nstates=nstates,mean=mu,var =var,frac=frac)
		nmol = self.maven.data.nmol
		nt = self.maven.data.nt
		result.idealized = np.zeros((nmol,nt)) + np.nan
		trace_level = {}
		result.ran = np.nonzero(keep)[0].tolist()
		result.dtype = dtype

		for i in range(len(y)):
			yi = y[i].astype('double')
			r = ml_em_hmm_parallel(yi,nstates,maxiters,converge,nrestarts,ncpu)
			ii = result.ran[i]
			pre = self.maven.data.pre_list[ii]
			post = self.maven.data.post_list[ii]
			result.idealized[ii,pre:post] = r.mean[viterbi(yi,r.mean,r.var,r.tmatrix,r.frac).astype('int')]
			trace_level_inst = trace_model_container(r, ii)
			trace_level_inst.dtype = dtype
			trace_level_inst.idealized = result.idealized[ii]
			trace_level[str(ii)] = trace_level_inst

		result.trace_level = trace_level
		self.model = result
		self.make_report(result)
		self.maven.emit_data_update()

	def run_vbhmm(self):
		nstates = self.maven.prefs['modeler.vbhmm.nstates']

		dtype = self.maven.prefs['modeler.dtype']
		success,keep,y = self.get_traces(dtype)
		if not success:
			return

		maxiters, converge, nrestarts, ncpu = self.get_model_specs()

		priors = self.get_priors('modeler.vbhmm', nstates, np.concatenate(y))


		from .fxns.hmm import viterbi
		from .model_container import trace_model_container
		from .hmm_vb import vb_em_hmm_parallel

		mu = np.ones(nstates)*np.nan
		var = mu.copy()
		frac = mu.copy()
		result = self.model_container(type='vb HMM',nstates=nstates,mean=mu,var =var,frac=frac)

		result.idealized = np.zeros_like(self.maven.data.corrected[:,:,0]) + np.nan
		trace_level = {}
		result.ran = np.nonzero(keep)[0].tolist()
		result.dtype = dtype

		for i in range(len(y)):
			yi = y[i].astype('double')
			r = vb_em_hmm_parallel(yi,nstates,maxiters,converge,nrestarts,priors=priors,ncpu=ncpu)
			ii = result.ran[i]
			pre = self.maven.data.pre_list[ii]
			post = self.maven.data.post_list[ii]
			result.idealized[ii,pre:post] = r.mean[viterbi(yi,r.mean,r.var,r.tmatrix,r.frac).astype('int')]
			trace_level_inst = trace_model_container(r, ii)
			trace_level_inst.idealized = result.idealized[ii]
			trace_level_inst.dtype = dtype
			trace_level[str(ii)] = trace_level_inst

		result.trace_level = trace_level
		self.model = result
		self.make_report(result)
		self.maven.emit_data_update()

	def run_vbhmm_modelselection(self):
		nstates_min = self.maven.prefs['modeler.vbhmm.nstates_min']
		nstates_max = self.maven.prefs['modeler.vbhmm.nstates_max']
		if nstates_min > nstates_max:
			logger.info('nstates min > max')
			return

		dtype = self.maven.prefs['modeler.dtype']
		success,keep,y = self.get_traces(dtype)
		if not success:
			return

		maxiters, converge, nrestarts, ncpu = self.get_model_specs()

		from .fxns.hmm import viterbi
		from .model_container import trace_model_container
		from .hmm_vb import vb_em_hmm_parallel

		mu = np.ones(nstates_max)*np.nan
		var = mu.copy()
		frac = mu.copy()
		result = self.model_container(type='vb HMM_model selection',nstates=nstates_max,mean=mu,var =var,frac=frac)

		result.idealized = np.zeros_like(self.maven.data.corrected[:,:,0]) + np.nan
		trace_level = {}
		result.ran = np.nonzero(keep)[0].tolist()
		result.dtype = dtype
		y_flat = np.concatenate(y)

		for i in range(len(y)):
			results = []
			yi = y[i].astype('double')
			for nstates in range(nstates_min,nstates_max+1):
				priors = self.get_priors('modeler.vbhmm', nstates, y_flat)
				results.append(vb_em_hmm_parallel(yi,nstates,maxiters,converge,nrestarts,priors=priors,ncpu=ncpu))

			elbos = np.array([ri.likelihood[-1,0] for ri in results])
			modelmax = np.argmax(elbos)
			r = results[modelmax]
			ii = result.ran[i]
			pre = self.maven.data.pre_list[ii]
			post = self.maven.data.post_list[ii]
			result.idealized[ii,pre:post] = r.mean[viterbi(yi,r.mean,r.var,r.tmatrix,r.frac).astype('int')]
			trace_level_inst = trace_model_container(r, ii)
			trace_level_inst.dtype = dtype
			trace_level_inst.idealized = result.idealized[ii]
			trace_level[str(ii)] = trace_level_inst

		result.trace_level = trace_level
		self.model = result
		self.make_report(result)
		self.maven.emit_data_update()
	
	def run_vbhmm_one(self):
		nstates = self.maven.prefs['modeler.vbhmm.nstates']

		dtype = self.maven.prefs['modeler.dtype']
		success,keep,y = self.get_traces(dtype)
		if not success:
			return
		if keep.sum() != 1:
			## WARNING:"Error: Run all, Apply all","You have more than one molecule turned on"
			return
		
		maxiters, converge, nrestarts, ncpu = self.get_model_specs()

		priors = self.get_priors('modeler.vbhmm', nstates, np.concatenate(y))

		from .hmm_vb import vb_em_hmm_parallel
		result = vb_em_hmm_parallel(y[0].astype('double'),nstates,maxiters,converge,nrestarts,priors=priors,ncpu=ncpu)
		result.dtype = dtype
		result.ran = [np.nonzero(keep)[0]]
		result.idealize = lambda : self.idealize_hmm(result)
		result.idealize()
		self.model = result
		self.make_report(result)
		self.maven.emit_data_update()

	def run_mlhmm_one(self):
		nstates = self.maven.prefs['modeler.mlhmm.nstates']

		dtype = self.maven.prefs['modeler.dtype']
		success,keep,y = self.get_traces(dtype)
		if not success:
			return
		
		if keep.sum() != 1:
			#Warning "Error: Run all, Apply all","You have more than one molecule turned on")
			return
		
		maxiters, converge, nrestarts, ncpu = self.get_model_specs()

		from .hmm_ml import ml_em_hmm_parallel
		result = ml_em_hmm_parallel(y[0].astype('double'),nstates,maxiters,converge,nrestarts,ncpu)
		result.ran = [np.nonzero(keep)[0]]
		result.dtype = dtype
		result.idealize = lambda : self.idealize_hmm(result)
		result.idealize()
		self.model = result
		self.maven.emit_data_update()

	'''