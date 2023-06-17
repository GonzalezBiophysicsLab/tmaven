import numpy as np
import logging
logger = logging.getLogger(__name__)

class model_container(object):
	'''
	A container object for the details of any (ensemble) model learnt from sm data.
	A model must contain:
		- type (str): Identifies the specific model used
		- ran (list/None): List of indices of molecules the model was run on or None
		- idealized (np.ndarray/None): representation of the model in dataspace -- shape should probably match maven.data.raw.shape
		- nstates (int): Number of states in the model. For the parent (ensemble) model, this means number of state for the entire ensemble.
		- mean (np.ndarray[1 x K, float64]): The locations of the states
		- var (np.ndarray[1 x K, float64]): The noise of the states
		- frac(np.ndarray[1 x K, float64]): The fraction of time spent in each state
		- tmatrix(np.ndarray[K x K, float64]/None): Transition matrix defining the transition probabilities between the states
		- rates(complicated): Rate matrix defining transition rates between states. Potentially complicated. Come back to this.
		- rate_type(str): Identifies how the rate matrix was generated. Defaults to generating from tmatrix unless specified


	Optional model information (populated depending on 'type'):
		These are all duck-typed and placed under model_spec. Check to make sure that the model type has one of these.

	.idealize is an overwritten function that should update .idealized all by itself (or just not do anything)
	'''
	def __init__(self, type,
				 nstates = None, mean=None, var=None, frac=None,
				 likelihood=None,
				 idealized=None,
				 tmatrix=None, rates=None, dwells =None,
				 ran=[],
				 **kwargs):

		# Book-keeping parameters
		self.type = type
		self.ran = ran

		# Idealized traces
		self.idealized = idealized

		# Non-kinetic parameters
		self.nstates = nstates
		self.mean = mean
		self.var = var
		self.frac = frac

		# Kinetic parameters
		self.tmatrix = tmatrix
		self.rates = rates
		self.dwells = dwells

		#self.check_consistency()

		if not tmatrix is None:
			self.rate_type = "Transition Matrix"
			norm_tmatrix = self.tmatrix.copy()
			for i in range(norm_tmatrix.shape[0]):
				norm_tmatrix[i] /= norm_tmatrix[i].sum()
			self.norm_tmatrix = norm_tmatrix
		else:
			self.rate_type = "N/A"

		# Marginal likelihood/evidence
		self.likelihood = likelihood

		# Model specifics
		self.__dict__.update(kwargs)

		# Book-keeping
		import time
		self.time_made = time.ctime()

	def check_consistency(self):

		K = self.nstates
		if self.mean.shape[0] != K or self.var.shape[0] != K or self.frac.shape[0] != K:
			logger.error("Model has inconsistent number of parameters")

		if not self.tmatrix is None:
			if self.tmatrix.shape[0] != K:
				logger.error("Model has inconsistent number of parameters")

	def description(self):
		mem = hex(id(self))
		try:
			t = '{} ({})'.format(self.type,str(self.nstates))
		except:
			t = self.type

		return '[{}] {} - {}'.format(mem,t,self.time_made)

	def convert_to_dict(self):
		model_dict = self.__dict__

		if 'trace_level' in model_dict.keys():
			trace_models = model_dict['trace_level']
			for i in trace_models.keys():
				trace_models[i] = trace_models[i].convert_to_dict()

			model_dict['trace_level'] = trace_models

		return model_dict

	def convert_from_dict(self,dicty):
		self.__dict__.update(dicty)
		if 'trace_level' in dicty.keys():
			trace_models = dicty['trace_level']
			for i in trace_models.keys():
				trace_model_inst = model_container(type = trace_models[i]['type'])
				trace_model_inst.convert_from_dict(trace_models[i])
				trace_models[i] = trace_model_inst
			self.trace_level = trace_models

	def idealize(self):
		pass

	def add_rates(self):
		pass


class trace_model_container(model_container):
	'''
	Stores a trace level model inherits from the general parent model container.
	Has all attributes of the original model container, but the type is amended
	and a trace_id is added to keep track of the modelled trace.

	It is created by converting the original model container.
	'''
	def __init__(self, parent_model, trace_id):

		self.__dict__.update(parent_model.__dict__)
		self.type = self.type + "_tracelevel"

		self.trace_id = trace_id
