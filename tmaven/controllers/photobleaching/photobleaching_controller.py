import logging
logger = logging.getLogger(__name__)

default_prefs = {
	'photobleach.sum':True,
	'photobleach.entire':True,
	'photobleach.movepre':False,
	'photobleach.maxent':True,
	'photobleach.prior.a':1.,
	'photobleach.prior.b':1.,
	'photobleach.prior.beta':1.,
	'photobleach.prior.mu':1000.,
}

class controller_photobleaching(object):
	''' Handles automatic photobleaching calculations

	Parameters
	----------

	Notes
	-----
	These functions use Bayesian model selection to calculate where the photobleaching point in a time series occurs. Generally, only the 1D case is coded, so it operates on the sum of the colors if `prefs['photobleach.sum'] is True and otherwise works on the last color if False`.

	'''
	def __init__(self,maven):
		self.maven = maven
		self.maven.prefs.add_dictionary(default_prefs)

	def calc_single_photobleach(self,index):
		''' Photobleach calculation for one trace

		Calculates photobleaching point for trace `gui.index`. Uses function `get_point_pbtime` from `tmaven/photobleaching/photobleaching.py`

		Notes
		-----
		prefs['photobleach.sum']
			* True : sum over colors
			* False : only use last color
		prefs['photobleach.entire']
			* True : Use the entire trace
				prefs['photobleach.movepre']
					* True : pre = 0
					* False : pre = current pre
			* False : Only use the already defined area of pre:post

		prefs['photobleach.prior.a'] = alpha of gamma distribution over noise

		prefs['photobleach.prior.b'] = beta of gamma distribution over noise

		prefs['photobleach.prior.mu'] = mu of normal-gamma over signal location

		prefs['photobleach.prior.beta'] = precision of normal over signal location

		'''
		if self.maven.data.nmol == 0 or index >= self.maven.data.nmol:
			return
		if self.maven.prefs['photobleach.sum'] is True:
			qq = self.maven.data.corrected[index].sum(-1)
		else:
			qq = self.maven.data.corrected[index,:,-1]

		if self.maven.prefs['photobleach.entire']:
			if self.maven.prefs['photobleach.movepre']:
				self.maven.data.pre_list[index] = 0
			self.maven.data.post_list[index] = self.maven.data.ntime

		qq = qq[self.maven.data.pre_list[index]:self.maven.data.post_list[index]].astype('double')

		from .photobleaching import get_point_pbtime
		a = self.maven.prefs['photobleach.prior.a']
		b = self.maven.prefs['photobleach.prior.b']
		beta = self.maven.prefs['photobleach.prior.beta']
		mu = self.maven.prefs['photobleach.prior.mu']
		self.maven.data.post_list[index] = get_point_pbtime(qq,a,b,beta,mu) + self.maven.data.pre_list[index]
		logger.info('Photobleach Trajectory %d: old method, t_bleach = %d'%(index,self.maven.data.post_list[index]))
		# self.maven.modeler.clear_hmm()
		# self.maven.data_update.emit()

	def photobleach_sum(self):
		''' Photobleach calculation all of the data

		Calculates photobleaching point for trace `gui.index`. Uses function `pb_ensemble` from `tmaven/photobleaching/photobleaching.py`

		Notes
		-----
		prefs['photobleach.sum']
			* True : sum over colors
			* False : only use last color
		'''
		logger.info('Running Photobleaching. JIT compiling functions')
		## def photobleach_step...
		from .photobleaching import pb_ensemble
		if self.maven.prefs['photobleach.sum']:
			qq = self.maven.data.corrected.sum(2) ## sum of all...
		else:
			qq = self.maven.data.corrected[:,:,-1] ## otherwise use 'red'
		keep = self.maven.modeler.get_traces()

		a = self.maven.prefs['photobleach.prior.a']
		b = self.maven.prefs['photobleach.prior.b']
		beta = self.maven.prefs['photobleach.prior.beta']
		mu = self.maven.prefs['photobleach.prior.mu']
		self.maven.data.post_list[keep] = pb_ensemble(qq[keep],a,b,beta,mu)[1]
		self.maven.emit_data_update()
