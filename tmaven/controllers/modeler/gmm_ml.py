#### 1D GMM - EM Max Likelihood
import numpy as np
import numba as nb
from .fxns.statistics import p_normal
from .fxns.initializations import initialize_gmm

@nb.jit(nb.types.Tuple((nb.float64[:],nb.float64[:],nb.float64[:,:],nb.float64[:],nb.float64,nb.int64))(nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.int64,nb.float64),nopython=True)
def outer_loop(x,mu,var,ppi,maxiters,threshold):
	prob = p_normal(x,mu,var)
	r = np.zeros_like(prob)

	p_unif = 1./(x.max()-x.min()) ## Assume data limits define the range
	mu[-1] = 0. ## set outlier class
	var[-1] = 1. ## set outlier class

	iteration = 0
	ll1 = -np.inf
	ll0 = -np.inf

	while iteration < maxiters:
		# E Step
		prob = p_normal(x,mu,var)
		prob[:,-1] = p_unif ## Last state is uniform for outlier detection

		ll0 = ll1
		ll1 = 0
		for i in range(prob.shape[0]):
			ll1i = 0
			for j in range(prob.shape[1]):
				ll1i += ppi[j]*prob[i,j]
			ll1 += np.log(ll1i)

		for i in range(r.shape[0]):
			r[i] = ppi*prob[i]
			r[i] /= np.sum(r[i])

		## likelihood
		if iteration > 1:
			dl = np.abs((ll1 - ll0)/ll0)
			if dl < threshold or np.isnan(ll1):
				break

		## M Step
		nk = np.sum(r,axis=0) + 1e-300
		for i in range(nk.size-1): ## ignore the outlier class
		# for i in range(nk.size):
			mu[i] = 0.
			for j in range(r.shape[0]):
				mu[i] += r[j,i]*x[j]
			mu[i] /= nk[i]

			var[i] = 0.
			for j in range(r.shape[0]):
				var[i] += r[j,i]*(x[j] - mu[i])**2.
			var[i] /= nk[i]
		mu[-1] = p_unif ## info for outlier class
		var[-1] = 10**300. ## info for outlier class
		ppi = nk/np.sum(nk)

		iteration += 1
	return mu,var,r,ppi,ll1,iteration


def ml_em_gmm(x,nstates,maxiters=1000,threshold=1e-6,init_kmeans=True):
	'''
	Data convention is NxK

	* Outlier Detection
		Has outlier detection, where the last state is a uniform distribution over the data limits
		mu and var of this state don't mean anything
	'''

	if x.ndim != 1:
		raise Exception("Input data isn't 1D")

	mu,var,ppi = initialize_gmm(x,nstates,init_kmeans)

	#### run calculations
	mu,var,r,ppi,likelihood,iteration = outer_loop(x,mu,var,ppi,maxiters,threshold)

	#### Collect results
	from .model_container import model_container
	out = model_container(type='ml GMM',
						  nstates=nstates,mean=mu,var=var,frac=ppi,
						  r=r,likelihood=likelihood,iteration=iteration)
	# out.idealized = out.r.argmax(-1)
	return out

def ml_em_gmm_parallel(x,nstates,maxiters=1000,threshold=1e-6,init_kmeans=True,nrestarts=1,ncpu=1):
	return ml_em_gmm(x,nstates,maxiters,threshold,init_kmeans)
