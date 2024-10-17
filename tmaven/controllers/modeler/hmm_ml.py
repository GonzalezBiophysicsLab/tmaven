#### 1D HMM - EM Max Likelihood

import numpy as np
import numba as nb
from sys import platform
import multiprocessing as mp

from .fxns.numba_math import psi
from .fxns.statistics import p_normal,dirichlet_estep
from .fxns.hmm import forward_backward
from .fxns.initializations import initialize_gmm,initialize_tmatrix


@nb.jit(nb.types.Tuple((nb.float64[:],nb.float64[:],nb.float64[:,:],nb.float64[:],nb.float64[:,:],nb.float64,nb.int64))(nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:,:],nb.int64,nb.float64),nopython=True,cache=True)
def outer_loop(x,mu,var,ppi,tmatrix,maxiters,threshold):
	prob = p_normal(x,mu,var)

	iteration = 0
	ll1 = -np.inf
	ll0 = -np.inf
	while iteration < maxiters:
		## E Step
		# Forward-Backward
		ll0 = ll1
		prob = p_normal(x,mu,var)
		r, xi, ll1 = forward_backward(prob, tmatrix, ppi)

		## likelihood
		if iteration > 1:
			dl = np.abs((ll1 - ll0)/ll0)
			if dl < threshold:
				break

		## M Step
		nk = np.sum(r,axis=0) + 1e-300
		for i in range(nk.size):
			mu[i] = np.sum(r[:,i]*x)/nk[i]
			var[i] = np.sum(r[:,i]*(x - mu[i])**2.)/nk[i]
		ppi = nk/nk.sum()

		for i in range(tmatrix.shape[0]):
			for j in range(tmatrix.shape[1]):
				tmatrix[i,j] = np.mean(xi[:,i,j])
		for i in range(tmatrix.shape[0]):
			tmatrix[i] /= np.sum(tmatrix[i])

		iteration += 1
	return mu,var,r,ppi,tmatrix,ll1,iteration


def ml_em_hmm(x,nstates,maxiters=1000,threshold=1e-6,init_kmeans=False):
	'''
	Convention is NxK
	'''

	if x.ndim != 1:
		raise Exception("Input data isn't 1D")

	# from ml_em_gmm import ml_em_gmm
	# o = ml_em_gmm(x,nstates+1)
	# mu = o.mu[:-1]
	# var = o.var[:-1]
	# ppi = o.ppi[:-1]
	# ppi /= ppi.sum() ## ignore outliers

	mu,var,ppi = initialize_gmm(x,nstates,init_kmeans)
	tmatrix = initialize_tmatrix(nstates)

	#### run calculation
	mu,var,r,ppi,tmatrix,likelihood,iteration = outer_loop(x,mu,var,ppi,tmatrix,maxiters,threshold)

	#### collect results
	from .model_container import model_container

	#getting soft counts
	tmatrix *= len(x) 
	
	out = model_container(type='ml HMM',nstates=nstates,r=r,mean=mu,var=var,
						frac=ppi,tmatrix=tmatrix,likelihood=likelihood,iteration=iteration)

	return out

def ml_em_hmm_parallel(x,nstates,maxiters=1000,threshold=1e-10,nrestarts=1,ncpu=1):
	#
	# if platform != 'win32' and ncpu != 1 and nrestarts != 1:
	# 	pool = mp.Pool(processes = ncpu)
	# 	results = [pool.apply_async(ml_em_hmm, args=(x,nstates,maxiters,threshold)) for i in range(nrestarts)]
	# 	results = [p.get() for p in results]
	# 	pool.close()
	# else:
	# 	results = [ml_em_hmm(x,nstates,maxiters,threshold) for i in range(nrestarts)]
	#
	# try:
	# 	best = np.nanargmax([r.likelihood for r in results])
	# except:
	# 	best = 0
	# return results[best]
	return ml_em_hmm(x,nstates,maxiters,threshold,True)
