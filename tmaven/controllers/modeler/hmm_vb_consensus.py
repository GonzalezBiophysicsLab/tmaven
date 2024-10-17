#### 1D HMM - EM Max Likelihood

import numpy as np
import numba as nb
from sys import platform
import multiprocessing as mp

from .fxns.statistics import p_normal,dkl_dirichlet
from .fxns.numba_math import psi,gammaln
from .fxns.initializations import initialize_gmm, initialize_tmatrix
from .fxns.hmm import forward_backward, viterbi

@nb.jit(nb.types.Tuple((nb.float64[:],nb.float64[:],nb.float64[:]))(nb.float64[:],nb.float64[:,:]),nopython=True,cache=True)
def m_sufficient_statistics(x,r):
	#### M Step
	## Sufficient Statistics
	xbark = np.zeros(r.shape[1])
	sk = np.zeros_like(xbark)

	nk = np.sum(r,axis=0) + 1e-10
	for i in range(nk.size):
		xbark[i] = 0.
		for j in range(r.shape[0]):
			xbark[i] += r[j,i]*x[j]
		xbark[i] /= nk[i]

		sk[i] = 0.
		for j in range(r.shape[0]):
			sk[i] += r[j,i]*(x[j] - xbark[i])**2.
		sk[i] /= nk[i]
	return nk,xbark,sk

@nb.jit(nb.types.Tuple((nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:]))(nb.float64[:],nb.float64[:,:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:]),nopython=True,cache=True)
def m_updates(x,r,a0,b0,m0,beta0):
	#### M Step
	## Updates
	nk,xbark,sk = m_sufficient_statistics(x,r)

	beta = np.zeros_like(m0)
	m = np.zeros_like(m0)
	a = np.zeros_like(m0)
	b = np.zeros_like(m0)

	## Hyperparameters
	for i in range(nk.size):
		beta[i] = beta0[i] + nk[i]
		m[i] = 1./beta[i] *(beta0[i]*m0[i] + nk[i]*xbark[i])
		a[i] = a0[i] + nk[i]/2.
		b[i] = b0[i] + .5*(nk[i]*sk[i] + beta0[i]*nk[i]/(beta0[i]+nk[i])*(xbark[i]-m0[i])**2.)

	return a,b,m,beta,nk,xbark,sk

@nb.jit(nb.float64[:](nb.float64[:,:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:,:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:,:],nb.float64),nopython=True,cache=True)
def calc_lowerbound(r,a,b,m,beta,pik,tm,nk,xbark,sk,E_lnlam,E_lnpi,a0,b0,m0,beta0,pi0,tm0,lnz):

	lt74 = 0.
	lt77 = 0.
	## Precompute
	lnb0 = a0*np.log(b0) - gammaln(a0)
	lnbk = a*np.log(b) - gammaln(a)
	hk = -lnbk -(a-1.)*E_lnlam + a
	for i in range(m.shape[0]):

		## Normal Wishart
		lt74 += .5*(np.log(beta0[i]/2./np.pi) + E_lnlam[i] - beta0[i]/beta[i] - beta0[i]*a[i]/b[i]*(m[i]-m0[i])**2.)
		lt74 += lnb0[i] + (a0[i]-1.)*E_lnlam[i] - a[i]*b0[i]/b[i]
		lt77 += .5*E_lnlam[i] + .5*np.log(beta[i]/2.*np.pi) - .5 - hk[i]
		Fgw = lt74 - lt77

	## Starting point
	Fpi = - dkl_dirichlet(pik,pi0)

	## Transition matrix
	Ftm = 0.
	for i in range(tm.shape[0]):
		Ftm -= dkl_dirichlet(tm[i],tm0[i])

	ll1 = lnz + Fgw + Fpi + Ftm
	return np.array((ll1,lnz,Fgw,Fpi,Ftm))

@nb.jit(nb.types.Tuple((nb.float64[:,:],nb.float64[:,:,:],nb.float64,nb.float64[:,:],nb.float64[:],nb.float64[:]))(nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:,:]),nopython=True,cache=True)
def individual_e_step(x,a,b,beta,m,pik,tm):
	E_lntm = np.zeros_like(tm)
	E_dld = np.zeros((x.size,m.size))
	E_lnlam = np.zeros_like(m)
	E_lnpi = np.zeros_like(m)

	for i in range(E_dld.shape[0]):
		E_dld[i] = 1./beta + a/b*(x[i]-m)**2.
	E_lnlam = psi(a)-np.log(b)
	E_lnpi = psi(pik)-psi(np.sum(pik))
	for i in range(E_lntm.shape[0]):
		E_lntm[i] = psi(tm[i])-psi(np.sum(tm[i]))

	prob = np.zeros((x.size,m.size))
	for i in range(prob.shape[1]):
		prob[:,i] = (2.*np.pi)**-.5 * np.exp(-.5*(E_dld[:,i] - E_lnlam[i]))
	r, xi, lnz = forward_backward(prob, np.exp(E_lntm), np.exp(E_lnpi))

	return r,xi,lnz,E_lntm,E_lnlam,E_lnpi


@nb.jit(nb.types.Tuple(
	(nb.float64[:,:],
	 nb.float64[:],
	 nb.float64[:],
	 nb.float64[:],
	 nb.float64[:],
	 nb.float64[:],
	 nb.float64[:,:],
	 nb.float64[:],
	 nb.float64[:],
	 nb.float64[:,:],
	 nb.float64[:,:],
	 nb.int64))
	 (nb.int64[:],
	 nb.float64[:],
	 nb.float64[:],
	 nb.float64[:],
	 nb.float64[:,:],
	 nb.int64,
	 nb.float64,
	 nb.types.Tuple((nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:,:]))),
	 nopython=True,
	 cache=True)
def outer_loop(xind,xdata,mu,var,tm,maxiters,threshold,priors):

	N = xind[-1]+1
	K = mu.size

	## priors
	m0 = priors[0]
	beta0 = priors[1]
	a0 = priors[2]
	b0 = priors[3]
	pi0 = priors[4]
	tm0 = priors[5]

	# initialize
	prob = p_normal(xdata,mu,var)
	T,K = prob.shape
	r = np.zeros((T,K))
	for i in range(r.shape[0]):
		r[i] = prob[i]
		r[i] /= np.sum(r[i]) + 1e-10 ## for stability

	a,b,_,beta,nk,xbark,sk = m_updates(xdata,r,a0,b0,m0,beta0)
	m = m0 + np.random.normal(K)/np.sqrt(beta) ## zuzsh it up a lot
	m = np.sort(m)
	prob = p_normal(xdata,m,b/a)
	for i in range(r.shape[0]):
		r[i] = prob[i]
		r[i] /= np.sum(r[i]) + 1e-10 ## for stability
	a,b,_,beta,nk,xbark,sk = m_updates(xdata,r,a,b,m,beta)

	pik = np.sqrt(nk+1)
	tm = tm0.copy()
	paths = viterbi(xdata,m,b/a,tm,pik).astype('int')
	for k in range(K):
		keep = paths == k
		if keep.sum() > 0:
			pik[k] = float(keep.sum())/float(xdata.size)*np.sqrt(float(N))
			for kk in range(K):
				if k != kk:
					tm[k,kk] = np.sum(np.bitwise_and(paths[1:]==kk,paths[:-1]==k)) + 1
			tm[k] /= np.sum(tm[k])*10.
			tm[k,k] = 1. - np.sum(tm[k])
			tm *= np.sqrt(float(N))
	# print(m,np.sqrt(b/a),pik,tm.flatten())

	iteration = 0
	ll1 = -np.inf
	ll0 = -np.inf
	ll = np.zeros((maxiters,5))

	while iteration < maxiters:
		# E Step

		r = np.zeros((T,K))

		xi_sum = np.zeros((K,K))
		pi_sum = np.zeros((K))

		ll0 = ll1

		for Ni in range(N):
			ind = xind==Ni
			trace = xdata[ind]
			ri,xii,lnzi,E_lntmi,E_lnlami,E_lnpii = individual_e_step(trace,a,b,beta,m,pik,tm)
			ri /= ri.sum(1)[:,None]
			r[ind] = ri

			pi_sum += ri[0]
			xi_sum += xii.sum(0)

			## This isn't quite right... multiple-counting the priors.... err like separate priors for everything
			lli = calc_lowerbound(ri,a,b,m,beta,pik,tm,nk,xbark,sk,E_lnlami,E_lnpii,a0,b0,m0,beta0,pi0,tm0,lnzi)
			ll[iteration] += lli
		ll1 = ll[iteration,0]

		## likelihood
		if iteration > 1:
			dl = (ll1 - ll0)/np.abs(ll0)
			if dl < threshold or np.isnan(ll1):
				break

		a,b,m,beta,nk,xbark,sk = m_updates(xdata,r,a0,b0,m0,beta0)

		### indivudal - copy this from simulataneous
		pik = pi0 + pi_sum
		tm = tm0.copy()
		tm  += xi_sum

		if iteration < maxiters:
			iteration += 1

	_,_,_,E_lntm,E_lnlam,E_lnpi = individual_e_step(xdata,a,b,beta,m,pik,tm)
	return r,a,b,m,beta,pik,tm,E_lnlam,E_lnpi,E_lntm,ll,iteration


def consensus_vb_em_hmm(x,nstates,maxiters=1000,threshold=1e-10,nrestarts=1,priors=None,init_kmeans=False,mu_mode=False):
	'''
	Data convention is NxTxK
	'''

	if x[0].ndim != 1:
		raise Exception("Input data isn't 1D")


	# from ml_em_gmm import ml_em_gmm
	# o = ml_em_gmm(x,nstates+1)
	# mu = o.mu[:-1]
	# var = o.var[:-1]
	# ppi = o.ppi[:-1]
	# ppi /= ppi.sum() ## ignore outliers
	xind = np.concatenate([np.zeros(x[i].size,dtype='int64')+i for i in range(len(x))])
	xdata = np.concatenate(x)

	# mu,var,ppi = initialize_gmm(xdata,nstates,init_kmeans)
	tmatrix = initialize_tmatrix(nstates)

	## Priors - beta, a, b, pi, alpha... mu is from GMM
	# if priors is None:
	# 	beta_prior = np.ones_like(mu) *0.25
	# 	a_prior = np.ones_like(mu) *2.5
	# 	b_prior = np.ones_like(mu)*0.01
	# 	pi_prior = np.ones_like(mu)
	# 	tm_prior = np.ones_like(tmatrix)
	# 	mu_prior = mu

	# 	priors = (mu_prior,beta_prior, a_prior, b_prior, pi_prior, tm_prior)

	if priors is None:
		beta_prior = np.ones(nstates) *0.25
		a_prior = np.ones(nstates) *2.5
		b_prior = np.ones(nstates)*0.01
		pi_prior = np.ones(nstates)
		tm_prior = np.ones(nstates,nstates)
		mu_prior = np.percentile(xdata,np.linspace(0,100,nstates+2))[1:-1]
	else:
		mu_prior = np.percentile(xdata,np.linspace(0,100,nstates+2))[1:-1]
		beta_prior = priors[1]
		a_prior = priors[2]
		b_prior = priors[3]
		pi_prior = priors[4]
		tm_prior = priors[5]

	priors = (mu_prior,beta_prior, a_prior, b_prior, pi_prior, tm_prior)


	#### run calculation
	res = []
	ll_restarts = []

	for nr in range(nrestarts):
		kmu = mu_prior
		kvar = b_prior/a_prior
		r,a,b,mu,beta,pi,tmatrix,E_lnlam,E_lnpi,E_lntm,likelihood,iteration = outer_loop(xind,xdata,kmu,kvar,tmatrix,maxiters,threshold,priors)
		likelihood = likelihood[:iteration+1]
		ll_restarts.append(likelihood[-1,0])
		res.append([r,a,b,mu,beta,pi,tmatrix,E_lnlam,E_lnpi,E_lntm,likelihood,iteration])		
		# print(nr,likelihood[-1,0],mu)#,np.sqrt(b/a),pi,tmatrix.flatten())
	ll_restarts = np.array(ll_restarts)

	try:
		best = np.nanargmax(ll_restarts)

	except:
		import logging
		logger = logging.getLogger(__name__)
		logger.info("vb Consensus HMM restarts all returned NaN ELBOs, Defaulting to first restart")

		best = 0

	#print(ll_restarts, best, ll_restarts[best])
	best_res = res[best]
	r,a,b,mu,beta,pi,tmatrix,E_lnlam,E_lnpi,E_lntm,likelihood,iteration = best_res

	#### collect results
	from .model_container import model_container
	ppi = np.sum(r,axis=0)
	ppi /= ppi.sum()
	var = 1./np.exp(E_lnlam)

	new_r = []
	for i in range(xind[-1]+1):
		ind = xind==i
		new_r.append(r[ind])

	priors = {'mu_prior':mu_prior,
			  'beta_prior':beta_prior,
			  'a_prior':a_prior,
			  'b_prior':b_prior,
			  'pi_prior':pi_prior,
			  'tm_prior':tm_prior}


	out = model_container(type='vb Consensus HMM',
						  nstates = nstates,mean=mu,var=var,frac=ppi,
						  tmatrix=tmatrix,
						  likelihood=likelihood,
						  iteration=iteration, r=new_r,a=a,b=b,beta=beta, pi=pi,
						  E_lnlam=E_lnlam,E_lnpi=E_lnpi,E_lntm=E_lntm,
						  priors=priors)
	return out

def consensus_vb_em_hmm_parallel(x,nstates,maxiters=1000,threshold=1e-10,nrestarts=1,priors=None,ncpu=1,init_kmeans=False):
	#
	# if platform != 'win32' and ncpu != 1 and nrestarts != 1:
	# 	pool = mp.Pool(processes = ncpu)
	# 	results = [pool.apply_async(consensus_vb_em_hmm, args=(x,nstates,maxiters,threshold,prior_strengths,True)) for i in range(nrestarts)]
	# 	results = [p.get() for p in results]
	# 	pool.close()
	# else:
	# 	results = [consensus_vb_em_hmm(x,nstates,maxiters,threshold,prior_strengths,True) for i in range(nrestarts)]
	#
	# try:
	# 	best = np.nanargmax([r.likelihood[-1,0] for r in results])
	# except:
	# 	best = 0
	# return results[best]
	return consensus_vb_em_hmm(x,nstates,maxiters,threshold,nrestarts,priors,init_kmeans)
