import numpy as np
import numba as nb
from math import lgamma as gammaln
import logging
logging.getLogger("numba").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
import os

_windows = False
if os.name == 'nt':
	_windows = True

@nb.njit(nb.double(nb.double[:]),cache=True)
def sufficient_xbar(x):
	n = x.size

	# xbar
	xbar = 0.
	for i in range(n):
		xbar += x[i]
	xbar /= float(n)
	return xbar

@nb.njit(nb.double(nb.double[:],nb.double),cache=True)
def sufficient_s2(x,m):
	n = x.size

	#s2
	s2 = 0.
	for i in range(n):
		d = x[i]-m
		s2 += d*d
	return s2

################################################################################
### \mathcal{N} with an unknown \mu and an unknown \sigma
################################################################################
@nb.njit(nb.types.Tuple((nb.double,nb.double,nb.double,nb.double))(nb.double[:],nb.double,nb.double,nb.double,nb.double),cache=True)
def normal_update(x,a0,b0,k0,m0):
	# sufficient statistics
	xbar = sufficient_xbar(x)
	s2 = sufficient_s2(x,xbar)

	# posterior parameters
	n = x.size
	kn = k0 + n
	an = a0 + n/2.
	bn = b0 + .5*s2 + k0*n*(xbar - m0)**2. / (2.*(k0+n))
	mn = (k0*m0 + n*xbar)/kn
	return an,bn,kn,mn

@nb.njit([nb.double(nb.double[:],nb.double,nb.double,nb.double,nb.double)],cache=True)
def normal_ln_evidence(x,a0,b0,k0,m0):

	n = x.size
	an,bn,kn,mn = normal_update(x,a0,b0,k0,m0)
	ln_evidence = gammaln(an) - gammaln(a0) + a0*np.log(b0) - an*np.log(bn) +.5*np.log(k0) - .5*np.log(kn) - n/2. * np.log(2.*np.pi)

	return ln_evidence

################################################################################
### \mathcal{N} with an known \mu and an unknown \sigma
################################################################################
@nb.njit(nb.types.Tuple((nb.double,nb.double))(nb.double[:],nb.double,nb.double,nb.double),cache=True)
def normal_mu_update(x,mu,a0,b0):
	# sufficient statistics
	# s2 = np.sum((x-mu)**2.)
	s2 = sufficient_s2(x,mu)

	# posterior parameters
	n = x.size
	an = a0 + n/2.
	bn = b0 + .5*s2
	return an,bn

@nb.njit(nb.double(nb.double[:],nb.double,nb.double,nb.double),cache=True)
def normal_mu_ln_evidence(x,mu,a0,b0):
	n = x.size

	an,bn = normal_mu_update(x,mu,a0,b0)
	ln_evidence = gammaln(an) - gammaln(a0) + a0*np.log(b0) - an*np.log(bn) - n/2. * np.log(2.*np.pi)

	return ln_evidence

################################################################################
### Photobleaching model - start w/ N(\mu) go to N(0) at time t
################################################################################

@nb.njit(nb.double[:](nb.double[:],nb.double,nb.double,nb.double,nb.double),cache=True)
def ln_likelihood(d,a0,b0,k0,m0):
	lnl = np.zeros_like(d)

	lnl[0] = normal_mu_ln_evidence(d,0.,a0,b0)

	for i in range(1,d.shape[0]-1):
		lnl[i] = normal_ln_evidence(d[:i],a0,b0,k0,m0) + normal_mu_ln_evidence(d[i:],0.,a0,b0)
	lnl[-1] = normal_ln_evidence(d,a0,b0,k0,m0)
	return lnl

@nb.njit(nb.double(nb.double[:],nb.double,nb.double,nb.double,nb.double),cache=True)
def ln_evidence(d,a0,b0,k0,m0):

	lnl  = ln_likelihood(d,a0,b0,k0,m0)
	# uniform priors for t
	lmax = lnl.max()
	ev = np.log(np.sum(np.exp(lnl-lmax)))+lmax
	return ev

@nb.njit(nb.double(nb.double[:],nb.double,nb.double,nb.double,nb.double),cache=True)
def ln_bayes_factor(d,a0,b0,k0,m0):
	# a0 = 1.
	# b0 = 1.
	# k0 = 1.
	# m0 = 1000.

	return ln_evidence(d,a0,b0,k0,m0) - normal_ln_evidence(d,a0,b0,k0,m0)

@nb.njit(nb.double[:](nb.double[:],nb.double,nb.double,nb.double,nb.double,nb.double),cache=True)
def posterior(d,k,a0,b0,k0,m0):
	t = np.arange(d.size)
	lnp = ln_likelihood(d,a0,b0,k0,m0) + np.log(k) - k*t
	return lnp


################################################################################
### Single Step Model
################################################################################

@nb.njit(nb.int64(nb.double[:],nb.double,nb.double,nb.double,nb.double),cache=True)
def get_point_pbtime(d,a0,b0,k0,m0):
	lnl = ln_likelihood(d,a0,b0,k0,m0)
	for i in range(lnl.shape[0]):
		if np.isnan(lnl[i]):
			lnl[i] = -np.inf
	# pbt = lnl.argmax()
	pbt = np.argmax(lnl)
	if pbt == d.shape[0]-1:
		pbt = d.shape[0]
	return pbt

@nb.njit(nb.double(nb.double[:],nb.double,nb.double,nb.double,nb.double),cache=True)
def get_expectation_pbtime(d,a0,b0,k0,m0):
	# a0 = 1.
	# b0 = 1.
	# k0 = 1.
	# m0 = 1000.
	# # a0 = 2.5
	# # b0 = .01
	# # k0 = .25
	# # m0 = .5
	lnl = ln_likelihood(d,a0,b0,k0,m0)
	t = np.arange(lnl.size)
	lmax = np.max(lnl)
	p = np.exp(lnl-lmax)
	psum = np.sum(p)
	pbt = np.sum(p*t)/psum
	return pbt

if _windows:
	@nb.njit(nb.types.Tuple((nb.double,nb.int64[:]))(nb.double[:,:],nb.double,nb.double,nb.double,nb.double),cache=True)
	def pb_ensemble(d,a0,b0,k0,m0):
		'''
		Input:
			* `d` is a np.ndarray of shape (N,T) of the input data
		Output:
			* `e_k` is the expectation of the photobleaching time rate constant
			* `pbt` is a np.ndarray of shape (N) with the photobleaching time
		'''
		# a0 = 1.
		# b0 = 1.
		# k0 = 1.
		# m0 = 1000.
		pbt = np.zeros(d.shape[0],dtype=nb.int64)
		for i in range(d.shape[0]):
			pbt[i] = get_expectation_pbtime(d[i],a0,b0,k0,m0)
		e_k = (1.+pbt.size)/(1.+np.sum(pbt))
		for i in range(d.shape[0]):
			pbt[i] = np.argmax(posterior(d[i],e_k,a0,b0,k0,m0))
			if pbt[i] == d.shape[1]-1:
				pbt[i] = d.shape[1]
		return e_k,pbt

else:
	@nb.njit(nb.types.Tuple((nb.double,nb.int64[:]))(nb.double[:,:],nb.double,nb.double,nb.double,nb.double),parallel=True,cache=True)
	def pb_ensemble(d,a0,b0,k0,m0):
		'''
		Input:
			* `d` is a np.ndarray of shape (N,T) of the input data
		Output:
			* `e_k` is the expectation of the photobleaching time rate constant
			* `pbt` is a np.ndarray of shape (N) with the photobleaching time
		'''
		# a0 = 1.
		# b0 = 1.
		# k0 = 1.
		# m0 = 1000.

		pbt = np.zeros(d.shape[0],dtype=nb.int64)
		for i in nb.prange(d.shape[0]):
			pbt[i] = get_expectation_pbtime(d[i],a0,b0,k0,m0)
		e_k = (1.+pbt.size)/(1.+np.sum(pbt))
		for i in nb.prange(d.shape[0]):
			pbt[i] = np.argmax(posterior(d[i],e_k,a0,b0,k0,m0))
			if pbt[i] == d.shape[1]-1:
				pbt[i] = d.shape[1]
		return e_k,pbt

# @nb.njit(nb.double[:](nb.double[:,:]),cache=True)
# def pb_snr(d,a0,b0,k0,m0):
# 	pbt = pb_ensemble(d,a0,b0,k0,m0)[1]
# 	snrr = np.zeros(d.shape[0],dtype=nb.double)
# 	for i in range(snrr.size):
# 		if pbt[i] > 5:
# 			if pbt[i] < d.shape[1] - 5:
# 				snrr[i] = (np.mean(d[i,:pbt[i]]) - np.mean(d[i,pbt[i]:])) / np.std(d[i,:pbt[i]])
# 			else:
# 				snrr[i] = np.mean(d[i])/np.std(d[i])
# 		else:
# 			snrr[i] = 0.
# 	return snrr
#
# ################################################################################
# ## Check to see if a signal is zero
# ################################################################################
# @nb.njit(nb.double[:](nb.double[:,:]),cache=True)
# def model_comparison_signal(x):
# 	out = np.zeros(x.shape[0],dtype=nb.double)
# 	a0 = 1.
# 	b0 = 1.
# 	k0 = 1.
# 	m0 = 1000.
# 	for i in range(out.size):
# 		lnp_m2 = normal_mu_ln_evidence(x[i],0.,a0,b0)
# 		lnp_m1 = normal_ln_evidence(x[i],a0,b0,k0,m0)
# 		p = 1./(1.+np.exp(lnp_m2-lnp_m1))
# 		out[i] = p
# 	return out
