import numpy as np
import numba as nb
from .numba_math import psi,gammaln

@nb.jit(nb.double[:,:](nb.double[:],nb.double[:],nb.double[:]),nopython=True)
def ln_p_normal(x,mu,var):
	out = np.zeros((x.size,mu.size))
	for j in range(mu.size):
		if var[j] > 0:
			for i in range(x.size):
				out[i,j] = -.5*np.log(2.*np.pi) -.5*np.log(var[j]) - .5/var[j]*(x[i]-mu[j])**2.
		else:
			for i in range(x.size):
				out[i,j] = -np.inf
	return out

@nb.jit(nb.double[:,:](nb.double[:],nb.double[:],nb.double[:]),nopython=True)
def p_normal(x,mu,var):
	return np.exp(ln_p_normal(x,mu,var))


@nb.jit(nb.float64[:,:](nb.float64[:,:]),nopython=True)
def dirichlet_estep(alpha):
	E_ln_theta = psi(alpha)
	for i in range(alpha.shape[0]):
		ps = psi(np.sum(alpha[i]))
		for j in range(alpha.shape[1]):
			E_ln_theta[i,j] -= ps
	# E_ln_theta = psi(alpha) - psi(np.sum(alpha,axis=-1))[...,None]
	return E_ln_theta

@nb.jit(nb.float64(nb.float64[:],nb.float64[:]),nopython=True)
def dkl_dirichlet(p,q):
	phat = np.sum(p)
	qhat = np.sum(q)

	dkl = gammaln(phat) - gammaln(qhat)
	dkl -= np.sum(gammaln(p) - gammaln(q))
	dkl += np.sum((p-q)*(psi(p)-psi(phat)))
	return dkl

def kernel_sample(x,nstates):
	from scipy.stats import gaussian_kde
	kernel = gaussian_kde(x)
	m = kernel.resample(nstates).flatten()
	m.sort()
	return m
