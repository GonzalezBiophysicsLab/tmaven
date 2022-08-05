import numpy as np
import numba as nb
from .numba_math import psi,gammaln

@nb.jit(nb.double[:,:](nb.double[:],nb.double[:],nb.double[:]),nopython=True,cache=True)
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

@nb.jit(nb.double[:,:](nb.double[:],nb.double[:],nb.double[:]),nopython=True,cache=True)
def p_normal(x,mu,var):
	return np.exp(ln_p_normal(x,mu,var))


@nb.jit(nb.float64[:,:](nb.float64[:,:]),nopython=True,cache=True)
def dirichlet_estep(alpha):
	E_ln_theta = psi(alpha)
	for i in range(alpha.shape[0]):
		ps = psi(np.sum(alpha[i]))
		for j in range(alpha.shape[1]):
			E_ln_theta[i,j] -= ps
	# E_ln_theta = psi(alpha) - psi(np.sum(alpha,axis=-1))[...,None]
	return E_ln_theta

@nb.jit(nb.float64(nb.float64[:],nb.float64[:]),nopython=True,cache=True)
def dkl_dirichlet(p,q):
	phat = np.sum(p)
	qhat = np.sum(q)

	dkl = gammaln(phat) - gammaln(qhat)
	dkl -= np.sum(gammaln(p) - gammaln(q))
	dkl += np.sum((p-q)*(psi(p)-psi(phat)))
	return dkl

@nb.jit(nb.float64(nb.float64[:,:],nb.float64[:,:]),nopython = True,cache=True)
def dkl_dirichlet_2D(p,q):
	D_kl = 0.
	for i in range(p.shape[0]):
		D_kl += dkl_dirichlet(p[i],q[i])
	return D_kl

@nb.jit(nopython = True,cache=True)
def log_B(log_det_W, nu, D):
    ans = - (nu / 2) * log_det_W - (nu * D / 2) * np.log(2) - (D * (D-1) / 4) * np.log(np.pi) \
        - gammaln(0.5 * nu)
    return ans

@nb.jit(nopython = True)
def dkl_normgamma(p_mu, p_beta, p_a, p_b, q_mu, q_beta, q_a, q_b,cache=True):

    # Copied & translated from JWvdM so will use Normal-Wishart
    p_nu = 2 * p_a
    p_W = 1 / (2 * p_b)
    q_nu = 2 * q_a
    q_W = 1 / (2 * q_b)

    if len(p_mu.shape) == 1:
        D = 1

    E_ln_det_L = np.log(p_W) + np.log(2) + psi(0.5 * p_nu)
    log_det_W_q = np.log(q_W)
    log_det_W_p = np.log(p_W)
    E_Tr_Winv_L = p_W / q_W
    dmWdm = p_W * (p_mu - q_mu)**2

    E_log_NW_w = 0.5 * E_ln_det_L + 0.5 * np.log(p_beta / (2*np.pi)) - 0.5 + log_B(log_det_W_p, p_nu, D) \
        + 0.5 * (p_nu-D-1) * E_ln_det_L - 0.5 * p_nu

    E_log_Norm_u = 0.5 * (np.log(q_beta / (2*np.pi)) + E_ln_det_L - q_beta / p_beta - q_beta * p_nu * dmWdm)

    E_log_Wish_u = log_B(log_det_W_q, q_nu, D) + 0.5 * (q_nu - D - 1) * E_ln_det_L - 0.5 * p_nu * E_Tr_Winv_L

    E_log_NW_u = E_log_Norm_u + E_log_Wish_u

    D_kl = E_log_NW_w - E_log_NW_u

    return np.sum(D_kl)


def kernel_sample(x,nstates):
	from scipy.stats import gaussian_kde
	kernel = gaussian_kde(x)
	m = kernel.resample(nstates).flatten()
	m.sort()
	return m
