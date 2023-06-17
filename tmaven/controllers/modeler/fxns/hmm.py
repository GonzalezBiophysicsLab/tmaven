import numpy as np
import numba as nb
from .statistics import p_normal

@nb.jit(nb.types.Tuple((nb.double[:,:],nb.double[:,:,:],nb.double))(nb.double[:,:],nb.double[:,:],nb.double[:]),nopython=True,cache=True)
def forward_backward(p_x_z,A,pi):
	### Copied and Translated from JWvdM's ebFRET mex file

	T,K = p_x_z.shape
	a = np.zeros((T,K),dtype=nb.double)
	b = np.zeros((T,K),dtype=nb.double)
	c = np.zeros((T),dtype=nb.double)

	g = np.zeros((T,K),dtype=nb.double)
	xi = np.zeros((T-1,K,K),dtype=nb.double)
	ln_z = 0.

	# Forward Sweep
	for k in range(K):
		a[0,k] = pi[k] * p_x_z[0,k]
		c[0] += a[0,k]

	# normalize a(0,k) by c(k)
	for k in range(K):
		a[0,k] /= c[0] + 1e-300

	for t in range(1,T):
		# a(t, k)  =  sum_l p_x_z(t,k) A(l, k) alpha(t-1, l)
		for k in range(K):
			for l in range(K):
				a[t,k] += p_x_z[t,k] * A[l,k] * a[t-1,l]
			c[t] += a[t,k]

		# normalize a(t,k) by c(t)
		for k in range(K):
			a[t,k] /= c[t] + 1e-300

	# Back sweep - calculate
	for k in range(K):
		b[T-1,k] = 1.

	# t = T-2:0
	for tt in range(T-1):
		t = T - 2 - tt
		# b(t, k)  =  sum_l p_x_z(t+1,l) A(k, l) beta(t+1, l)
		for k in range(K):
			for l in range(K):
				b[t,k] += p_x_z[t+1,l] * A[k,l] * b[t+1,l]
			# normalize b(t,k) by c(t+1)
			b[t,k] /= c[t+1] + 1e-300

	# g(t,k) = a(t,k) * b(t,k)
	for k in range(K):
		for t in range(T):
			g[t,k] = a[t,k]*b[t,k]

	# xi(t, k, l) = alpha(t, k) A(k,l) p_x_z(t+1, l) beta(t+1, l) / c(t+1)
	for t in range(T-1):
		for k in range(K):
			for l in range(K):
				xi[t,k,l] = (a[t,k] * A[k,l] * p_x_z[t+1,l] * b[t+1,l]) / (c[t+1] + 1e-300)

	# ln_Z = sum_t log(c[t])
	for t in range(T):
		ln_z += np.log(c[t]+1e-300)

	return g,xi,ln_z

@nb.jit("int64[:](float64[:,:],int64[:,:])",nopython=True,cache=True)
def _vit_calc_zhat(omega,zmax):
	zhat = np.empty(omega.shape[0],dtype=nb.int64)
	zhat[-1] = np.argmax(omega[-1])
	# for t in range(zhat.shape[0])[::-1][1:]:
	n = zhat.size
	for tt in range(n-1):
		t = n-tt-2
		zhat[t] = zmax[t+1,zhat[t+1]]
	return zhat

@nb.jit(nb.types.Tuple((nb.float64[:,:],nb.int64[:,:]))(nb.float64[:,:],nb.float64[:,:],nb.float64[:]),nopython=True,cache=True)
def _vit_calc_omega(ln_p_x_z,ln_A,ln_ppi):
	omega = np.empty_like(ln_p_x_z)
	zmax = np.empty_like(ln_p_x_z,dtype=nb.int64)
	omega[0] = ln_ppi + ln_p_x_z[0]
	for t in range(1,omega.shape[0]):
		for i in range(ln_A.shape[0]):
			omega[t,i] = ln_p_x_z[t,i] + np.max(ln_A[:,i] + omega[t-1])
			zmax[t,i] = np.argmax(ln_A[:,i] + omega[t-1])
		# omega[t] = ln_p_x_z[t] + np.max(ln_A + omega[t-1][:,None],axis=0)
		# zmax[t] = np.argmax(ln_A + omega[t-1][:,None],axis=0)
	return omega,zmax

@nb.jit(nb.int64[:](nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:,:],nb.float64[:]),nopython=True,cache=True)
def viterbi(x,mu,var,tmatrix,ppi):
	### v is vbhmm_result
	tm = np.zeros_like(tmatrix)
	for i in range(tmatrix.shape[0]):
		tm[i] = tmatrix[i] / np.sum(tmatrix[i])

	ln_p_x_z = np.log(p_normal(x,mu,var))
	ln_A = np.log(tm)
	ln_ppi = np.log(ppi)

	omega,zmax = _vit_calc_omega(ln_p_x_z,ln_A,ln_ppi)
	zhat = _vit_calc_zhat(omega,zmax)

	return zhat
