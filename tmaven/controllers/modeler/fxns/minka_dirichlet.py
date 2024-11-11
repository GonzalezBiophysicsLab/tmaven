import numba as nb
import numpy as np
from .numba_math import gammaln, psi, trigamma
# from numba_math import gammaln, psi, trigamma

@nb.njit(cache=True)
def fixedpoint2(nik,alpha,maxiters,threshold,eps):
	N,K= nik.shape
	lnL = np.zeros(maxiters)
	ni = np.sum(nik+eps,axis=1)

	for it in range(maxiters):
		## Minka 65
		asum = np.sum(alpha+eps)
		lnL[it] = N*gammaln(asum)
		lnL[it] -= N*np.sum(gammaln(alpha+eps))
		lnL[it] -= np.sum(gammaln(ni+asum))
		num = np.zeros(K)
		denom = 0.
		for i in range(N):
			denom += (ni[i]+eps) / (ni[i]+eps-1.+asum)
			for k in range(K):
				lnL[it] += gammaln(nik[i,k]+eps+alpha[k])
				num[k] += (nik[i,k]+eps)/(nik[i,k]+eps-1.+alpha[k])

		# print(it,lnL[it],asum)
		if (it > 2):
			if np.abs(lnL[it]-lnL[it-1])/np.abs(lnL[it-1]) < threshold:
				break
		alpha = alpha*num/denom

	return lnL[:it+1],alpha

@nb.njit(cache=True)
def initialize_alpha(nik):
	I,K = nik.shape

	ni = nik.sum(1)
	pik = np.zeros((I,K))
	for k in range(K):
		sk = np.sum(1.+nik[:,k])
		for i in range(I):
			pik[i,k] += (1.+nik[i,k])/sk
	
	E_pk = np.zeros(K)
	for k in range(K):
		E_pk[k] = np.mean(pik[:,k])
	
	pi0 = pik[:,0]
	E_pi01 = np.mean(pi0)
	E_pi02 = np.mean(pi0*pi0)
	asum = (E_pi01-E_pi02)/(E_pi02-E_pi01*E_pi01)
	alpha = E_pk * asum

	return alpha

@nb.njit(cache=True)
def estimate_dirichlet(nik,maxiter=100,threshold=1e-10,eps=1e-16):
	N,K = nik.shape
	# alpha = initialize_alpha(nik)
	alpha = np.ones(K)
	l,alpha = fixedpoint2(nik,alpha,int(maxiter),threshold,eps)
	alpha[np.isnan(alpha)] = 0
	return alpha


if __name__ == '__main__':
	import matplotlib.pyplot as plt

	N = 1000
	K = 250
	amax = 50
	np.random.seed(666)
	alpha0 = np.random.rand(K)*amax
	p0 = np.random.dirichlet(alpha0,size=N)
	nik = np.array([np.random.multinomial(np.random.randint(low=100,high=1000),p0i,size=1).flatten() for p0i in p0])

	# alpha = initialize_alpha(nik)
	alpha = estimate_dirichlet(nik)
	# l,alpha = fixedpoint2(nik,alpha,10000,1e-10,1e-8)

	fig,ax = plt.subplots(1)
	ax.plot(alpha0,alpha,'o')
	ax.plot((0,50),(0,50),'k',lw=1)
	plt.show()