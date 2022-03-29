import numpy as np
import numba as nb
from math import lgamma
import time

@nb.njit
def _kmeans(x,nstates):
	## normalize
	xmean = x.mean()
	xstd = x.std()
	x = (x-xmean)/xstd

	xmin = x.min()
	xmax = x.max()

	delta = xmax - xmin
	# mu = np.linspace(xmin,xmax,nstates)
	# for k in range(nstates):
	# 	mu[k] += np.random.normal()*delta/10.
	mu = np.zeros(nstates)
	for k in range(nstates):
		mu[k] = x[np.random.randint(0,x.size)]+np.random.normal()*delta/10.
	mu.sort()

	dist = np.zeros((nstates))
	n = np.zeros((nstates))
	r = np.zeros((x.size),dtype=nb.int64)

	success = False
	for iteration in range(500):
		muold = mu.copy()
		mu *= 0
		n *= 0
		for i in range(x.size):
			dist = np.abs(x[i]-muold)
			r[i] = np.argmin(dist)
			mu[r[i]] += x[i]
			n[r[i]] += 1.
		mu = mu/n
		## take care of empties by reinjecting them near good places
		xbad = np.isnan(mu)
		if np.any(xbad):
			nbad = np.nonzero(xbad)[0]
			for k in range(nbad.size):
				mu[nbad[k]] = x[np.random.randint(x.size)] + np.random.normal()*1e-3
			mu.sort()
			flag = False

			# xgood = n > 0
			# nbad = np.nonzero(xbad)[0]
			# ngood = np.nonzero(xgood)[0]
			# std = np.std(mu[xgood])
			# for k in range(nbad.size):
			# 	mu[nbad[k]] = mu[ngood[np.random.randint(ngood.size)]] + np.random.normal()*std
			# # mu.sort()
			# flag = False ## reinject flag -- things are messed up, so don't stop here

		if flag and np.linalg.norm(muold) != 0:
			relchange = np.linalg.norm(mu-muold)/np.linalg.norm(muold)
			if relchange < 1e-10:
				success = True
				break
		flag = True

	x = x*xstd + xmean ## denormalize

	return r,mu,iteration,success

def kmeans(x,nstates):
	## run K-means...
	t0 = time.time()
	r,mu,iteration,success = _kmeans(x,nstates)
	t1 = time.time()
	# print(x.size,nstates,iteration,t1-t0,success)

	## Calculate statistics...
	x0 = np.array([float(np.sum(r==k)) for k in range(nstates)])
	x1 = np.array([np.sum(x[r==k]**1.) for k in range(nstates)])
	x2 = np.array([np.sum(x[r==k]**2.) for k in range(nstates)])

	# print(x0)
	# print(x1)
	# print(x2)
	## Output parameters
	pi = x0/x0.sum()
	mu = x1/x0
	var = x2/x0 - mu**2.
	resp = np.zeros((x.size,mu.size))
	for k in range(mu.size):
		resp[r == k,k] = 1.

	# print(pi)
	from .model_container import model_container
	out = model_container(type='kmeans',mean=mu,var=var,frac=pi, nstates=nstates, r=resp)
	return out


if __name__ == '__main__':
	import matplotlib.pyplot as plt
	# np.random.seed(666)

	nstates = 4
	mu0 = np.array([np.random.rand()*100 for _ in range(nstates)])
	print(mu0[mu0.argsort()],100,101)
	d = np.concatenate([np.random.normal(size=np.random.randint(low=50,high=500))+mu0[i] for i in range(nstates)])
	d = np.append(d,100.)
	d = np.append(d,101.)
	# print(d.shape)
	import time
	out = kmeans(d,4) ## force compile
	ts = []
	for _ in range(5):
		t0 = time.time()
		out = kmeans(d,4)
		t1 = time.time()
		ts.append(t1-t0)
	print('time',np.median(ts))
	print(out.mu)
	# print(var)
	# print(pi)

	xmean = d.mean()
	xstd = d.std()
	d = (d-xmean)/xstd
	t = np.arange(d.size)
	for i in range(out.mu.size):
		xkeep = out.r.argmax(1) == i
		plt.plot(t[xkeep],d[xkeep],ls='None',marker='o',alpha=.05)
	plt.show()
	#
	# plt.hist(d,bins=1000)
	# for i in range(mu.size):
	# 	plt.axvline(mu[i],color='k')
	# plt.show()
