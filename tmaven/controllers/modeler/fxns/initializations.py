import numpy as np

def initialize_gmm(x,nstates,flag_kmeans=False):
	np.random.seed()
	from .statistics import kernel_sample
	if not flag_kmeans:
		# xx = x[np.random.randint(low=0,high=x.size,size=np.min((xx.size,100)),dtype='i')]
		mu = kernel_sample(x,nstates)

		# distinv = 1./np.sqrt((x[:,None] - mu[None,:])**2.)
		# r = np.exp(-distinv) + .1
		# r = r/r.sum(1)[:,None]
		# var = np.sum(r*(x[:,None]-mu)**2.,axis=0)/np.sum(r,axis=0) + 1e-300
		# ppi = r.mean(0)
		var = np.var(x)/nstates + np.zeros(nstates)
		ppi = 1./nstates + np.zeros(nstates)

	else:
		# try:
			from ..kmeans import kmeans
			out = kmeans(x,nstates)
			mu = out.mu
			var = out.var
			ppi = out.ppi
		# except:
			# mu = kernel_sample(x,nstates)
			# var = np.var(x)/nstates + np.zeros(nstates)
			# ppi = 1./nstates + np.zeros(nstates)

		# mu = np.random.normal(loc=mu,scale=np.sqrt(var),size=nstates)

	return mu,var,ppi

def initialize_tmatrix(nstates):
	if nstates > 1:
		tmatrix = np.zeros((nstates,nstates)) + .1/(nstates-1)
		for i in range(tmatrix.shape[0]):
			tmatrix[i,i] = 1. - .1
	else:
		tmatrix = np.ones((1,1))

	return tmatrix
