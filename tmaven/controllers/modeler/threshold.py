import numpy as np
import numba as nb
from .model_container import model_container

def calc_threshold(y,threshold):
	''' Idealize an np.ndarray with a threshold into classes 0 (<) and 1 (>)'''

	#nstates = 2
	yclass = (y > threshold).astype('int')
	mean = np.array([y[yclass==j].mean() for j in [0,1]]).astype('double')
	var = np.array([y[yclass==j].var() for j in [0,1]]).astype('double')
	frac = np.array([(yclass==j).sum() for j in [0,1]]).astype('double')
	frac /= frac.sum()

	result = model_container(type='threshold',nstates=2,mean=mean,var=var,frac=frac,threshold=threshold)
	return result


def calc_threshold_jump(y,threshold_jump_delta,threshold_jump_n,threshold_jump_sigma):
	''' Idealize an np.ndarray with a threshold into classes 0 (<) and 1 (>)'''

	nstates = 1
	result = model_container(
		type='threshold',
		nstates=nstates,
		mean=np.linspace(0,1,nstates),
		var=np.ones(nstates),
		frac=np.ones(nstates)/float(nstates),
		threshold_jump_delta=threshold_jump_delta,
		threshold_jump_n=threshold_jump_n,
		threshold_jump_sigma=threshold_jump_sigma,
	)
	return result

@nb.njit
def mean(y): ## numba nonsense
	if y.size == 1:
		return y[0]
	else:
		return np.mean(y)

@nb.njit
def ideal_threshold_jump(y,pre,post,delta,nn,sigma):
	## y is [time,colors]
	chain = np.zeros(y.shape[0],dtype='int')
	ideal = np.zeros(y.shape[0],dtype=y.dtype) + np.nan
	if post <= pre:
		return chain,ideal

	## find all local jumps
	i = pre
	first = i
	last = i+nn
	ind = 1
	while i+nn+sigma-1 < post-nn:
		s2 = i+nn
		e2 = i+nn+(sigma-1)
		check = np.abs(mean(y[s2:e2+1])-mean(y[first:i+nn+1])) > delta
		if check:
			last = i + nn - 1
			chain[first:last+1] += ind
			ideal[first:last+1] = mean(y[first:last+1])
			ind += 1
			first = i + nn
			i += nn
		else:
			i += 1
	last = post
	chain[first:last+1] += ind
	ideal[first:last+1] = mean(y[first:last+1])

	## squash small dwells, note: do these in order or you can, e.g., create t=2 by fixing a t = 1 and it blows up...
	for dt in np.arange(1,sigma+1):	
		for i in range(pre,post-1):
			if chain[i] == chain[i+1]:
				continue
			keep = chain == chain[i+1]
			ind = np.nonzero(keep)[0]
			if keep.sum() == dt:
				ind = np.min(ind)
				if np.abs(ideal[ind]-ideal[ind-1]) < np.abs(ideal[ind+dt]-ideal[ind]): ## closer to the L than the R
					chain[ind:post+1] -= 1
				else:
					chain[ind+dt:post+1] -= 1
				keep = chain == chain[ind]
				ideal[keep] = mean(y[keep])

	## merge neighbors
	for i in range(pre,post-1):
		## look for division between neighbors
		chain1 = chain[i]
		chain2 = chain[i+1]
		ideal1 = ideal[i]
		ideal2 = ideal[i+1]

		if chain2 != chain1:
			if np.abs(ideal1-ideal2) <= delta: ## should they be merged?
				chain[i+1:post+1] -= 1
				keep = chain == chain1
				ideal[keep] = mean(y[keep]) ## this will make "assert np.abs(ideal[i+1]-ideal[i]) > delta" fail b/c it reassess only one side... but better RMSD
				# ideal[keep] = ideal1 ## this will NOT fail.... but worse RMSD reconstruction. Don't use this except for debugging...
				assert chain[i] == chain[i+1]
				assert ideal[i] == ideal[i+1]

	# # check
	# for i in range(pre,post-1):
	# 	if chain[i] != chain[i+1]:
	# 		assert np.abs(ideal[i+1]-ideal[i]) > delta

	return chain,ideal
