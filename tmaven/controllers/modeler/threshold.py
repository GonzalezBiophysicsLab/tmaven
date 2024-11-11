import numpy as np
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