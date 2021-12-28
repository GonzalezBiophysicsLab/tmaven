import numpy as np
from math import erf as erff
from scipy import optimize
from ..photobleaching.photobleaching import ln_likelihood
import numba as nb

@nb.vectorize
def erf(x):
	return erff(x)

def normal_cdf(x,m,v):
	return .5*(1.+erf((x-m)/np.sqrt(2.*v)))
def normal_pdf(x,m,v):
	return 1./np.sqrt(2.*np.pi*v)*np.exp(-.5/v*(x-m)**2.)

def normal_min_cdf(x,m,v,n):
	return (1.-normal_cdf(x,m,v))**(n)

def estimate_bg_normal(data):
	xs = np.logspace(np.log10(1./data.size),-1.,100)
	ps = np.percentile(data.flatten(),xs*100.)

	params, covars = (optimize.curve_fit(normal_cdf,ps,xs,p0 = np.array((np.median(data),np.var(data)))))

	return params

def p_normal_min_cdf(x,mu,var,n):
	return (1.-normal_cdf(x,mu,var))**n

def estimate_bg_normal_min(data,n):
	xs = np.logspace(np.log10(1./data.size),-1.,100)
	ps = np.percentile(data.flatten(),xs*100.)
	xs = 1. - xs

	params, covars = optimize.curve_fit(lambda xx, mm, vv: p_normal_min_cdf(xx,mm,vv,n),ps,xs,p0 = np.array((np.median(data),np.var(data))),maxfev=10000)

	return params

@nb.njit(nb.double[:,:](nb.double[:,:],nb.double,nb.double,nb.double,nb.double,nb.int64),nogil=True,parallel=True)
def _model_select_many_numba(data,bg_mu,bg_var,sbr_low,sbr_high,min_frames):

	# sbr_low = 2.
	# sbr_high = 5.
	n_std_bg = 3.

	# a,b,k,m
	low_sbr_prior = np.array((1.,bg_var,n_std_bg*np.sqrt(bg_var),bg_mu+sbr_low*np.sqrt(bg_var)))
	high_sbr_prior = np.array((1.,bg_var,n_std_bg*np.sqrt(bg_var),bg_mu+sbr_high*np.sqrt(bg_var)))

	probs = np.zeros((data.shape[0],6))
	for i in nb.prange(data.shape[0]):
		trace = data[i].copy()
		ln_low = ln_likelihood(trace, low_sbr_prior[0],low_sbr_prior[1],low_sbr_prior[2],low_sbr_prior[3])
		ln_high = ln_likelihood(trace, high_sbr_prior[0],high_sbr_prior[1],high_sbr_prior[2],high_sbr_prior[3])

		model_1 = ln_low[:min_frames].max() ## dead,low
		model_2 = ln_low[min_frames:-3].max() ## good,low
		model_3 = ln_low[-3:].max() ## doesn't bleach,low
		model_4 = ln_high[:min_frames].max() ## dead,high
		model_5 = ln_high[min_frames:-3].max() ## good,high
		model_6 = ln_high[-3:].max() ## doesn't bleach,high

		# ## ad hoc to catch negative intensities
		# aml = ln_low.argmax()
		# if aml > min_frames and aml < ln_low.size-3:
		# 	if trace[:aml].mean() < 0:
		# 		model_1 += model_2
		# 		model_2 = -np.inf
		# amh = ln_high.argmax()
		# if amh > min_frames and amh < ln_high.size-3:
		# 	if trace[:amh].mean() < 0:
		# 		model_4 += model_5
		# 		model_5 = -np.inf

		probs[i] = np.array((model_1,model_2,model_3,model_4,model_5,model_6))
		emax = probs[i].max()
		probs[i] = np.exp(probs[i]-emax)
		probs[i] /= probs[i].sum()

		## model priors
		for j in range(6):
			probs[i,j] *= 1./6. ## equal priors
		probs[i] /= np.sum(probs[i])


	return probs

def model_select_many(data,sbr_low,sbr_high,min_frames):
	bg_mu,bg_var = estimate_bg_normal(data)
	return _model_select_many_numba(data,bg_mu,bg_var,sbr_low,sbr_high,min_frames)

# def get_highsbr_behaved(data):
# 	prob = model_select_many(data)
# 	keep = np.nonzero(prob.argmax(1) == 4)
# 	return keep
