import numpy as np
import numba as nb
import scipy.optimize as sopt
import logging
logger = logging.getLogger(__name__)

def generate_dwells(trace, dwell_list, means):
	trace = trace[~np.isnan(trace)]
	#print(trace)
	if len(trace) > 0: #proetcting if all is NaN
		dwell_split = np.split(trace, np.argwhere(np.diff(trace)!=0).flatten()+1)

		if len(dwell_split) > 2: #protecting against no or single transition in a trace
			dwell_split = dwell_split[1:-1] #skipping first and last dwells
			for d in dwell_split:
				ind = int(np.argwhere(d[0] == means))
				dwell_list[str(ind)].append(len(d))

	return dwell_list

def calculate_dwells(result):
	traces = result.idealized
	means = result.mean
	dwell_list = {str(i):[] for i in range(result.nstates)}

	for t in range(len(traces)):
		dwell_list = generate_dwells(traces[t],dwell_list, means)

	result.dwells = dwell_list
	logger.info('Dwell time calculated')

def convert_tmatrix(tmatrix):
	tmstar = tmatrix.copy()
	for i in range(tmstar.shape[0]):
		tmstar[i] /= tmstar[i].sum()

	if tmstar.shape[0] > 1:
		rates = -np.log(1.-tmstar)/1.
		for i in range(rates.shape[0]):
			rates[i,i] = 0.
	else:
		rates = np.zeros_like(tmstar)

	return rates

@nb.njit
def survival(dist):
	n = np.int(np.max(dist))

	raw_surv = np.zeros(n)

	for i in np.arange(n):
		temp = np.zeros_like(dist)
		temp[np.where(dist > i)] = 1
		raw_surv[i] = np.sum(temp)

	norm_surv = raw_surv/raw_surv[0]

	return np.arange(n), norm_surv


def single_exp_surv(tau, k, A):
    return A*np.exp(-k*tau)

def double_exp_surv(tau, k1, k2,A,B):
    return A*np.exp(-k1*tau) + B*np.exp(-k2*tau)

def triple_exp_surv(tau, k1, k2, k3, B, C, A):
    return A*(B*np.exp(-k1*tau) + C*np.exp(-k2*tau) + (1-B-C)*np.exp(-k3*tau))

def stretched_exp_surv(tau, k, beta, A):
    return A*np.exp(-(k*tau)**beta)

def single_exp_hist(tau, k, A):
    return A*k*np.exp(-k*tau)

def double_exp_hist(tau, k1, k2, A, B):
    return A*k1*np.exp(-k1*tau) + B*k2*np.exp(-k2*tau)

def triple_exp_hist(tau, k1, k2, k3, B, C, A):
    return A*(B*k1*np.exp(-k1*tau) + C*k2*np.exp(-k2*tau) + (1-B-C)*k3*np.exp(-k3*tau))

def stretched_exp_hist(tau, k, beta, A):
	return A*np.exp(-(k*tau)**beta)*((tau)**beta)*beta*(k**(beta - 1))

def optimize_single_surv(tau, surv, fix_A = False):
	if fix_A:
		A = 1.
		popt, pcov = sopt.curve_fit(lambda tau, k: single_exp_surv(tau, k, A), tau, surv, bounds = (0,np.inf))
		k = popt[0]
	else:
		popt, pcov = sopt.curve_fit(single_exp_surv, tau, surv, bounds = (0,np.inf))
		k, A = popt

	return np.array([k]),np.array([A])

def optimize_double_surv(tau, surv, fix_A = False):
	if fix_A:
		popt, pcov = sopt.curve_fit(lambda tau, k1,k2,B: double_exp_surv(tau,k1,k2,1-B,B), tau, surv, bounds = ([0.,0.,0.],[np.inf,np.inf,1.]))
		k1,k2,B = popt
		A = 1 - B
	else:
		popt, pcov = sopt.curve_fit(double_exp_surv, tau, surv, bounds = ([0.,0.,0.,0.],[np.inf,np.inf,1.,1.]))
		k1,k2,A,B = popt
		errors = np.sqrt(np.diag(pcov))
		print(errors)

	ks = np.array([k1,k2])
	x = np.argsort(ks)
	ks = ks[x]
	As = np.array([A,B])
	As = As[x]
	print(As)
	return ks,As

def optimize_triple_surv(tau, surv, fix_A = False):
	if fix_A:
		A = 1.
		popt, pcov = sopt.curve_fit(lambda tau, k1,k2,k3,B,C: triple_exp_surv(tau,k1,k2,k3,B,C,A), tau, surv, bounds = ([0.,0.,0.,0.,0.],[np.inf,np.inf,np.inf,1.,1.]))
		k1,k2,k3,B,C = popt
	else:
		popt, pcov = sopt.curve_fit(triple_exp_surv, tau, surv, bounds = ([0.,0.,0.,0.,0.,0.],[np.inf,np.inf,np.inf,1.,1.,np.inf]))
		k1,k2,k3,B,C,A = popt

	ks = np.array([k1,k2,k3])
	x = np.argsort(ks)
	ks = ks[x]
	As = np.array([A*B,A*C,A*(1-B-C)])
	As = As[x]
	return ks,As

def optimize_stretch_surv(tau, surv, fix_A = False):
	if fix_A:
		A = 1.
		popt, pcov = sopt.curve_fit(lambda tau, k,beta: stretched_exp_surv(tau,k,beta,A), tau, surv, bounds = ([0.,0.],[np.inf,1.]))
		k,beta = popt
	else:
		popt, pcov = sopt.curve_fit(stretched_exp_surv, tau, surv, bounds = ([0.,0.,0.],[np.inf,1.,np.inf]))
		k,beta,A = popt

	return np.array([k]),np.array([beta]),np.array([A])
