import numpy as np
import numba as nb
import scipy.optimize as sopt
import logging
logger = logging.getLogger(__name__)

from .fxns.exponentials import single_exp_surv, double_exp_surv, triple_exp_surv, stretched_exp_surv

def generate_dwells(trace, dwell_list, means, first_flag):
	trace = trace[~np.isnan(trace)]
	#print(trace)
	if len(trace) > 0: #protecting if all is NaN
		dwell_split = np.split(trace, np.argwhere(np.diff(trace)!=0).flatten()+1)

		if len(dwell_split) > 1: #protecting against no transitions in a trace
			if first_flag:
				start = 0
			else:
				start = 1
			dwell_split = dwell_split[start:-1] #skipping last dwells
			for d in dwell_split:
				ind = int(np.argwhere(d[0] == means))
				dwell_list[str(ind)].append(len(d))

	return dwell_list

def calculate_dwells(result, first_flag):
	dwell_list = {str(i):[] for i in range(result.nstates)}

	if result.type == 'eb HMM':
		for key in result.trace_level.keys():
			trace = result.trace_level[key].chain
			means = np.arange(result.nstates,dtype='int')
			dwell_list = generate_dwells(trace,dwell_list, means, first_flag)

	else:
		traces = result.idealized
		means = result.mean
		for t in range(len(traces)):
			dwell_list = generate_dwells(traces[t],dwell_list, means, first_flag)

	result.dwells = dwell_list
	logger.info('Dwell time calculated')


@nb.njit
def survival(dist):
	if dist.size == 0:
		return np.array([0]), np.array([0.0])

	n = np.int32(np.max(dist))

	raw_surv = np.zeros(n)

	for i in np.arange(n):
		temp = np.zeros_like(dist)
		temp[np.where(dist > i)] = 1
		raw_surv[i] = np.sum(temp)

	if raw_surv[0] == 0:
		norm_surv = np.zeros_like(raw_surv)
	else:
		norm_surv = raw_surv/raw_surv[0]

	return np.arange(n), norm_surv

def optimize_single_surv(tau, surv, fix_A = False):
	if fix_A:
		A = 1.
		popt, pcov = sopt.curve_fit(lambda tau, k: single_exp_surv(tau, k, A), tau, surv, bounds = (0,np.inf))
		k = popt[0]
		error = np.array([[np.sqrt(pcov[0][0])],[0.]])
	else:
		popt, pcov = sopt.curve_fit(single_exp_surv, tau, surv, bounds = (0,np.inf))
		k, A = popt
		error = np.sqrt(np.diag(pcov)).reshape(2,1)
	
	ss_res = np.sum((surv - single_exp_surv(tau, k, A))**2)
	ss_tot = np.sum((surv-np.mean(surv))**2)
	R2 = 1 - ss_res/ss_tot
	
	return np.array([k]), np.array([A]), error, R2

def optimize_double_surv(tau, surv, fix_A = False):
	if fix_A:
		popt, pcov = sopt.curve_fit(lambda tau, k1,k2,B: double_exp_surv(tau,k1,k2,1-B,B), tau, surv, bounds = ([0.,0.,0.],[np.inf,np.inf,1.]))
		k1,k2,B = popt
		A = 1 - B
		error_k = np.sqrt(np.diag(pcov))[:2]
		error_A = np.array([[0.],[np.sqrt(np.diag(pcov))[-1]]])
	else:
		popt, pcov = sopt.curve_fit(double_exp_surv, tau, surv, bounds = ([0.,0.,0.,0.],[np.inf,np.inf,1.,1.]))
		k1,k2,A,B = popt
		error_k = np.sqrt(np.diag(pcov))[:2]
		error_A = np.sqrt(np.diag(pcov))[-2:]

	ks = np.array([k1,k2])
	x = np.argsort(ks)
	ks = ks[x]
	As = np.array([A,B])
	As = As[x]
	error = np.array([error_k, error_A])

	ss_res = np.sum((surv - double_exp_surv(tau, k1, k2, A, B))**2)
	ss_tot = np.sum((surv-np.mean(surv))**2)
	R2 = 1 - ss_res/ss_tot
	
	return ks, As, error, R2

def optimize_triple_surv(tau, surv, fix_A = False):
	if fix_A:
		A = 1.
		popt, pcov = sopt.curve_fit(lambda tau, k1,k2,k3,B,C: triple_exp_surv(tau,k1,k2,k3,1-B-C,B,C), tau, surv, bounds = ([0.,0.,0.,0.,0.],[np.inf,np.inf,np.inf,1.,1.]))
		k1,k2,k3,B,C = popt
		error_k = np.sqrt(np.diag(pcov))[:3]
		error_A = np.zeros_like(error_k)
		error_A[-2:] = np.sqrt(np.diag(pcov))[-2:]
	else:
		popt, pcov = sopt.curve_fit(triple_exp_surv, tau, surv, bounds = ([0.,0.,0.,0.,0.,0.],[np.inf,np.inf,np.inf,1.,1.,np.inf]))
		k1,k2,k3,A,B,C = popt
		error_k = np.sqrt(np.diag(pcov))[:3]
		error_A = np.sqrt(np.diag(pcov))[-3:]

	ks = np.array([k1,k2,k3])
	x = np.argsort(ks)
	ks = ks[x]
	As = np.array([A,B,C])
	As = As[x]
	error = np.array([error_k, error_A])

	ss_res = np.sum((surv - triple_exp_surv(tau, k1, k2, k3, A, B, C))**2)
	ss_tot = np.sum((surv-np.mean(surv))**2)
	R2 = 1 - ss_res/ss_tot
	
	return ks, As, error, R2

def optimize_stretch_surv(tau, surv, fix_A = False):
	if fix_A:
		A = 1.
		popt, pcov = sopt.curve_fit(lambda tau, k,beta: stretched_exp_surv(tau,k,beta,A), tau, surv, bounds = ([0.,0.],[np.inf,1.]))
		k,beta = popt
		error = np.array([[np.sqrt(np.diag(pcov)[0])], [np.sqrt(np.diag(pcov)[1])], [0.]])
	else:
		popt, pcov = sopt.curve_fit(stretched_exp_surv, tau, surv, bounds = ([0.,0.,0.],[np.inf,1.,np.inf]))
		k,beta,A = popt
		error = np.sqrt(np.diag(pcov)).reshape(3,1)

	ss_res = np.sum((surv - stretched_exp_surv(tau, k, beta, A))**2)
	ss_tot = np.sum((surv-np.mean(surv))**2)
	R2 = 1 - ss_res/ss_tot

	return np.array([k]), np.array([beta]), np.array([A]), error, R2
