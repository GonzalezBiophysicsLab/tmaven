import numpy as np
import numba as nb
from sys import platform
import multiprocessing as mp
import warnings

from .fxns.numba_math import psi, trigamma, invert_psi
from .fxns.statistics import dkl_dirichlet,dkl_dirichlet_2D, dkl_normgamma,p_normal,ln_p_normal
# from .fxns.initializations import initialize_gmm,initialize_tmatrix
from .hmm_vb import outer_loop as vb_outer_loop
from .hmm_vb import vb_em_hmm
from .kmeans import _kmeans
from .fxns.hmm import viterbi
from .hmm_vb_consensus import m_updates

@nb.jit(nopython=True, cache=True)
def h_step_normgamma(w_m, w_beta, w_a, w_b):
	'''
	Empirical Bayes step for a Normal-Gamma prior. Solves the following equations
		m = - E[m l] /  E[-l]
		beta = 1 / (E[m^2 l] - E[m l]^2 / E[l])
		psi(a) - log(a) = E[log l] - log(E[l])
		b = - a / E[-l]

	Inputs
	----------

	w_m, w_beta, w_a, w_b : [N x K] array
							Normal-Gamma posterior parameters for K states and N samples

	Outputs
	----------

	u_m, u_beta, u_a, u_b : [K x 1] array
							Solved Normal-Gamma hyperparameters
	'''

	### Copied and Translated from JWvdM's ebfret normgamma.h_step.m file, available at link below
	### https://github.com/ebfret/ebfret-gui/blob/master/src/%2Bebfret/%2Banalysis/%2Bdist/%2Bnormgamma/h_step.m

	# Initialize
	MAX_ITER = 1000
	N,K = w_m.shape
	threshold = 1e-6
	a0 = 2.*np.ones(K)

	N, K = w_m.shape

	# Initialize u from naive average
	u_m = np.sum(w_m, 0)/N
	u_beta = np.sum(w_beta, 0)/N
	u_a = np.sum(w_a, 0)/N
	u_b = np.sum(w_b, 0)/N

	# Expectation Value E[lambda] = w_a / w_b)
	E_l = np.sum(w_a / w_b, 0)/N
	log_E_l = np.log(E_l)

	# Expectation Value E[mu * lambda] = w_m * w_a / w_b
	E_ml = np.sum(w_m * w_a / w_b, 0)/N

	# Expectation Value E[mu^2 * lambda] = 1 / w_beta + w_m^2 w_a / w_b
	E_m2l = np.sum(1 / w_beta + w_m**2 * w_a / w_b, 0)/N

	# Expectation Value E[log lambda] = psi(w_a) - log(w_b)
	E_log_l = np.sum(psi(w_a) - np.log(w_b), 0)/N

	# Use Newton's Method to numerically solve for u_a
	# psi(u_a) - log(u_a) = E[log(lambda)] - log(E[lambda])
	a_old = a0
	a = a0


	for it in range(MAX_ITER):
		# gradient
		# g = psi(a) - log(a) - (E[log(l)] - log(E[l]))
		g = psi(a) - np.log(a) - (E_log_l - log_E_l)
		# hessian H
		H = trigamma(a) - 1. / a
		# da = (H^-1 g)
		da = g / H
		# Ensure a > 1
		a = np.maximum(a - da, np.ones(K) * (1. + 1e-3 * (a - 1.)))

		# Check convergence
		if np.all((np.abs(a - a_old) / a) < threshold):
			break
		a_old = a

	# m = E[m l] / E[l]
	u_m = E_ml / E_l

	# beta = 1 / (E[m^2 l] - E[m l]^2 / E[l])
	u_beta = 1 / (E_m2l - E_ml**2 / E_l)

	# psi(u_a) - log(u_a) = E[log(lambda)] - log(E[lambda])
	u_a = a

	# b = a / E[l]
	u_b = a / E_l

	# Enforce constraints
	u_a[u_a < 1+1e-3] = 1+1e-3
	u_beta[u_beta < 1e-3] = 1e-3

	'''
	# Warn if maximum iterations reached
	if it == MAX_ITER:
		warnings.warn('Newton solver for hyperparameters did not converge in %d iterations.' % MAX_ITER)
	'''
	return u_m, u_beta, u_a, u_b

@nb.njit(cache=True)
def estimate_dirichlet(pnk,maxiters=1000,eps=1e-16):
	N,K = pnk.shape
	# assert not np.any(np.isnan(pnk))

	## calculate statistics
	E = np.zeros(K)
	for n in range(N):		
		pn0 = psi(np.sum(pnk[n]+eps))
		for k in range(K):
			E[k] += (psi(pnk[n,k]+eps) - pn0)/float(N)
	# test = np.mean(psi(pnk)-psi(np.sum(pnk,axis=1))[:,None],axis=0)
	# assert np.allclose(E,test)
	
	## initialize ## doesn't seem to be stable
	# Ep = np.zeros(K)
	# Epp = np.zeros(K)
	# for k in range(K):
	# 	Ep[k] += np.mean(pnk[:,k])
	# 	Epp[k] += np.mean((pnk[:,k])**2.)
	# # Ep = np.mean(pnk+eps,axis=0)
	# # Epp = np.mean((pnk+eps)**2., axis=0)
	# alpha0 = (Ep[0]-Epp[0])/(Epp[0]-Ep[0]**2.)
	# alpha = alpha0*Ep
	# assert np.all(alpha>0)
	# assert not np.any(np.isnan(alpha))
	alpha = np.ones(K)

	## Invert
	last_pa0 = np.inf
	for iteration in range(maxiters):
		psi_alpha_0 = psi(np.sum(alpha)+eps)
		for k in range(K):
			alpha[k] = invert_psi(psi_alpha_0 + E[k] + eps)
		if np.isclose(psi_alpha_0,last_pa0):
			break
		else:
			last_pa0 = psi_alpha_0
	
	# ## debug
	# if np.any(np.isnan(alpha)):
	# 	for n in range(N):
	# 		print(pnk[n])
	# 	print(alpha)
	# 	assert not np.any(np.isnan(alpha))
	return alpha

@nb.jit(nopython=True, cache=True)
def h_step(w_mu, w_beta, w_a, w_b, w_A, w_pi,
			E_z, E_z1, E_zz, E_x, V_x,
			mu_old, beta_old, a_old, b_old, tm_old, pi_old):
	'''
	Hyper parameter updates for empirical Bayes inference (EB) on a single-molecule FRET dataset.

	The input of this method is a set of  posteriors produced by running
	Variational Bayes Expectation Maxmimization (VBEM) on a series
	of FRET traces. This method maximizes the total summed evidence by solving
	the system of equations:

		Sum_n  Grad_u L_n  =  0

	Where u is the set of hyperparameters that determine the form of
	the prior distribution on the parameters. L_n is the lower bound
	evidence for the n-th trace, defined by:

		L  =  Integral d z d theta  q(z) q(theta)
			  ln[ p(x, z | theta) p(theta) / (q(z) q(theta)) ]

	The approximate posteriors q(z) and q(theta), which have been
	optimized in the VBEM process, are now kept constant as Sum L_n is
	maximized wrt to u.


	Inputs
	------

	Variational parameters of approximate posterior distribution
	for parameters q(theta | w)

	w_tm   : [N x K x K] array
			 Dirichlet prior for each row of transition matrix

	w_pi   : [N x K] array
			 Dirichlet prior for initial state probabilities

	w_m    : [N x K] array
			 Normal-Gamma prior - state means

	w_beta : [N x K] array
			 Normal-Gamma prior - state occupation count

	w_a    : [N x K] array
			 Normal-Gamma prior - shape parameter

	w_b    : [N x K] array
			 Normal-Gamma prior - rate parameter

	E_z    : [N x K] array
			 Expected statistics

	E_z1   : [N x K] array
			 Expected statistics

	E_zz   : [N x K x K] array
			 Expected statistics

	E_x    : [N x K] array
			 Expected statistics

	V_x    : [N x K] array
			 Expected statistics

	mu_old : [1 x K] array
			 Previous prior

	beta_old : [1 x K] array
			 Previous prior

	a_old  : [1 x K] array
			 Previous prior

	b_old  : [1 x K] array
			 Previous prior

	tm_old : [K x K] array
			 Previous prior

	pi_old : [1 x K] array
			 Previous prior

	Outputs
	-------

	Hyperparameters for the prior distribution p(theta | u)
	(same fields as w)

	u_tm, u_pi, u_m, u_beta, u_a, u_b :
	'''

	### Copied and Translated from JWvdM's ebfret h_step.m file, available at link below
	### https://github.com/ebfret/ebfret-gui/blob/master/src/%2Bebfret/%2Banalysis/%2Bhmm/h_step.m

	'''
	# Process input data
	w_mu, w_beta, w_a, w_b = posterior_arr[0], posterior_arr[1], posterior_arr[2], posterior_arr[3]
	w_A, w_pi = posterior_arr[4], posterior_arr[5]
	E_z, E_z1, E_zz, E_x, V_x = exp_arr[0], exp_arr[1], exp_arr[2], exp_arr[3], exp_arr[4]
	mu_old, beta_old, a_old, b_old = prior_arr[0], prior_arr[1], prior_arr[2], prior_arr[3]
	tm_old, pi_old  = prior_arr[4], prior_arr[5]
	'''
	# Initialize
	MAX_ITER = 100
	threshold = 1e-5
	kl0 = -np.inf
	kl1 = -np.inf
	it = 0

	while True:
		# Run normal-gamma updates for emission model parameters
		# assert not np.any(np.isnan(w_mu))
		# assert not np.any(np.isnan(w_beta))
		# assert not np.any(np.isnan(w_a))
		# assert not np.any(np.isnan(w_b))
		mu, beta, a, b = h_step_normgamma(w_mu, w_beta, w_a, w_b)
		
		## Run dirichlet updates for transition matrix
		# assert not np.any(np.isnan(w_A))
		tm = tm_old.copy()
		for k in range(tm.shape[1]):
			tm[k] = estimate_dirichlet(w_A[:,k])

		# assert not np.any(np.isnan(w_pi))
		pi = estimate_dirichlet(w_pi)
	
		assert not np.any(np.isnan(mu))
		assert not np.any(np.isnan(tm))
		assert not np.any(np.isnan(pi))
		# print(w_pi.shape,w_pi.dtype,w_pi,pi)

		# Check iteration count
		if it >= MAX_ITER:
			break

		# Check for convergence
		if (it >= 1):
			D_kl_pi = dkl_dirichlet(pi, pi_old)
			D_kl_tm = dkl_dirichlet_2D(tm,tm_old)
			D_kl_ng = dkl_normgamma(mu, beta, a, b, mu_old, beta_old, a_old, b_old)
			kl1 = D_kl_pi + D_kl_tm + D_kl_ng

		if (it >= 2) and np.abs(kl0 - kl1) / (1 - kl1) < threshold:
			break

		# Update posteriors
		w_pi = pi[None,:] + E_z1
		w_A = tm[None,:] + E_zz
		w_beta = beta[None,:] + E_z
		w_mu = (E_z * E_x + beta[None,:] * mu[None,:]) / w_beta
		w_a = 0.5 * E_z + a[None,:]
		w_b = 0.5 * (2 * b[None,:] + (E_z * V_x + ((E_z * beta[None,:]) / w_beta * (E_x - mu[None,:])**2)))

		# Next iteration
		it += 1
		kl0 = kl1

		mu_old = mu
		beta_old = beta
		a_old = a
		b_old = b
		tm_old = tm
		pi_old = pi

	return mu, beta, a, b, pi, tm

@nb.jit(nb.types.Tuple(
	(nb.float64[:],
	nb.float64[:],
	nb.float64[:],
	nb.float64[:],
	nb.float64[:],
	nb.float64[:,:],
	nb.float64[:],
	nb.int64,
	nb.float64[:,:]))
	(nb.int64[:],
	nb.float64[:],
	nb.int64,
	nb.int64,
	nb.float64,
	nb.float64[:],
	nb.float64[:],
	nb.float64[:],
	nb.float64[:],
	nb.float64[:],
	nb.float64[:,:],
	nb.int64),nopython=True,parallel=False, cache=True)
def eb_outer_loop(xind,xdata,nstates,maxiters,threshold,mu_prior,beta_prior,a_prior,b_prior,pi_prior,tm_prior,nrestarts):

	N = xind[-1]+1
	K = nstates

	emp_mu = np.zeros((nrestarts,nstates))
	emp_beta = np.zeros((nrestarts,nstates))
	emp_a = np.zeros((nrestarts,nstates))
	emp_b = np.zeros((nrestarts,nstates))
	emp_pi = np.zeros((nrestarts,nstates))
	emp_tm = np.zeros((nrestarts,nstates,nstates))
	L_global = np.zeros((nrestarts,maxiters))-np.inf
	iteration = np.zeros((nrestarts),dtype=np.int64)

	E_z = np.zeros((N,K))
	E_z_out = np.zeros((nrestarts,N,K))
	E_z1 = np.zeros((N,K))
	E_zz = np.zeros((N,K,K))
	E_x = np.zeros((N,K))
	E_xx = np.zeros((N,K))
	pos_mu = np.zeros((N,K))
	pos_beta = np.zeros((N,K))
	pos_a = np.zeros((N,K))
	pos_b = np.zeros((N,K))
	pos_pi = np.zeros((N,K))
	pos_tm = np.zeros((N,K,K))

	## split up the observed data range into K evenly spaced regions. Pull a random percentile from each region as the initial mu
	xlim = np.linspace(0,100,K+1)
	cuts = np.zeros((nrestarts*K))
	for nr in range(nrestarts):
		cuts[nr*K:(nr+1)*K] = np.random.rand(K)*(xlim[1:]-xlim[:-1]) + xlim[:-1]
	_mus = np.percentile(xdata,cuts)
	for nr in range(nrestarts):
		emp_mu[nr] = np.sort(_mus[nr*K:(nr+1)*K])
	var = np.var(xdata)/float(K) + emp_mu*0.

	## INITIALIZATIONS
	for nr in range(nrestarts):
		mu_prior = emp_mu[nr]

		prob = p_normal(xdata,emp_mu[nr],var[nr]) 
		r = np.zeros_like(prob)
		for i in range(r.shape[0]):
			r[i] = prob[i]  ## for stability
			r[i] /= np.sum(r[i])
		assert not np.any(np.isnan(r))
		
		## randomize the prior...
		# emp_mu[nr] = mu_prior + np.random.normal(size=K)*np.sqrt(emp_b[nr]/emp_a[nr]) ## zuzsh it up a lot
		# emp_mu[nr] = np.sort(emp_mu[nr])
		# print(emp_mu[nr])
		
		emp_a[nr],emp_b[nr],_,emp_beta[nr],nk,_,_ = m_updates(xdata,r,a_prior,b_prior,mu_prior,beta_prior)

		# print(var[nr],emp_b[nr]/emp_a[nr])
		# prob = p_normal(xdata,emp_mu[nr],emp_b[nr]/emp_a[nr])
		# r = np.zeros_like(prob)
		# for i in range(r.shape[0]):
		# 	r[i] = prob[i] #+ 1e-16 ## for stability
		# 	r[i] /= np.sum(r[i]) 
		# emp_a[nr],emp_b[nr],_,emp_beta[nr],nk,_,_ = m_updates(xdata,r,emp_a[nr],emp_b[nr],emp_mu[nr],emp_beta[nr])

		emp_pi[nr] = np.sqrt(nk+1.)
		emp_tm[nr] = tm_prior.copy()
		paths = viterbi(xdata,emp_mu[nr],emp_b[nr]/emp_a[nr],emp_tm[nr],emp_pi[nr]).astype('int')
		for k in range(K):
			keep = paths == k
			if keep.sum() > 0:
				emp_pi[nr][k] = float(keep.sum())/float(xdata.size)*np.sqrt(float(N))
				for kk in range(K):
					if k != kk:
						emp_tm[nr][k,kk] = np.sum(np.bitwise_and(paths[1:]==kk,paths[:-1]==k))
				emp_tm[nr][k] /= np.sum(emp_tm[nr][k])*10.
				emp_tm[nr][k,k] = 1. - np.sum(emp_tm[nr][k])
				emp_tm[nr] *= np.sqrt(float(N))

	## RUN: ebFRET Restarts
	for nr in range(nrestarts):
		while iteration[nr] < maxiters:
			L_sum = 0

			for Ni in range(N):
				trace = xdata[xind==Ni]
				r,a,b,mu,beta,pi,tmatrix,E_lnlam,E_lnpi,E_lntm,vblikelihood,vbiteration,xi,xbark,sk = vb_outer_loop(trace,emp_mu[nr],emp_b[nr]/emp_a[nr],emp_tm[nr],emp_mu[nr],emp_beta[nr],emp_a[nr],emp_b[nr],emp_pi[nr],emp_tm[nr],maxiters,threshold)
				r /= r.sum(1)[:,None]
				assert not np.any(np.isnan(r))

				# Collect posteriors
				for k in range(nstates):
					pos_mu[Ni,k] = mu[k]
					pos_beta[Ni,k] = beta[k]
					pos_a[Ni,k] = a[k]
					pos_b[Ni,k] = b[k]
					pos_pi[Ni,k] = pi[k]
					for k2 in range(nstates):
						pos_tm[Ni,k,k2] = tmatrix[k,k2]
				
				# Map expected statistics
				E_z1[Ni] = 0.
				E_z[Ni] = 0.
				E_zz[Ni] = 0.
				for k in range(K):
					E_z1[Ni,k] = r[0,k]
					for n in range(1,r.shape[0]): ## skip n = 0
						E_z[Ni,k] += r[n,k]
				for n in range(xi.shape[0]):
					for k1 in range(xi.shape[1]):
						for k2 in range(xi.shape[2]):
							E_zz[Ni,k1,k2] += xi[n,k1,k2]
				E_x[Ni] = xbark
				E_xx[Ni] = sk + xbark**2.

				# assert not np.any(np.isnan(E_z1[Ni]))
				# assert not np.any(np.isnan(E_z[Ni]))
				# assert not np.any(np.isnan(E_zz[Ni]))
				# assert not np.any(np.isnan(E_x[Ni]))
				# assert not np.any(np.isnan(E_xx[Ni]))

				# # Add individual trace contribution to global ELBO
				if vbiteration == vblikelihood.shape[0]:
					vbiteration -= 1
				L_sum += vblikelihood[vbiteration,0]

			V_x = E_xx - E_x**2
			emp_mu[nr], emp_beta[nr], emp_a[nr], emp_b[nr], emp_pi[nr], emp_tm[nr] =  h_step(pos_mu, pos_beta, pos_a, pos_b, pos_tm, pos_pi, E_z, E_z1, E_zz, E_x, V_x, mu_prior, beta_prior, a_prior, b_prior, tm_prior, pi_prior)
			
			# if np.any(np.isnan(emp_mu)):
			# 	print(emp_mu)
			# 	assert False
			# print(nr,iteration[nr],L_global[nr,iteration[nr]],emp_mu[nr],np.sqrt(emp_b[nr]/emp_a[nr]),emp_pi[nr],emp_tm[nr].flatten())

			# Data processing
			L_global[nr,iteration[nr]] = L_sum

			print(nr, iteration[nr], emp_mu[nr])
			if np.isnan(L_global[nr,iteration[nr]]):
			# 	# L_global[nr,iteration[nr]] = -np.inf
				break

			if (iteration[nr] >= 2) and ((L_global[nr,iteration[nr]] - L_global[nr,iteration[nr]-1]) < threshold * np.abs(L_global[nr,iteration[nr]]) or (L_global[nr,iteration[nr]] - L_global[nr,iteration[nr]-1]) < 0.1):
				break

			iteration[nr] += 1
		E_z_out[nr] = E_z

	Lbest = np.zeros(nrestarts)
	for nr in range(nrestarts):
		Lbest[nr] = L_global[nr, iteration[nr]]

	# Initialize `best` with a placeholder index
	best = -1
	for nr in range(nrestarts):
		if np.isfinite(Lbest[nr]):
			best = nr
			break

	# Now compare remaining finite entries if `best` was initialized
	if best != -1:
		for nr in range(nrestarts):
			if np.isfinite(Lbest[nr]) and Lbest[nr] > Lbest[best]:
				best = nr

	print(best, Lbest[best])

	# print(nstates,Lbest,L_global[best,iteration[best]-1],iteration, best)
	return emp_mu[best], emp_beta[best], emp_a[best], emp_b[best], emp_pi[best], emp_tm[best], L_global[best,:iteration[best]], iteration[best], E_z_out[best]



def eb_em_hmm(x,nstates,maxiters=1000,nrestarts=1,threshold=1e-10,priors=None,ncpu=1,init_kmeans=False):
	'''
	Data convention is NxTxK
	'''
	## Tuning:
	## It takes about 1 minute to compile everything with parallel off
	## It takes about 3 minutes to compile with parallel on
	## It takes O(1-10) seconds per restart so... barely worth it...


	if x[0].ndim != 1:
		raise Exception("Input data isn't 1D")

	xind = np.concatenate([np.zeros(x[i].size,dtype='int64')+i for i in range(len(x))])
	xdata = np.concatenate(x)

	#### The prior mu is the major source of variation between restarts.
	#### Individual restarts converge down to ~+/- 1. (controlled by eb_outer_loop)
	#### Re-runs (though the menu) converge down to ~ +/- 10.0 (controlled by this function)

	### Example: states [Ls by restart...] [iterations by restart ...]
	### Run 1: 2 [133457.22471814 133457.05170969 133457.77358034 133457.21525145] [14  5 19  9]
	### Run 2: 2 [133451.96569838 133452.89787639 133453.45190357 133452.88836708] [18 14 19  9]

	### Seems to be a lot more reproducible if you drop beta from .25 to .0025??

	## fairly good option
	# mmn = 10
	# mu = np.zeros(nstates)
	# for _ in range(mmn):
	# 	kxdata = np.random.choice(xdata,size=(xdata.size//mmn))
	# 	kr,kmu,_,_ = _kmeans(kxdata,nstates)
	# 	mu += np.sort(kmu) /float(mmn)
	# 	mu_prior = mu.copy()

	## Priors - beta, a, b, pi, alpha... mu is from GMM
	if priors is None:
		beta_prior = np.ones(nstates) *0.25
		a_prior = np.ones(nstates) *2.5
		b_prior = np.ones(nstates)*0.01
		pi_prior = np.ones(nstates)
		tm_prior = np.ones(nstates,nstates)
		mu_prior = np.percentile(xdata,np.linspace(0.,100.,nstates+2))[1:-1]
	else:
		mu_prior,beta_prior, a_prior, b_prior, pi_prior, tm_prior = priors

	mu,beta,a,b,pi,tmatrix,likelihood,iteration,E_z = eb_outer_loop(xind,xdata,nstates,maxiters,threshold,mu_prior,beta_prior, a_prior, b_prior, pi_prior, tm_prior, nrestarts)

	## Calculate the individual results
	vb_results = []
	for i in range(len(x)):
		vb_results.append(vb_em_hmm(x[i],nstates,maxiters,threshold,[mu,beta,a,b,pi,tmatrix],init_kmeans, True))

	var = 1/(a/b)
	E_z += 1
	frac = E_z.sum(0)/E_z.sum()
	#frac = pi/(pi.sum())

	priors = {'mu_prior':mu_prior,
			  'beta_prior':beta_prior,
			  'a_prior':a_prior,
			  'b_prior':b_prior,
			  'pi_prior':pi_prior,
			  'tm_prior':tm_prior}

	from .model_container import model_container

	out = model_container(type='eb HMM',
						nstates=nstates,mean=mu,var=var,frac=frac,
						tmatrix=tmatrix,
						likelihood=likelihood,
						iteration=iteration,a=a,b=b,beta=beta, pi=pi,
						priors=priors)

	# print(nstates, likelihood[-1])
	return out, vb_results
