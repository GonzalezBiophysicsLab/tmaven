## Autocorrelations
import numpy as np
import numba as nb
from scipy import stats
from math import lgamma
from scipy.optimize import minimize

#### testing
# dt = 0.025
# mu = np.array((.1,.3,.6,.9))
# noise = 0.01
# var = np.zeros_like(mu) + noise**2.
# rates = 10. * np.array(((0.,.005,.003,.001),(.001,0.,.002,.002),(.004,.003,.0,.004),(0.002,.003,.001,0.)))
# from scipy.linalg import expm
# q = rates.copy()
# for i in range(q.shape[0]):
# 	q[i,i] = - q[i].sum()
# tmatrix = expm(q*dt)
# tinf = 1000./np.abs(q).min()
# popplot.hmm.t,popplot.hmm.y = gen_mc_acf(1.,popplot.ens.y.size,tmatrix,mu,var,ppi)
# popplot.hmm.t,popplot.hmm.y = gen_mc_acf_q(dt,popplot.ens.y.size,q,mu,var,ppi)
# popplot.hmm.t /= dt

@nb.vectorize
def vgamma(x):
	return np.exp(lgamma(x))

def gen_mc_acf_q(tau,nsteps,q,mu,var,ppi):
	## Using a Q matrix, not a transition probability matrix
	## note..... these are transposed relative to tmatrix out of hmm routines

	nstates,_ = q.shape
	pi0 = np.eye(nstates)

	## get steady state probabilities
	from scipy.linalg import expm
	tinf = 1000./np.abs(q).min()
	pinf = expm(q*tinf)[0]

	## use fluctuations
	mubar =  (pinf*mu).sum()
	mm = mu - mubar

	### expectation here
	E_y0yt = np.zeros(nsteps)
	for k in range(nsteps): # loop over time delay steps
		tp = expm(q*tau*k)
		for i in range(nstates): # loop over initial state
			for j in range(nstates): # loop over final state
				E_y0yt[k] += mm[i]*mm[j] * np.dot(tp.T,pi0[i])[j] * pinf[i]

	## add gaussian noise terms
	for i in range(nstates):
		E_y0yt[0] += var[i]*pinf[i]
	## normalize
	E_y0yt /= E_y0yt[0]

	t = tau*np.arange(nsteps)
	return t,E_y0yt

@nb.njit
def gen_mc_acf(tau,nsteps,tmatrix,mu,var,ppi):
	## Using a transition probability matrix, not a rate matrix, or Q matrix
	## because this comes straight out of an HMM.... so not quick exact

	nstates,_ = tmatrix.shape
	pi0 = np.eye(nstates)

	## get steady state probabilities
	pinf = np.linalg.matrix_power(tmatrix.T,100*int(1./tmatrix.min()))[:,0]

	## use fluctuations
	mubar =  (pinf*mu).sum()
	mm = mu - mubar

	### expectation here
	E_y0yt = np.zeros(nsteps)
	for i in range(nstates): # loop over initial state
		for j in range(nstates): # loop over final state
			for k in range(nsteps): # loop over time delay steps
				## E[y_0*t_t] = \sum_ij m_i * m_j * (A^n \cdot \delta (P_i))_j * P_inf,i
				E_y0yt[k] += mm[i]*mm[j] * (np.dot(np.linalg.matrix_power(tmatrix.T,k),pi0[i])[j]) * pinf[i]

	## add gaussian noise terms
	for i in range(nstates):
		E_y0yt[0] += var[i]*pinf[i]
	## normalize
	E_y0yt /= E_y0yt[0]

	t = tau*np.arange(nsteps)
	return t,E_y0yt

################################################################################

# @nb.jit(nopython=True)
# def acorr(d):
# 	## calculate the autocorrelation function
# 	c = acorr_counts(d)
# 	return c[0]/c[1]
#
# @nb.jit(nopython=True)
# def acorr_counts(d):
# 	## calc sum and counts for single trace
# 	out = np.zeros((2,d.size))
# 	for i in range(d.size): ## delay
# 		for j in range(0,d.size-i):
# 			a = d[j]*d[i+j]
# 			if not np.isnan(a):
# 				out[0,i] += a
# 				out[1,i] += 1.0
# 	return out

@nb.njit
def acf_estimator_correct(x):
	## Following Lu, and Bout J. Chem. Phys. 125, 124701 (2006)
	## Equations 12,18

	nmol,nframes = x.shape

	# xbar = np.zeros((nframes))
	# for t in range(nframes):
	# 	T = 0.0
	# 	for n in range(nmol):
	# 		if not np.isnan(x[n,t]):
	# 			xbar[t] += x[n,t]
	# 			T += 1.0
	# 	if T > 0:
	# 		xbar[t] /= T
	# # xbar = 0.0
	# # T = 0.0
	# # for n in range(nmol):
	# # 	for t in range(nframes):
	# # 		if not np.isnan(x[n,t]):
	# # 			xbar += x[n,t]
	# # 			T += 1.0
	# # xbar /= T

	acf = np.zeros((nframes),dtype=x.dtype)
	for k in range(nframes):
		count = 0.
		for t in range(0,nframes-k):
			for n in range(nmol):
				# a = (x[n,t]-xbar)*(x[n,t+k]-xbar)
				# a = (x[n,t]-xbar[t])*(x[n,t+k]-xbar[t+k])
				a = (x[n,t])*(x[n,t+k])
				if not np.isnan(a):
					acf[k] += a
					count += 1.
		if count > 0:
			acf[k] /= count
	acf /= acf[0]
	return acf

## FUCK. cannot numba np.fft .... realized this way too late.
## Also, ffts are so wrong for this. ugh.
## Also Also, padding and duplicating ugh.
def acf_estimator_fft(x):
	nmol,nframes = x.shape

	acf = np.zeros((nframes))
	nacfs = np.zeros((nframes))

	# xbar = np.nanmean(x,axis=0)

	for n in range(nmol):
		xkeep = np.isfinite(x[n]) ## NaNs only on the edges
		nx = xkeep.sum()
		## pad it to damp out fluctuations (gah people do terrible things - how can you possibly justify this?!)
		# xx = np.zeros(nx*2,dtype=x.dtype)
		# xtemp = x[n][xkeep].copy()
		# xx[:nx] = xtemp - xtemp.mean()
		# xx[:nx] = x[n][xkeep] - xbar

		xx = np.zeros(nx*2,dtype=x.dtype)
		# xx[:nx] = x[n][xkeep] - xbar[xkeep]
		# xx[nx:] = (x[n][xkeep] - xbar[xkeep])[::-1]
		xx[:nx] = x[n][xkeep]
		xx[nx:] = (x[n][xkeep])[::-1]


		# xx = x[n][xkeep] - xbar


		f = np.fft.fft(xx)
		out = np.fft.ifft(f*f.conj())[:nx].real
		# out /= out[0]
		# acf[:nx] += out.copy()
		# nacfs[:nx] += 1.

		# out /= out[0]
		acf[:nx] += out
		# nacfs[:nx] += 1.

	# acf /= nacfs
	acf /= acf[0]

	acf[np.isnan(acf)] = 0.
	return acf



# @nb.jit(nopython=True)
# def ensemble_bayes_acorr(dd):
# 	## dd shape is NxT with NaN pads on left and right for bad data
# 	N,T = dd.shape
# 	y = np.zeros((T))
# 	n = np.zeros((T))
#
# 	nm1 = 0.
# 	nm2 = 0.
# 	for i in range(N):
# 		for j in range(T):
# 			if not np.isnan(dd[i,j]):
# 				nm1 +=dd[i,j]
# 				nm2 += 1.
# 	abar = nm1/nm2
#
# 	for i in range(N):
# 		# temp = acorr_counts(dd[i]-abar)
# 		temp = acorr_counts(dd[i])
# 		y += temp[0]
# 		n += temp[1]
#
# 	## calculate posterior
# 	## mn,kn,an,bn
# 	posterior = np.zeros((4,y.shape[0]))
#
# 	#### Regular analysis
# 	# Priors
# 	a0 = 1.
# 	k0 = .1
# 	# m0 = abar**2.
# 	m0 = 0.
# 	b0 = 10.
#
# 	ybar = y/n
#
# 	an = a0 + n/2.
# 	kn = k0 + n
# 	mn = (k0*m0 + y)/kn
# 	bn = b0 + k0*n*(ybar-m0)**2. / (2.*kn)
#
# 	for k in range(N):
# 		d = dd[k]
# 		for i in range(d.size):
# 			for j in range(0,d.size-i):
# 				yy = d[j]*d[i+j]
# 				if not np.isnan(yy): ## these datapoints do
# 					bn[i] += .5*(yy - ybar[i])**2.
# 	# #### Reference Analysis
# 	# ybar = y/n
# 	# mn = ybar
# 	# kn = n
# 	# an = (n-1.)/2.
# 	# bn = 0.*n
# 	# for k in range(N):
# 	# 	d = dd[k]
# 	# 	for i in range(d.size):
# 	# 		for j in range(0,d.size-i):
# 	# 			yy = d[j]*d[i+j]
# 	# 			if not np.isnan(yy): ## these datapoints do
# 	# 				bn[i] += .5*(yy - ybar[i])**2.
#
#
# 	posterior[0] = mn
# 	posterior[1] = kn
# 	posterior[2] = an
# 	posterior[3] = bn
#
# 	## messing up FFT
# 	# for i in range(y.size):
# 	# 	if n[i] == 0:
# 	# 		posterior[0,i] = np.nan
#
# 	return posterior
#
#
# def credible_interval(posterior,p=.95):
# 	## generate credible interval lower and upper lines from a posterior
# 	dp = (1.-p)/2. ## ie 2.5 if p = 95%
# 	ps = np.array([dp,1.-dp]) ## ie 2.5 to 97.5 if p = 95%
#
# 	mn,kn,an,bn = posterior
# 	nu = 2.*an
# 	sig = np.sqrt(bn/(an*kn))
#
# 	# Use Student's T-distribution PPF from scipy.stats.t
# 	ci = stats.t.ppf(ps[:,None], nu[None,:], loc=mn[None,:], scale=sig[None,:])
# 	return ci
#
# def plot_bayes_acorr(d,axis=None,color='blue',normalize = True):
# 	import matplotlib.pyplot as plt
# 	if axis is None:
# 		f,axis = plt.subplots(1)
#
# 	posterior = ensemble_bayes_acorr(d)
# 	ci = credible_interval(posterior)
# 	t = np.arange(posterior[0].size)
#
# 	if normalize:
# 		norm = posterior[0][0]
# 	else:
# 		norm = 1.
#
# 	axis.fill_between(t, ci[0]/norm, ci[1]/norm, alpha=.3, color=color)
# 	axis.plot(t, posterior[0]/norm, color=color, lw=1., alpha=.9)

# ## tests
# if __name__ == '__main__':
# 	from scipy.ndimage import median_filter
# 	import matplotlib.pyplot as plt
#
# 	n = 1000
# 	dd = np.zeros((10,n))
# 	for i in range(dd.shape[0]):
# 		d = np.random.normal(size=n).cumsum()
# 		d = d - median_filter(d,n/100)
# 		l = np.random.randint(low=0,high=n)
# 		d[:l] += np.nan
# 		h = np.random.randint(low=l, high=n)
# 		# print l,h
# 		d[h:] += np.nan
# 		dd[i] = d - np.nanmean(d)
#
# 	plot_bayes_acorr(dd)
# 	for i in range(dd.shape[0]):
# 		y = acorr(dd[i])
# 		plt.plot(y/y[0],'r',lw=.5,alpha=.1)
#
# 	# plt.xscale('log')
# 	plt.xlim(0.0,100.)
# 	plt.ylim(-1,1)
# 	plt.show()

################################################################################

#### Following Kaufman Lab - Mackowiak JCP 2009
@nb.njit
def stretched_exp(t,k,t0,b):
	if  b < 0 or t0 <= 0:
		return t*0 + np.inf
	# if t0 >= t[1]:
	else:
		return k*np.exp(-(t/t0)**b)
	# else:
		# q = np.zeros_like(t+k)
	# 	q[0] = 1.
		# return q
@nb.njit
def single_exp(t,k,t0):
	if t0 <= 0:
		return t*0 + np.inf
	else:
		return k*np.exp(-t/t0)

@nb.njit
def bi_exp(t,k1,k2,t1,t2):
	if t1 <= 0 or t2 <= 0:
		return t*0 + np.inf
	else:
		return k1*np.exp(-t/t1) + k2*np.exp(-t/t2)

@nb.njit
def linearfxn(t,k,t0):
	q = k*(1.-t/t0)
	q[q<0] = 0.
	q[t == 0] = 1.
	return q

@nb.njit
def flatfxn(t,c,cut):
	return (t<cut)*c

################################################################################

@nb.njit
def minfxn_stretch(t,y,x):
	k,t0,b = x
	f = stretched_exp(t[0],k,t0,b)
	tc = t0/b*vgamma(1./b)
	if b < 0. or tc < t[0]/2.:
		return np.inf
	return np.sum(np.square(stretched_exp(t,k,t0,b) - y)/(1.+t))
	# return np.sum(np.square(stretched_exp(t,k,t0,b) - y)/(1.+t**.5))

@nb.njit
def minfxn_single(t,y,x):
	k,t0 = x
	f = single_exp(t[0],k,t0)
	if f > 2.0 or f < 0.0 or t0 >= y.size*2. or t0 < 1.:
		return np.inf
	return np.sum(np.square(single_exp(t,k,t0) - y)/(1.+t))
	# return np.sum(np.square(single_exp(t,k,t0) - y)/(1.+t**.5))

@nb.njit
def minfxn_bi(t,y,x):
	k1,k2,t1,t2 = x
	f = bi_exp(t[0],k1,k2,t1,t2)
	if f > 2.0 or f < 0.0 or t1 >= y.size*2. or t2 >= y.size*2. or k1 < 0 or k2 < 0:
		return np.inf
	return np.sum(np.square(bi_exp(t,k1,k2,t1,t2) - y)/(1.+t))
	# return np.sum(np.square(bi_exp(t,k1,k2,t1,t2) - y)/(1.+t**.5))

@nb.njit
def minfxn_linear(t,y,x):
	k,t0 = x
	if t0 < 0:
		return np.inf
	return np.sum(np.square(linearfxn(t,k,t0) - y)/(1.+t))
	# return np.sum(np.square(linearfxn(t,k,t0) - y)/(1.+t**.5))

@nb.njit
def minfxn_flat(t,y,x):
	c,cut = x
	return np.sum(np.square(flatfxn(t,c,cut)-y))
################################################################################

class obj(object): ## generic class to take anything you throw at it...
	def __init__(self,*args):
		self.args = args

class fit_solution(obj):
	def __call__(self,t):
		if not self.fxn is None and not self.params is None:
			return self.fxn(t,*self.params)

################################################################################

class fit_flat(fit_solution):
	def __init__(self,t,y):
		self.type = "flat"
		self.fxn = flatfxn
		self.params = None
		self.tau = 1.
		self.tc = 0.
		self.r2 = 0.
		self.fit(t,y)

	def set(self,c):
		self.params = np.array((c))

	def fit(self,t,y):
		x0 = np.array((np.nanmean(y[:10]),10))

		self.fit_result = minimize(lambda x: minfxn_flat(t,y,x),x0,method='Nelder-Mead',options={'maxiter':1000})

		if self.fit_result.success:
			self.params = self.fit_result.x
			self.func_val = self.fit_result.fun
			self.tc = self.params[1]*self.params[0]
		else:
			self.params = None
			self.func_val = np.inf
			self.tc = 0.

	def calc_tc(self):
		return self.tc*self.tau

	def __str__(self):
		return r"$c=%.3f, t_cut=%.3f$"%(self.params[0],self.params[1]*self.tau)

class fit_exponential(fit_solution):
	def __init__(self,t,y):
		self.type = "single exponential"
		self.fxn = single_exp
		self.params = None
		self.fit(t,y)
		self.tau = 1.

	def set(self,k,t0):
		self.params = np.array((k,t0))

	def fit(self,t,y,x0=None):
		if x0 is None:
			dt = t[1]-t[0]
			m = np.max((dt,(y*t).sum()/y.sum()))
			x0 = np.array((y[0],m))

		self.fit_result = minimize(lambda x: minfxn_single(t,y,x),x0,method='Nelder-Mead',options={'maxiter':1000})

		if self.fit_result.success:
			self.params = self.fit_result.x
			self.func_val = self.fit_result.fun
		else:
			self.params = None
			self.func_val = np.inf

	def calc_tc(self):
		if not self.params is None:
			return self.params[1]*self.tau
		return 0.

	def __str__(self):
		return r"$k=%.3f, t_0=%.3f$"%(self.params[0],self.tau*self.params[1])

class fit_linear(fit_solution):
	def __init__(self,t,y):
		self.type = "single exponential linear"
		self.fxn = linearfxn
		self.params = None
		self.fit(t,y)
		self.tau = 1.

	def set(self,k,t0):
		self.params = np.array((k,t0))

	def fit(self,t,y,x0=None):
		if x0 is None:
			dt = t[1]-t[0]
			if np.any(y<0):
				m = t[np.nonzero(y<0)[0][0]]
			else:
				m = np.max((dt,(y*t).sum()/y.sum()))
			x0 = np.array((y[0],m))

		self.fit_result = minimize(lambda x: minfxn_linear(t,y,x),x0,method='Nelder-Mead',options={'maxiter':1000})

		if self.fit_result.success:
			self.params = self.fit_result.x
			self.func_val = self.fit_result.fun
		else:
			self.params = None
			self.func_val = np.inf

	def calc_tc(self):
		if not self.params is None:
			return self.params[1]*self.tau
		return 0.

	def __str__(self):
		return r"$k=%.3f, t_0=%.3f$"%(self.params[0],self.tau*self.params[1])


class fit_biexponential(fit_solution):
	def __init__(self,t,y):
		self.type = "bi exponential"
		self.fxn = bi_exp
		self.params = None
		self.fit(t,y)
		self.tau = 1.

	def set(self,k1,k2,t1,t2):
		self.params = np.array((k1,k2,t1,t2))

	def fit(self,t,y,x0=None):
		if x0 is None:
			## initial guess from a single exponential fit
			dt = t[1]-t[0]
			m = np.max((dt,(y*t).sum()/y.sum()))
			x0 = np.array((y[0],m))
			out = minimize(lambda x: minfxn_single(t,y,x),x0,method='Nelder-Mead',options={'maxiter':1000})
			if out.success:
				x0 = np.array((.5,.5,out.x[1],5.))
			else:
				x0 = np.array((y[0]*.8,y[0]*.2,m,5.))

		self.fit_result = minimize(lambda x: minfxn_bi(t,y,x),x0,method='Nelder-Mead',options={'maxiter':1000})

		if self.fit_result.success:
			p = self.fit_result.x.copy()
			if p[2] > p[3]:
				p[0] = self.fit_result.x[1]
				p[1] = self.fit_result.x[0]
				p[2] = self.fit_result.x[3]
				p[3] = self.fit_result.x[2]
			self.params = p
			self.func_val = self.fit_result.fun
		else:
			self.params = None
			self.func_val = np.inf

	def calc_tc(self):
		if not self.params is None:
			k1,k2,t1,t2 = self.params
			f = k1/(k1+k2)
			return (f*t1 + (1.-f)*t2)*self.tau
		return 0.

	def __str__(self):
		return r"$k_1=%.3f, k_2=%.3f, t_1=%.3f, t_2=%.3f$"%(self.params[0],self.params[1],self.tau*self.params[2],self.tau*self.params3)

class fit_stretched(fit_solution):
	def __init__(self,t,y):
		self.type = "stretched exponential"
		self.fxn = stretched_exp
		self.params = None
		self.fit(t,y)
		self.tau = 1.

	def set(self,k,t,b):
		self.params = np.array((k,t,b))

	def fit(self,t,y,x0=None):
		if x0 is None:
			dt = t[1]-t[0]
			m = np.max((1.,(y*t).sum()/y.sum()))
			x0 = np.array((y[0],m,1.))

		self.fit_result = minimize(lambda x: minfxn_stretch(t,y,x),x0,method='Nelder-Mead',options={'maxiter':10000})

		if self.fit_result.success:
			self.params = self.fit_result.x
			self.func_val = self.fit_result.fun

		else:
			self.params = None
			self.tau = np.nan
			self.func_val = np.inf

	def calc_tc(self):
		if not self.params is None:
			k,t,b = self.params
			return t/b*vgamma(1./b)*self.tau
		return 0.

	def __str__(self):
		return r"$k=%.3f, t_0=%.3f, \beta =%.3f $"%(self.params[0],self.params[1]*self.tau,self.params[2])

################################################################################

def fit_acf(t,y,ymin = 0.05,biexp=False):
	yy = y<ymin
	if np.any(yy > 0):
		cutoff = np.argmax(yy)
		if cutoff > y.size:
			cutoff = -1
	else:
		cutoff = -1
	start = 1

	if y[start:cutoff].size > 3:
		ytot = np.sum(np.square(y[start:cutoff] - y[start:cutoff].nanmean()))
		f1 = fit_flat(t[start:cutoff],y[start:cutoff])
		r2 = 1. - f1.func_val/ytot
		print('1',r2,f1.func_val)
		if r2 > .95 and r2 <= 1.0:
			return f1
		f2 = fit_linear(t[start:cutoff],y[start:cutoff])
		r2 = 1. - f2.func_val/ytot
		print('2',r2,f2.func_val)
		if r2 > .95 and r2 <= 1.0:
			return f2
		f3 = fit_stretched(t[start:cutoff],y[start:cutoff])
		r2 = 1. - f3.func_val/ytot
		print('3',r2,f3.func_val)
		if r2 > .95 and r2 <= 1.0:
			return f3
		# ff = np.array((f1.func_val,f2.func_val,f3.func_val))
		# if np.any(np.isfinite(ff)):
		# 	f = [f1,f2,f3][ff.argmin()]
		# 	# if f.calc_tc() <= t.max():
		# 		# return f
		# 	return f
		# f2 = fit_linear(t[start:cutoff],y[start:cutoff])
		# f3 = fit_stretched(t[start:cutoff],y[start:cutoff])
		# f3.calc_tc()
		# if f3.params is None:
			# f3 = fit_stretched(t[:5],y[:5]) ## AH HA! This is the culprit
			# f3.calc_tc()
		# return f3
	# f3 = fit_stretched(t[:5],y[:5]) ## AH HA! This is the culprit
	# f3.calc_tc()
	# return fit_linear(t[1:5],y[1:5])
	return None
	# return fit_flat(t,y)

################################################################################

def power_spec(t,y):
	dt = t[1]-t[0]
	f = np.fft.fft(y)*dt/np.sqrt(2.*np.pi)
	w = np.fft.fftfreq(t.size)*2.*np.pi/dt
	# f /= f[0] ## normalize to zero frequency
	x = w.argsort()
	return w[x],np.abs(f)[x]

def S_exp(w,k):
	analytic = np.sqrt(2./np.pi)*k/(k**2.+w**2.)
	return analytic
