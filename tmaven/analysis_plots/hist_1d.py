import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from PyQt5.QtWidgets import QPushButton,QApplication
# import multiprocessing as mp

class obj(object): ## generic class to take anything you throw at it...
	def __init__(self,*args):
		self.args = args

default_prefs = {
	'fig_height':2.5,
	'fig_width':2.5,
	'subplots_top':0.98,
	'subplots_left':0.17,
	'subplots_right':0.97,
	'subplots_bottom':0.22,
	'xlabel_offset':-0.15,
	'ylabel_offset':-0.2,

	'fret_min':-.25,
	'fret_max':1.25,
	'fret_nbins':151,
	'fret_clip_low':-1.,
	'fret_clip_high':2.,
	'fret_nticks':6,

	'hist_on':True,
	'hist_type':'stepfilled',
	'hist_color':'tab:blue',
	'hist_edgecolor':'tab:blue',
	'hist_log_y':False,
	'hist_force_ymax':False,
	'hist_ymax':5.0,
	'hist_ymin':0.0,
	'hist_nticks':5,

	'kde_bandwidth':0.01,

	'gmm_on':True,
	'gmm_nrestarts':4,
	'gmm_threshold':1e-10,
	'gmm_maxiters':1000,

	'hmm_on':True,
	'idealized':True,
	'hmm_states':False,

	# 'vb_maxstates':8,
	# 'prior_vb_beta':0.25,
	# 'prior_vb_a':2.5,
	# 'prior_vb_b':0.01,
	# 'prior_vb_alpha':1.,

	# 'ncpu':mp.cpu_count(),
	# 'filter':False,

	'textbox_x':0.965,
	'textbox_y':0.9,
	'textbox_fontsize':7.0,
	'textbox_nmol':True,

	'xlabel_text':r'E$_{\rm{FRET}}$',
	'ylabel_text':r'Probability',

	# 'biasd_on':True,
	# 'biasd_tau':1.,
}

def get_nstates(gui):
	from PyQt5.QtWidgets import QInputDialog
	nstates,success = QInputDialog.getInt(gui,"Number of States","Number of States",min=1)
	return success,nstates

def setup(window):
	''' Setup the 1Dhist `popout_plot_container` window

	Adds buttons and then runs `recalc(window)`

	Parameters
	----------
	window - `the popout_plot_container`
	'''

	# fithistbutton = QPushButton("Fit Hist")
	# window.buttonbox.insertWidget(2,fithistbutton)
	# fithistbutton.clicked.connect(lambda x: fit_hist(gui))

	# window.biasdfitbutton = QPushButton("BIASD Fit")
	# window.buttonbox.insertWidget(2,window.biasdfitbutton)
	# window.biasdfitbutton.clicked.connect(lambda x: fit_biasd(window))
	window.prefs.add_dictionary(default_prefs)
	window.fitmlbutton = QPushButton("ML gmm Fit")
	window.buttonbox.insertWidget(2,window.fitmlbutton)
	window.fitmlbutton.clicked.connect(lambda x: fit_ml(window))

	window.recalcbutton = QPushButton("Recalculate")
	window.buttonbox.insertWidget(1,window.recalcbutton)
	window.recalcbutton.clicked.connect(lambda x: recalc(window))

	window.gmm_result = None
	window.biasd_result = None
	window.hist = obj()

	recalc(window)

# def normal_mixture(t,x0,nstates):
# 	q = np.zeros_like(t)
# 	for i in range(nstates-1):
# 		q += x0[3*i+2]*normal(t,x0[3*i+0],x0[3,i+1])
# 	q += (1.-x0[2::3].sum())*normal(t,x0[-2],x0[-1])
# 	return q
#
# def minfxn(t,y,x0,nstates):
# 	if x0[2::3].sum() > 1. or np.any(x0[2::3] < 0):
# 		return np.inf
# 	q = normal_mixture(t,x0,nstates)
# 	return np.sum(np.square(q-y))
#
# def fit_hist(gui):
# 	from scipy.optimize import minimize
# 	prefs = window.prefs
#
# 	success,nstates = gui.data.get_nstates()
# 	if success:
# 		hx = .5*(window.hist.x[1:]+window.hist.x[:-1])
# 		fxn = lambda x0: minfxn(hx,window.hist.y,x0,nstates)
# 		x0 = np.zeros(nstates*3-1)
# 		delta = (prefs['fret_max']-prefs['fret_min'])
# 		x0[::3] = delta*(np.arange(nstates)+1)/(nstates+2.) + prefs['fret_min']
# 		x0[1::3] = delta/nstates
# 		x0[2::3] = 1./nstates
# 		out = minimize(fxn,x0=x0,method='Nelder-mead',options={'maxiters':1000})
# 		if out.success:
# 			t = np.linspace(prefs['fret_min'],prefs['fret_max'],1000)
# 			y = normal_mixture(t,out.x,nstates)
# 		window.ax.plot(t,y,color='r',lw=1)
# 		window.f.draw()

# def add_path(path):
	# import sys
	# if not path in sys.path:
		# sys.path.end(path)

# def fit_biasd(window):
# 	window.gui.maven.error.emit('fit biasd not re-implemented')
# 	# if not window.gui.maven.data.corrected is None:
# 	# 	try:
# 	# 		import biasd as b
# 	# 	except:
# 	# 		return
# 	#
# 	# 	prefs = window.prefs
# 	#
# 	# 	fpb = window.fpb.flatten()
# 	# 	fpb = fpb[np.isfinite(fpb)]
# 	# 	keep = np.bitwise_and((fpb > prefs['fret_clip_low']),(fpb < prefs['fret_clip_high']))
# 	# 	fpb = fpb[keep]
# 	#
# 	# 	p,c = b.likelihood.fit_histogram(fpb,prefs['biasd_tau'],minmax=(prefs['fret_min'],prefs['fret_max']))
# 	# 	print(p)
# 	# 	window.biasd_result = p

def fit_vb(window):
	window.gui.maven.error.emit('fit vb not re-implemented')
	# if not window.gui.maven.data.corrected is None:
	# 	# window.window.gui.maven.statusbar.showMessage('Compiling...')
	# 	QApplication.instance().processEvents()
	# 	from ..supporting.hmms.vb_em_gmm import vb_em_gmm,vb_em_gmm_parallel
	# 	# window.window.gui.maven.statusbar.showMessage('')
	#
	# 	prefs = window.prefs
	#
	# 	fpb = window.fpb.flatten()
	# 	fpb = fpb[np.isfinite(fpb)]
	# 	bad = np.bitwise_or((fpb < prefs['fret_clip_low']),(fpb > prefs['fret_clip_high']))
	# 	fpb[bad] = np.random.uniform(low=prefs['fret_clip_low'],high=prefs['fret_clip_high'],size=int(bad.sum())) ## clip
	#
	# 	ll = np.zeros(prefs['vb_maxstates'])
	# 	rs = [None for _ in range(ll.size)]
	#
	# 	from ..main.ui_progressbar import progressbar
	# 	prog = progressbar()
	# 	prog.setRange(0,ll.size)
	# 	prog.setWindowTitle('VB gmm Progress')
	# 	prog.setLabelText('Number of States')
	# 	# self.flag_running = True
	# 	# prog.canceled.connect(self._cancel_run)
	# 	prog.show()
	#
	# 	priors = np.array([window.gui.maven.prefs[sss] for sss in ['prior_vb_beta','prior_vb_a','prior_vb_b','prior_vb_alpha']])
	#
	# 	for i in range(ll.size):
	# 		r = vb_em_gmm_parallel(fpb,i+1,maxiters=prefs['gmm_maxiters'],threshold=prefs['gmm_threshold'],nrestarts=prefs['gmm_nrestarts'],prior_strengths=priors,ncpu=prefs['ncpu'])
	# 		rs[i] = r
	# 		ll[i] = r.likelihood[-1,0]
	# 		prog.setValue(i+1)
	# 		QApplication.instance().processEvents()
	#
	# 	if np.all(np.isnan(ll)):
	# 		window.gmm_result = None
	# 		window.gui.maven.log.emit("VB gmm failed: all lowerbounds are NaNs")
	# 	else:
	# 		n = np.nanargmax(ll)
	# 		r = rs[n]
	# 		r.type = 'vb'
	# 		window.gmm_result = r
	#
	# 		recalc(window)
	# 		# plot(window)
	#
	# 		window.gui.maven.log.emit(r.report())

def fit_ml(window):
	''' Fit the FRET data using MLE

	Fits the FRET data using MLE to a mixture model of nstates -1 gaussians and one "junk" uniform distribution. Prompts user for number of states. JIT compiles the MLE function upon first run, so might be slow

	Parameters
	----------
	window : popout_plot_container
		the `gui.plots.hist1d` container

	Notes
	----------
	prefs['fret_clip_low'] : double
		remove any FRET data < this value. Bottom of uniform dist for "junk" class
	prefs['fret_clip_high'] : double
		remove any FRET data > this value. Top of uniform dist for "junk" class

	'''

	if not window.gui.maven.data.nmol == 0:
		# window.gui.maven.statusbar.showMessage('Compiling...')
		QApplication.instance().processEvents()
		from ..modeler.hmms.ml_em_gmm import ml_em_gmm
		# window.gui.maven.statusbar.showMessage('')

		prefs = window.prefs

		success,nstates = get_nstates(window.gui)
		if success:
			fpb = window.fpb.flatten()
			fpb = fpb[np.isfinite(fpb)]
			bad = np.bitwise_or((fpb < prefs['fret_clip_low']),(fpb > prefs['fret_clip_high']))
			fpb[bad] = np.random.uniform(low=prefs['fret_clip_low'],high=prefs['fret_clip_high'],size=int(bad.sum())) ## clip

			r = ml_em_gmm(fpb,nstates+1,maxiters=1000,threshold=1e-6)
			r.type = 'ml'
			window.gmm_result = r

			recalc(window)
			plot(window)

			window.gui.maven.log.emit('1D GMM Results:\n'+r.report())

# def draw_biasd(window):
# 	try:
# 		import biasd as b
# 	except:
# 		return
#
# 	p = window.biasd_result
# 	x = np.linspace(window.prefs['fret_min'],window.prefs['fret_max'],1000)
# 	y = (1.-p[-1])/(x[1]-x[0]) + p[-1]*np.exp(b.likelihood.nosum_log_likelihood(p[:-1],x,window.prefs['biasd_tau'],device=None))
#
# 	window.plot.ax.plot(x,y,color='k',lw=2,alpha=.8)
# 	window.plot.draw()


def draw_gmm(window):
	''' draw GMM likelihood for FRET on window.plot.ax

	Draws on gaussian for each class, and a uniform for junk

	Parameters
	----------
	window : popout_plot_container
		the `gui.plots.hist1d` container

	'''

	try:
		r = window.gmm_result
		x = window.gmm_x
		tot = window.gmm_tot
		ys = window.gmm_ys

		if r.type == 'vb':
			for i in range(len(ys)):
				y = ys[i]
				window.plot.ax.plot(x,y,color='k',lw=1,alpha=.8,ls='--')
		elif r.type == 'ml':
			for i in range(len(ys) - 1): ## ignore the outlier class
				y = ys[i]
				window.plot.ax.plot(x,y,color='k',lw=1,alpha=.8,ls='--')
			y = ys[-1]
			window.plot.ax.plot(x,y,color='k',lw=1,alpha=.8,ls='--')

		window.plot.ax.plot(x,tot,color='k',lw=2,alpha=.8)
		window.plot.draw()
	except:
		pass

def draw_hmm(window):
	''' Draw the HMM likelihood for FRET

	If it's a consensus vb HMM, then you know where the classes are, so can plot those.

	Parameters
	----------
	window : popout_plot_container
		the `gui.plots.hist1d` container

	'''
	try:
		r = window.gui.maven.modeler.model
		x = window.hmm_x
		tot = window.hmm_tot
		ys = window.hmm_ys

		if r.type == 'vb Consensus HMM':
			for y in ys:
				window.plot.ax.plot(x,y,color='k',lw=1,alpha=.8,ls='--')

		window.plot.ax.plot(x,tot,color='k',lw=2,alpha=.8)
		window.plot.draw()
	except:
		pass

def normal(x,m,v):
	''' Normal PDF

	Parameters
	----------
	x : np.ndarray
		data
	m :
		mu of gaussian
	v :
		variance of gaussian

	Returns
	-------
	pdf : np.ndarray like x
	'''
	return 1./np.sqrt(2.*np.pi*v)*np.exp(-.5/v*(x-m)**2.)

def studentt(x,a,b,k,m):
	''' Student's T PDF

	Parameters
	----------
	x : np.ndarray
		data
	a :
		a of student t
	b :
		b of student t
	k :
		degrees of freedom of student t
	m :
		m of student t

	Returns
	-------
	pdf : np.ndarray like x

	'''
	from scipy.special import gammaln
	lam = a*k/(b*(k+1.))
	lny = -.5*np.log(np.pi)
	lny += gammaln((2.*a+1.)/2.) - gammaln((a))
	lny += .5*np.log(lam/(2.*a))
	lny += -.5*(2.*a+1)*np.log(1.+lam*(x-m)**2./(2.*a))
	return np.exp(lny)


def kde(x,d,bw=None):
	''' Kernel Density Estimation (KDE)

	Uses implementation from
		* sklearn.neighbors import KernelDensity, and then
		* from scipy.stats import gaussian_kde if no sklearn

	Parameters
	----------
	x : np.ndarray
		x data to KDE at
	d : np.ndarray
		data to make KDE estimate of
	bw : float
		bandwidth of KDE

	Returns
	-------
	y : np.ndarray
		estimate y for x, shaped like x
	'''
	try:
		from sklearn.neighbors import KernelDensity
		if bw is None:
			bw = (0.+d.size)**(-1./5)
		_kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(d[:,None])
		y = np.exp(_kde.score_samples(x[:,None])).flatten()
		return y
	except:
		from scipy.stats import gaussian_kde
		if bw is None:
			kernel = gaussian_kde(d)
		else:
			kernel = gaussian_kde(d,bw)
		y = kernel(x)
		return y


def recalc(window):
	''' Calculate histograms and marginal lines

	This function will recalculate all the data plotted in `plot()`. Calls `plot()` once completed

	Parameters
	----------
	window : popout_plot_container
		the `gui.plots.hist1d` container

	Notes
	------
	prefs['idealized'] : bool
		* True - use idealized data from `window.gui.maven.modeler.model`
		* False - use data from `window.gui.maven.data.corrected`
	prefs['fret_min'] : double
		min FRET value for histogram
	prefs['fret_max'] : double
		max FRET value for histogram
	prefs['fret_nbins'] : int
		number of bins in FRET histogram
	prefs['kde_bandwidth'] : double
		if not zero, perform KDE with this as the bandwidth

	'''
	if window.gui.maven.data.nmol == 0:
		return
	## Data
	if window.prefs['idealized'] and not window.gui.maven.modeler.model is None:
		window.fpb = window.gui.analysis_plots.get_idealized_data(signal=True)
	else:
		window.fpb = window.gui.analysis_plots.get_plot_fret()[:,:,1].copy()

	## GMM
	if not window.gmm_result is None:
		r = window.gmm_result
		x = np.linspace(window.prefs['fret_min'],window.prefs['fret_max'],1001)

		tot = np.zeros_like(x)
		ys = []
		if 0:
			pass
		# if r.type == 'vb':
		# 	for i in range(r.mu.size):
		# 		y = r.ppi[i]*studentt(x,r.a[i],r.b[i],r.beta[i],r.m[i])
		# 		tot += y
		# 		ys.append(y)
		elif r.type == 'ml':
			for i in range(r.mu.size - 1): ## ignore the outlier class
				y = r.ppi[i]*normal(x,r.mu[i],r.var[i])
				tot += y
				ys.append(y)
			y = r.ppi[-1]*(x*0. + r.mu[-1])
			tot += y
			ys.append(y)

		window.gmm_x = x
		window.gmm_tot = tot
		window.gmm_ys = ys

	## HMM -- this needs to be standardized... kind of a mess
	if not window.gui.maven.modeler.model is None:
		r = window.gui.maven.modeler.model

		x = np.linspace(window.prefs['fret_min'],window.prefs['fret_max'],1001)
		tot = np.zeros_like(x)
		ys = []
		if r.type == 'vb Consensus HMM':
			rr = r#r.result
			for i in range(rr.mu.size):
				if window.prefs['hmm_states']:
					y = rr.ppi[i]*normal(x,rr.mu[i],1./rr.beta[i])
				else:
					y = rr.ppi[i]*studentt(x,rr.a[i],rr.b[i],rr.beta[i],rr.mu[i])
				tot += y
				ys.append(y)
		elif r.type in ['vb','ml','threshold']:
			nn = 0.
			if not window.prefs['hmm_states'] and not r.type in ['threshold']:
				for j in range(len(r.results)):
					rr = r.results[j]
					nn += rr.r.shape[0]
					for i in range(rr.mu.size):
						if r.type == 'ml':
							tot += rr.ppi[i]*normal(x,rr.mu[i],rr.var[i]) * rr.r.shape[0]
						else:
							tot += rr.ppi[i]*studentt(x,rr.a[i],rr.b[i],rr.beta[i],rr.mu[i]) * rr.r.shape[0]
				tot /= nn
			else:
				if window.prefs['idealized']:
					v = window.gui.analysis_plots.get_idealized_data(signal=True).flatten()
					v = v[np.isfinite(v)]
					if window.prefs['kde_bandwidth'] != 0.0:
						try:
							tot = kde(x,v,window.prefs['kde_bandwidth'])
						except:
							tot = kde(x,v)
					else:
						pp = window.prefs
						tot = kde(x,v,(pp['fret_max']-pp['fret_min'])/(1.+pp['fret_nbins']))

				elif not r.type in ['threshold']:
					for j in range(len(r.results)):
						rr = r.results[j]
						nn += rr.r.shape[0]
						for i in range(rr.mu.size):
							if r.type == 'ml':
								tot += rr.ppi[i]*normal(x,rr.mu[i],rr.var[i]/np.sqrt(rr.ppi[i]*rr.r.shape[0])) * rr.r.shape[0]
							else:
								tot += rr.ppi[i]*normal(x,rr.m[i],1./rr.beta[i]) * rr.r.shape[0]
					tot /= nn

		window.hmm_x = x
		window.hmm_tot = tot
		window.hmm_ys = ys
	plot(window)


def plot(window):
	''' Plots all lines calculated in `recalc`

	Parameters
	----------
	window : popout_plot_container
		the `gui.plots.hist1d` container

	Notes
	-----
	prefs['hist_on'] : bool
		whether to plot histogram or not
	prefs['hist_color'] : str
		color of histogram
	prefs['hist_edgecolor'] : str
		color histogram edges
	prefs['fret_min'] : double
		min FRET value for histogram
	prefs['fret_max'] : double
		max FRET value for histogram
	prefs['fret_nbins'] : int
		number of bins in FRET histogram
	prefs['histtype'] : str
		matplotlib histogram type
	prefs['hist_log_y'] : bool
		whether to log10 scale the y axis of the histogram
	prefs['hmm_on'] : bool
		whether to plot the HMM overlay
	prefs['gmm_on'] : bool
		whether to plot the GMM overlay
	prefs['hist_force_ymax'] : bool
		whether to use ymin,ymax values in preferences or autoscale
	prefs['hist_ymin'] : double
		minimum y axis value
	prefs['hist_ymax'] : double
		maximum y axis value
	prefs['hist_nticks'] : int
		number of ticks on y axis
	prefs['fret_nticks'] : int
		number of ticks on x axis

	several more...

	'''
	if window.gui.maven.data.nmol == 0:
		return
	if window.fpb is None:
		return
	pp = window.prefs
	window.plot.resize_figure()
	window.plot.cla()
	QApplication.instance().processEvents()

	if window.gui.maven.data.ncolors == 2:
		fpb = window.fpb.flatten()
		fpb = fpb[np.isfinite(fpb)]

		if pp['hist_on']:
			color = pp['hist_color']
			if not colors.is_color_like(color):
				color = 'steelblue'
			ecolor = pp['hist_edgecolor']
			if not colors.is_color_like(ecolor):
				ecolor = 'black'

			window.hist.y, window.hist.x = window.plot.ax.hist(fpb,bins=pp['fret_nbins'], range=(pp['fret_min'],pp['fret_max']),histtype=pp['hist_type'], alpha=.8,density=True,color=color, edgecolor=ecolor, log=pp['hist_log_y'])[:2]
		# else:
		# 	if pp['hist_log_y']:
		# 		window.plot.ax.set_yscale('log')

		ylim = window.plot.ax.get_ylim()
		if pp['hmm_on'] and not window.gui.maven.modeler.model is None:
			draw_hmm(window)
		if pp['gmm_on'] and not window.gmm_result is None:
			draw_gmm(window)
		# if pp['biasd_on'] and not window.biasd_result is None:
			# draw_biasd(window)
		window.plot.ax.set_xlim(pp['fret_min'],pp['fret_max'])
		window.plot.ax.set_ylim(*ylim) ## incase modeling gave crazy results

		if not pp['hist_log_y']:
			if pp['hist_force_ymax']:
				window.plot.ax.set_ylim(pp['hist_ymin'],pp['hist_ymax'])
				ticks = window.plot.best_ticks(pp['hist_ymin'],pp['hist_ymax'],pp['hist_nticks'])
			else:
				ticks = window.plot.best_ticks(0,window.plot.ax.get_ylim()[1],pp['hist_nticks'])
			window.plot.ax.set_yticks(ticks)


		ticks = window.plot.best_ticks(pp['fret_min'],pp['fret_max'],pp['fret_nticks'])
		window.plot.ax.set_xticks(ticks)

		dpr = window.plot.devicePixelRatio()

		fs = pp['label_fontsize']/dpr
		font = {
			'family': pp['font'],
			'size': fs,
			'va':'top'
		}

		window.plot.ax.set_xlabel(pp['xlabel_text'],fontdict=font)
		window.plot.ax.set_ylabel(pp['ylabel_text'],fontdict=font)

		window.plot.ax.yaxis.set_label_coords(pp['ylabel_offset'], 0.5)
		window.plot.ax.xaxis.set_label_coords(0.5, pp['xlabel_offset'])

		bbox_props = dict(boxstyle="square", fc="w", alpha=1.0,lw=1./dpr)
		lstr = 'N = %d'%(window.fpb.shape[0])

		window.plot.ax.annotate(lstr,xy=(pp['textbox_x'],pp['textbox_y']),xycoords='axes fraction',ha='right',color='k',bbox=bbox_props,fontsize=pp['textbox_fontsize']/dpr,family=pp['font'])

		window.plot.draw()
