import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QPushButton, QComboBox, QWidget,QApplication
from PyQt5.QtCore import Qt
from matplotlib import ticker
from ..ui_plots import tryexcept

default_prefs = {
'subplots_left':.18,
'subplots_right':.95,
'subplots_bottom':.2,
'subplots_top':.9,
'fig_width':4.0,
'fig_height':2.5,
'xlabel_offset':-.15,
'ylabel_offset':-.2,

'time_scale':'log',
'time_dt':1.0,
'time_nticks':5,
'time_min':0.0,
'time_max':2000.0,

'acorr_nticks':6,
'acorr_min':-0.1,
'acorr_max':1.0,
'acorr_method':False,

'power_nticks':6,
'power_min':.1,
'power_max':100.0,

'line_color':'blue',
'line_linewidth':1,
'line_alpha':0.9,

'fill_alpha':0.3,

'xlabel_rotate':0.,
'ylabel_rotate':0.,
'xlabel_text1':r'Time(s)',
'ylabel_text1':r'Autocorrelation Function',
'xlabel_text2':r'Frequency (s$^{-1}$)',
'ylabel_text2':r'Power Spectrum',
'xlabel_decimals':2,
'ylabel_decimals':2,

'textbox_x':0.95,
'textbox_y':0.93,
'textbox_fontsize':8,

'line_ind_alpha':.05,
'line_ens_alpha':.9,
'line_hmmcolor':'darkred',

'hist_color':'#0080FF',
'hist_nbins':30,
'hist_kmin':0.,
'hist_kmax':30.,
'hist_pmin':0.,
'hist_pmax':1.,

'kde_bandwidth':.1,
# 'filter_data':False,
# 'filter_ACF':False,
# 'filter_ACF_width':1.,

'show_ens':True,
'show_ind':True,
'show_mean':False,
'show_tc':True,
'show_stretch':True,
'show_hmm':True,
'show_zero':True,
'show_textbox':True,
'remove_viterbi':False,

'tc_cut':-1,

'beta_showens':False,
'beta_showmean':True,
'beta_nbins':41,
'tc_max':-1.,
'tc_min':-1.,
'tc_showgauss':True,
'tc_showkde':True,
'tc_nbins':41,
'tc_showens':False,
'tc_showmean':True,
'tc_fit_ymin':0.1,
'tc_ymax':0.5,
'tc_ynticks':5,
'tc_xnticks':5,
'tc_fitcut':1.,

'acorr_ind':0,
'fit_biexp':False,

'energy_delta_E':  0.1,
'energy_E_min' : 0.,
'energy_E_max' : 30,

}

class obj(object): ## generic class to take anything you throw at it...
	def __init__(self,*args):
		self.args = args

@tryexcept
def setup(window):
	'''setup interface'''
	recalcbutton = QPushButton("Recalculate")
	window.buttonbox.insertWidget(1,recalcbutton)
	recalcbutton.clicked.connect(lambda x: recalc(window))

	window.combo_plot = QComboBox()
	window.combo_plot.addItems(['ACF','Power','Mean','t_c','beta','tc v b','ind acf','energies'])
	window.buttonbox.insertWidget(2,window.combo_plot)
	window.combo_plot.setCurrentIndex(0)

	window.nmol = 0

	pp = window.prefs
	# clear_memory(window)
	# pp.commands['add to memory'] = lambda: add_to_memory(window)
	# pp.commands['clear memory'] = lambda: clear_memory(window)
	# pp.commands['write ind acf'] = lambda: write_ind_acf(window)
	# pp.update_commands()

	fixup_keypress(window)

	# window.filter = None
	if not window.gui.maven.data.nmol == 0:
		recalc(window)

def fixup_keypress(window):
	'''keypress left/right arrows for viewing individual ACFs'''
	def kpe(self,event):
		kk = event.key()
		index = self.combo_plot.currentIndex()
		if index == 6:
			if kk == Qt.Key_Right:
				self.prefs['acorr_ind'] = self.prefs['acorr_ind']+1
				plot(self)
			elif kk == Qt.Key_Left:
				self.prefs['acorr_ind'] = self.prefs['acorr_ind']-1
				plot(self)
		event.accept()

	window.keyPressEvent = lambda event: kpe(window,event)
	window.buttonbox.keyPressEvent = lambda event: kpe(window,event)
	for i in range(window.buttonbox.count()):
		try:
			window.buttonbox.itemAt(i).widget().keyPressEvent = lambda event: kpe(window,event)
		except:
			pass


# def add_to_memory(window):
# 	window.prefs.memory.append([window.ens.t,window.ens.y,None,window.ens.freq,window.ens.fft,window.prefs['line_color']])
#
# def clear_memory(window):
# 	window.prefs.memory = []
#
# def write_ind_acf(window):
# 	from PyQt5.QtWidgets import QFileDialog
#
# 	pp = window.prefs
# 	oname = QFileDialog.getSaveFileName(window, 'Export ACF', '_acf.npy','*.npy')
# 	if oname[0] != "" and not oname[0] is None:
# 		o = np.array((window.ind.t,window.ind.y[pp['acorr_ind']]))
# 		np.save(oname[0],o)
# 		window.gui.maven.log.emit('Saved ACF %d'%(pp['acorr_ind']))
#
#############################################################################
#
# def kde(x,d,bw=None):
# 	from scipy.stats import gaussian_kde
# 	if bw is None:
# 		kernel = gaussian_kde(d)
# 	else:
# 		kernel = gaussian_kde(d,bw)
# 	y = kernel(x)
# 	return y

#############################################################################

# @tryexcept
def recalc(window):
	'''Calculates all ACFs and power spectrum. Performs fitting'''
	pp = window.prefs
	from .autocorr import power_spec,fit_acf
	if pp['acorr_method']:
		from .autocorr import acf_estimator_correct as acf_estimator
	else:
		from .autocorr import acf_estimator_fft as acf_estimator

	if window.gui.maven.data.nmol == 0:
		return

	window.fpb = window.gui.analysis_plots.get_plot_fret()[:,:,-1].copy()
	# ensavg = np.nanmean(window.fpb,axis=0)
	# window.fpb -= ensavg[None,:]
	ensavg = np.nanmean(window.fpb)
	window.fpb -= ensavg
	t = np.arange(window.fpb.shape[1])

	window.ind = obj()
	window.ind.t = t
	window.ind.y = np.zeros_like(window.fpb) + np.nan
	current = 0
	for i in range(window.fpb.shape[0]):
		ff = window.fpb[i]
		ff = ff[np.isfinite(ff)][None,:]
		if ff.size > 1:
			y = acf_estimator(ff)
			window.ind.y[current,:y.size] = y
			current += 1
	window.ind.y = window.ind.y[:current]

	window.ind.fft = np.zeros_like(window.ind.y) + np.nan
	window.ind.freq = np.zeros_like(window.ind.y) + np.nan
	for i in range(window.ind.y.shape[0]):
		keep = np.isfinite(window.ind.y[i])
		yy = window.ind.y[i,keep]
		ft,f = power_spec(window.ind.t[:yy.size],yy)
		window.ind.fft[i,:f.size] = f
		window.ind.freq[i,:ft.size] = ft

	window.ens = obj()
	window.ens.y = np.nanmedian(window.ind.y,axis=0)
	# window.ens.y = acf_estimator(window.fpb)
	keep = np.isfinite(window.ens.y)
	window.ens.y = window.ens.y[keep]
	window.ens.t = t[:window.ens.y.size]

	window.ens.freq,window.ens.fft = power_spec(window.ens.t,window.ens.y)

	window.ens.y /= window.ens.y[0]
	window.ind.y /= window.ind.y[:,0][:,None]

	# window.ens.fit = fit_acf(window.ens.t,window.ens.y,pp['tc_fit_ymin'],False)
	# window.ens.tc = window.ens.fit.calc_tc()
	# if window.ens.fit.type == 'stretched exponential':
		# window.ens.beta = window.ens.fit.params[2]
	# else:
		# window.ens.beta = np.nan



	## get data
	# window.fpb = window.gui.analysis_plots.get_plot_fret()[:,:,-1].copy()
	# window.fpb[np.greater(window.fpb,1.5)] = np.nan ### won't skew ACFs
	# window.fpb[np.less(window.fpb,-.5)] = np.nan ### won't skew ACFs

	# hr = window.gui.maven.modeler.model
	# if not hr is None and pp['remove_viterbi']:
	# 	if hr.type == 'vb Consensus HMM':
	# 		mu = hr.result.mu
	# 		for i in range(window.fpb.shape[0]):
	# 			v = hr.result.viterbi[i]
	# 			pre = window.gui.maven.data.pre_list[hr.ran[i]]
	# 			window.fpb[i,pre:pre+v.size] -= mu[v]
	# 	elif hr.type == 'vb' or hr.type == 'ml':
	# 		for i in range(window.fpb.shape[0]):
	# 			mu = hr.results[i].mu
	# 			v = hr.results[i].viterbi
	# 			pre = window.gui.maven.data.pre_list[hr.ran[i]]
	# 			window.fpb[i,pre:pre+v.size] -= mu[v]

	# baseline = np.nanmean(window.fpb)
	# t = np.arange(window.fpb.shape[1])




	# #### Individual Data
	# ## y - list of ACF means
	# ## t - ACF time
	# ## tc - correlation time
	# ## beta - stretched exponent
	# ## fft - array of power spectra
	# ## freq - power spectrum freqs
	#
	# # # ensavg = np.nanmean(window.fpb)
	# # ensavg = np.nanmedian(window.fpb)
	# ensavg = np.nanmedian(np.array(window.fpb),axis=0)
	# # print(ensavg.shape,window.fpb.shape)
	# # print(ensavg)

	# window.ind = obj()
	# window.ind.y = []
	# for i in range(window.fpb.shape[0]):
	# 	# ff = window.fpb[i][None,:] - ensavg
	# 	ff = (window.fpb[i]- ensavg)[None,:]
	# 	if not np.all(np.isnan(ff)):
	# 		window.ind.y.append(list(acf_estimator(ff)))
	# window.ind.y = np.array(window.ind.y)
	# window.ind.y /= window.ind.y[:,0][:,None]
	# window.ind.t = t

	# window.ind.fft = []
	# window.ind.fit = []
	# window.ind.tc = []
	# window.ind.beta = []
	# for i in range(window.ind.y.shape[0]):
	# 	try:
	# 		ft,f = power_spec(window.ind.t,window.ind.y[i])
	# 		window.ind.fft.append(f)
	# 		fit = fit_acf(window.ind.t,window.ind.y[i],pp['tc_fit_ymin'],pp['fit_biexp'])
	# 		window.ind.fit.append(fit)
	# 		window.ind.tc.append(fit.calc_tc())
	# 		if fit.type == 'stretched exponential':
	# 			window.ind.beta.append(fit.params[2])
	# 		else:
	# 			window.ind.beta.append(np.nan)
	# 			window.ind.tc[-1] = np.nan
	# 	except:
	# 		print('messed up @ ',i)
	# window.ind.freq = ft
	#
	# window.ind.fft = np.array(window.ind.fft)
	# window.ind.tc = np.array(window.ind.tc)
	# window.ind.beta = np.array(window.ind.beta)
	#
	# window.ind.y[np.bitwise_and((window.ind.y != 0), (np.roll(window.ind.y,-1,axis=1)-window.ind.y == 0.))] = np.nan

	#### Ensemble Data
	## y - ACF vs time
	## t - time
	## fft - power spectrum
	## freq - power spectrum frequency axis
	## tc - correlation time
	## beta - stretched parameter

	# window.ens = obj()
	# # window.ens.y = list(acf_estimator(window.fpb))
	# # window.ens.y /= window.ens.y[0] ## filtering can mess it up a little, therefore renormalize
	# window.ens.y = np.nanmean(window.ind.y,axis=0)
	# window.ens.t = t
	# print('ens',window.ens.y.shape,window.ind.y.shape)
	#
	# window.ens.freq,window.ens.fft = power_spec(window.ens.t,window.ens.y)
	#
	# window.ens.fit = fit_acf(window.ens.t,window.ens.y,pp['tc_fit_ymin'],False)
	# window.ens.tc = window.ens.fit.calc_tc()
	# if window.ens.fit.type == 'stretched exponential':
	# 	window.ens.beta = window.ens.fit.params[2]
	# else:
	# 	window.ens.beta = np.nan


	# #### HMM Data
	# ## t - ACF time
	# ## y - ACF
	# ## freq - Power spectrum frequency
	# ## fft - Power spectrum
	# ## tc - correlation time
	# ## beta - stretched exponent
	# hr = window.gui.maven.modeler.model
	# window.hmm = None
	# if not hr is None:
	# 	window.hmm = obj()
	# 	if hr.type == 'vb Consensus HMM':
	# 		from .autocorr import gen_mc_acf
	#
	# 		mu = hr.result.mu
	# 		var = hr.result.var
	# 		tmatrix = hr.result.tmstar
	# 		ppi = hr.result.ppi
	# 		window.hmm.t,window.hmm.y = gen_mc_acf(1.,window.ens.y.size,tmatrix,mu,var,ppi)
	# 		window.hmm.freq,window.hmm.fft = power_spec(window.hmm.t,window.hmm.y)
	# 		window.hmm.fit = fit_acf(window.hmm.t,window.hmm.y,pp['tc_fit_ymin'],False)
	# 		window.hmm.tc = window.hmm.fit.calc_tc()
	# 		if window.hmm.fit.type == 'stretched exponential':
	# 			window.hmm.beta = window.hmm.fit.params[2]
	# 		else:
	# 			window.hmm.beta = np.nan
	#
	# 	# elif hr.type == 'vb':
	# 	# 	window.hmm.y = np.zeros_like(window.ens.y)
	# 	# 	for i in range(window.fpb.shape[0]):
	# 	# 		mu = hr.results[i].mu
	# 	# 		var = hr.results[i].var
	# 	# 		tmatrix = hr.results[i].tmstar
	# 	# 		ppi = hr.results[i].ppi
	# 	# 		t,y = gen_mc_acf(1.,window.ens.y.size,tmatrix,mu,var,ppi)
	# 	# 		window.hmm.y += y/window.fpb.shape[0]
	# 	# 	window.hmm.t = t
	# 	# 	window.hmm.freq,window.hmm.fft = power_spec(window.hmm.t,window.hmm.y)
	# 	# 	window.hmm.tc = np.sum(window.hmm.y)

@tryexcept
def infer_P_E(G_it,c_i = None,tau = 1.,delta_E=.1,E_min=0.,E_max=30.):
	''' MLE for Frauenfelder'''
	nu = 1e13
	R = .001987
	T = 22 + 273.

	# delta_E = 0.1
	# E_min = 0.
	# E_max = 30.
	n_E = int((E_max-E_min)//delta_E + 1)
	E_low = (np.arange(n_E)+0.)*delta_E + E_min
	E_high = (np.arange(n_E)+1.)*delta_E + E_min

	t = (np.arange(G_it.shape[1]) + 0.)*tau
	if c_i is None:
		c_i = np.ones(G_it.shape[0])

	E_it = R*T*np.log(nu*t[None,:]/np.log(c_i[:,None]/G_it))
	E_it = E_it.flatten()

	alpha0 = np.ones(n_E)
	alpha = alpha0 + np.nansum((E_it[None,:] <= E_high[:,None]) * (E_it[None,:] > E_low[:,None]),axis=1)

	return E_low,E_high,alpha



@tryexcept
def plot(window):
	''' Draw the current ACF plot

	Most of the function deals with universal aspects of the plots, regardless of method. Then the separate plot function for the selected method in `combo_plot` is ran
	Parameters
	----------
	window : popout_plot_container
		the `gui.plots.autocorr.autocorr_plot` container

	Notes
	-----
	method 0 : ensemble ACFs
	method 1 : ensemble power spectra
	method 2 : signal mean value versus time
	method 3 : histogram correlation times from fitting
	method 4 : histogram beta values from powerlaw fitting
	method 5 : scatter beta versus tau for powerlaw fitting
	method 6 : individual ACFs
	method 7 : Energy spectrum (Frauenfelder)

	'''
	if window.gui.maven.data.corrected is None:
		return
	window = window
	pp = window.prefs
	window.plot.cla()
	window.plot.resize_figure()
	window.gui.maven.app.processEvents()

	dpr = window.plot.devicePixelRatio()

	method_index = window.combo_plot.currentIndex()

	if method_index == 0:
		plot_autocorrelation(window)
	elif method_index == 1:
		plot_powerspectrum(window)
	elif method_index == 2:
		plot_mean(window)
	elif method_index == 3:
		plot_tc(window)
	elif method_index == 4:
		plot_beta(window)
	elif method_index == 5:
		plot_scatter(window)
	elif method_index == 6:
		plot_indacf(window)
	elif method_index == 7:
		plot_energy(window)
		return

	# ####################################################

	fs = pp['label_fontsize']/dpr
	font = {
		'family': pp['font'],
		'size': fs,
		'va':'top'
	}

	if method_index == 0:
		window.plot.ax.set_xlabel(pp['xlabel_text1'],fontdict=font)
		window.plot.ax.set_ylabel(pp['ylabel_text1'],fontdict=font)
		if not pp['time_scale'] == 'log':
			window.plot.ax.set_xticks(window.plot.best_ticks(pp['time_min'],pp['time_max'],pp['time_nticks']))
		window.plot.ax.set_yticks(window.plot.best_ticks(pp['acorr_min'],pp['acorr_max'],pp['acorr_nticks']))
	elif method_index == 1:
		window.plot.ax.set_xlabel(pp['xlabel_text2'],fontdict=font)
		window.plot.ax.set_ylabel(pp['ylabel_text2'],fontdict=font)
	elif method_index == 2:
		window.plot.ax.set_xlabel(r'Time (sec)',fontdict=font)
		window.plot.ax.set_ylabel(r'Mean',fontdict=font)
	elif method_index == 3:
		# window.plot.ax.set_xlabel(r'$ln(t_c)$',fontdict=font)
		window.plot.ax.set_xlabel(r'$t_c$',fontdict=font)
		window.plot.ax.set_ylabel('Probability',fontdict=font)
		xlim = window.plot.ax.get_xlim()
		window.plot.ax.set_xticks(window.plot.best_ticks(xlim[0],xlim[1],pp['tc_xnticks']))
		window.plot.ax.set_yticks(window.plot.best_ticks(0,pp['tc_ymax'],pp['tc_ynticks']))

	elif method_index == 4:
		window.plot.ax.set_xlabel(r'$\beta$',fontdict=font)
		window.plot.ax.set_ylabel('Counts',fontdict=font)
	elif method_index == 5:
		window.plot.ax.set_ylabel(r'$t_c$',fontdict=font)
		window.plot.ax.set_xlabel(r'$\beta$',fontdict=font)

	window.plot.ax.yaxis.set_label_coords(pp['ylabel_offset'], 0.5)
	window.plot.ax.xaxis.set_label_coords(0.5, pp['xlabel_offset'])

	if pp['show_textbox']:
		bbox_props = dict(boxstyle="square", fc="w", alpha=1.0,lw=1./dpr)
		lstr = r'N = %d'%(window.nmol)
		# print window.ens.tc*pp['time_dt'],np.median(window.ind.tc)*pp['time_dt']

		try:
			if pp['show_tc']:
				lstr += r', $t_c$=%.2f sec'%(np.around(window.ens.tc*pp['time_dt'],2))
		except:
			pass
		window.plot.ax.annotate(lstr,xy=(pp['textbox_x'],pp['textbox_y']),xycoords='axes fraction', ha='right', color='k', bbox=bbox_props, fontsize=pp['textbox_fontsize']/dpr)

	fd = {'rotation':pp['xlabel_rotate'], 'ha':'center'}
	if fd['rotation'] != 0: fd['ha'] = 'right'
	window.plot.ax.set_xticklabels(["{0:.{1}f}".format(x, pp['xlabel_decimals']) for x in window.plot.ax.get_xticks()], fontdict=fd)
	window.plot.ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos: "{0:.{1}f}".format(x,pp['xlabel_decimals'])))

	fd = {'rotation':pp['ylabel_rotate']}
	window.plot.ax.set_yticklabels(["{0:.{1}f}".format(y, pp['ylabel_decimals']) for y in window.plot.ax.get_yticks()], fontdict=fd)
	window.plot.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos: "{0:.{1}f}".format(x,pp['ylabel_decimals'])))
	window.plot.draw()

def plot_energy(window):
	''' Frauenfelder E_activation distribution plot'''

	pp = window.prefs
	ci = window.ind.y[:,1]
	# ci = np.ones(window.ind.y.shape[0]) + 0.
	E_low,E_high,alpha = infer_P_E(window.ind.y[:,1:],ci,pp['time_dt'],pp['energy_delta_E'],pp['energy_E_min'],pp['energy_E_max'])
	x = .5*(E_low+E_high)
	from scipy.special import betaincinv
	beta = alpha.sum() - alpha
	low = betaincinv(alpha,beta,.025)
	mid = betaincinv(alpha,beta,.5)
	high = betaincinv(alpha,beta,.975)

	window.plot.ax.fill_between(x,low,high,alpha=.5)
	window.plot.ax.plot(x,mid,color='k',alpha=.9)
	window.plot.draw()


@tryexcept
def plot_autocorrelation(window):
	'''plot autocorrection function ensemble'''
	pp = window.prefs
	tau = pp['time_dt']

	if pp['show_zero']:
		window.plot.ax.axhline(y=0.,color='k',alpha=.5,lw=1.)

	if pp['show_ind']:
		for i in range(window.ind.y.shape[0]):
			window.plot.ax.plot(window.ind.t*tau, window.ind.y[i], color='k', alpha=pp['line_ind_alpha'])

	## Ensemble plots
	if pp['show_ens']:
		window.plot.ax.plot(window.ens.t*tau, window.ens.y, color=pp['line_color'], lw=1., alpha=pp['line_ens_alpha'],zorder=1)
	# if pp['show_stretch']:
		# window.plot.ax.plot(window.ens.t*tau,window.ens.fit(window.ens.t),color='r',lw=1,alpha=pp['line_ens_alpha'])

	# if pp['show_mean']:
		# window.plot.ax.plot(window.ind.t*tau, np.nanmean(window.ind.y,axis=0), color='orange', alpha=pp['line_ens_alpha'])
		# window.plot.ax.plot(window.ind.t*tau, np.median(window.ind.y,axis=0), color='orange', alpha=pp['line_ens_alpha'])
	window.nmol = window.fpb.shape[0]

	window.plot.ax.set_xscale('linear')
	if pp['time_scale'] == 'log':
		window.plot.ax.set_xscale('log')
		if pp['time_min'] < pp['time_dt']:
			pp['time_min'] = pp['time_dt']
	window.plot.ax.set_xlim(pp['time_min'],pp['time_max'])
	window.plot.ax.set_ylim(pp['acorr_min'],pp['acorr_max'])

	# if pp['show_hmm']:
	# 	if not window.gui.maven.modeler.model is None and not window.hmm is None:
	# 		hr = window.gui.maven.modeler.model
	# 		if hr.type in ['vb Consensus HMM']:
	# 			window.plot.ax.plot(window.hmm.t*tau, window.hmm.y, color=pp['line_hmmcolor'],alpha=pp['line_ens_alpha'])

@tryexcept
def plot_powerspectrum(window):
	''' plot power spectra ensemble'''
	pp = window.prefs
	from .autocorr import power_spec
	tau = pp['time_dt']
	f = window.ens.freq

	if pp['show_ind']:
		for i in range(window.ind.y.shape[0]):
			window.plot.ax.semilogy(window.ind.freq[i]/tau,np.abs(window.ind.fft[i]),color='k',alpha=pp['line_ind_alpha'],zorder=-2)
	if pp['show_ens']:
		window.plot.ax.semilogy(f/tau, window.ens.fft,lw=1.,color=pp['line_color'],alpha=pp['line_ens_alpha'],zorder=1)
	# if pp['show_stretch']:
		# y = window.ens.fit(window.ens.t)
		# tt = window.ens.t
		# ww,fft = power_spec(tt,y)
		# window.plot.ax.semilogy(ww/tau, fft, lw=1., color='red', alpha=pp['line_ens_alpha'], zorder=1)
	# if pp['show_mean']:
	# 	q = np.nanmean(window.ind.y,axis=0)
	# 	w,f = power_spec(window.ens.t,q)
	# 	window.plot.ax.semilogy(w/tau, f, color='orange', alpha=pp['line_ens_alpha'])

	window.plot.ax.set_ylim(pp['power_min'],pp['power_max'])
	ft = window.ind.freq/tau
	window.plot.ax.set_xlim(ft[ft>0].min(),ft[ft>0].max())
	window.plot.ax.set_xscale('log')
	window.plot.ax.set_yscale('log')

	# if pp['show_hmm']:
	# 	if not window.gui.maven.modeler.model is None and not window.hmm is None:
	# 		hr = window.gui.maven.modeler.model
	# 		if hr.type in ['vb Consensus HMM']:
	# 			window.plot.ax.plot(window.hmm.freq/tau, window.hmm.fft, color=pp['line_hmmcolor'],alpha=pp['line_ens_alpha'])
	# window.nmol = window.fpb.shape[0]

@tryexcept
def plot_mean(window):
	''' plot mean signal value versus time'''
	pp = window.prefs
	tau = pp['time_dt']

	y = np.nanmean(window.fpb,axis=0)
	t = tau * np.arange(y.size)
	window.plot.ax.plot(t,y, color='blue', lw=1., alpha=pp['line_ens_alpha'])
	window.plot.ax.set_xlim(t.min(),t.max())
	window.plot.ax.set_ylim(-.25,1.25)

	window.nmol = window.fpb.shape[0]

@tryexcept
def plot_tc(window):
	'''histogram correlation times from fitting'''
	pp = window.prefs
	tau = pp['time_dt']
	beta = window.ind.beta.copy()
	tc = window.ind.tc.copy()
	# x = tf >= tf_cut
	# tc = tc[x]
	x = tc > 1.01 ## effectively 1.0
	tc = tc[x]
	# y = np.log(tc*tau)
	y = tc*tau
	y = y[np.isfinite(y)]
	# ymin = np.log(pp['tc_fitcut']*tau)
	ymin = pp['tc_fitcut']*tau
	window.nmol = (y[y>ymin]).size

	if pp['tc_min'] > 0 :
		rmin = np.log(pp['tc_min'])
	else:
		rmin = np.log(tau/2.)
	if pp['tc_max'] > 0:
		rmax = np.log(pp['tc_max'])
	else:
		rmax = np.min((np.nanmax(y),np.log(window.ens.t.size/1.)))

	rmin = np.exp(rmin)
	rmax = np.exp(rmax)
	hy = window.plot.ax.hist(y[y>ymin],bins=pp['tc_nbins'],range=(rmin,rmax),histtype='stepfilled',density=True,color=pp['hist_color'],alpha=.5)[0]
	# if pp['tc_showens']:
		# window.plot.ax.axvline(x=np.log(window.ens.tc*tau),color='k')
	if pp['tc_showmean']:
		yy = y[y>ymin]
		yy = yy[yy>rmin]
		yy = yy[yy<rmax]
		# qq = np.log(np.nanmean(np.exp(yy)))
		qq = np.nanmean(np.exp(yy))
		window.plot.ax.axvline(x=qq,color=pp['hist_color'],alpha=.9,lw=1.)

	if not window.gui.maven.modeler.model is None and not window.hmm is None:
		hr = window.gui.maven.modeler.model
		if hr.type in ['vb Consensus HMM']:
			# window.plot.ax.axvline(x=np.log(window.hmm.tc*tau),color=pp['hist_color'],alpha=.9,linestyle='--',lw=1.)
			window.plot.ax.axvline(x=(window.hmm.tc*tau),color=pp['hist_color'],alpha=.9,linestyle='--',lw=1.)

	# if pp['tc_showmean']:
	# 	ltc =  np.linspace(rmin,rmax,1000)
	# 	v = np.nanvar(np.log(tc*tau))
	# 	m = np.nanmean(np.log(tc*tau))
	# 	lp = (2.*np.pi*v)**-.5 * np.exp(-.5/v*(ltc-m)**2.)
	# 	window.plot.ax.plot(ltc,lp,color='k',lw=1,alpha=.9)
	if pp['tc_showkde']:
		ltc =  np.linspace(rmin,rmax,1000)
		lp = kde(ltc,y[y>ymin])
		window.plot.ax.plot(ltc,lp,color=pp['hist_color'],lw=1,alpha=.9)
	window.plot.ax.set_xlim(rmin,rmax)
	window.plot.ax.set_ylim(0.,pp['tc_ymax'])

	window.nmol = (y[y>ymin]).size

@tryexcept
def plot_beta(window):
	''' histogram beta value from fitting power law decay'''
	pp = window.prefs
	beta = window.ind.beta.copy()
	tc = window.ind.tc.copy()

	x = tc > 1.01 ## effectively 1.0
	# tf = window.ind.tfit.copy()
	# tf_cut = pp['tc_fitcut']
	# x = tf >= tf_cut
	# x = np.isfinite(beta)
	# beta = beta[x]
	window.plot.ax.hist(beta[x],bins=pp['beta_nbins'],range=(0,2),histtype='stepfilled',color=pp['hist_color'])
	if pp['beta_showens']:
		window.plot.ax.axvline(x=window.ens.beta,color='k')
	if pp['beta_showmean']:
		window.plot.ax.axvline(x=np.nanmean(beta[x]),color='k')
	window.plot.ax.set_xlim(0,2.)
	window.nmol = beta[x].size

@tryexcept
def plot_scatter(window):
	'''scatter plot tc versus beta from power law fitting'''
	pp = window.prefs
	from .autocorr import vgamma
	tau = pp['time_dt']
	beta = window.ind.beta.copy()
	tc = window.ind.tc.copy()


	# tf = window.ind.tfit.copy()
	tf_cut = pp['tc_fitcut']
	# x = (tf >= tf_cut)*np.isfinite(beta)
	x = tc > 1.01 ## effectively 1.0

	# x = np.isnan(beta)
	# beta[x] = 1.

	window.plot.ax.loglog(beta[x],tc[x]*tau,'o',alpha=.5,color=pp['hist_color'])
	# x = (tf < tf_cut)
	# window.plot.ax.loglog(beta[x],tc[x]*tau,'o',alpha=.5,color='r')
	# x = np.isnan(beta)
	# window.plot.ax.loglog(np.ones(int(x.sum())),tc[x]*tau,'o',alpha=.5,color='r')

	window.plot.ax.set_xlim(.1,3.)
	window.plot.ax.set_ylim(tau,tau*window.fpb.shape[1])
	bb = np.linspace(.1,3.,10000)
	# p = np.array((np.ones(bb.size),np.zeros(bb.size)+tf_cut,bb))
	# tt = tf_cut/bb*vgamma(1./bb)*tau
	# window.plot.ax.plot(bb,tt,color='k',ls='--',lw=1.,alpha=.9)
	window.nmol = int(np.isfinite(beta[x]).sum())

	if not window.gui.maven.modeler.model is None and not window.hmm is None:
		hr = window.gui.maven.modeler.model
		if hr.type in ['vb Consensus HMM']:
			window.plot.ax.axhline(y=window.hmm.tc*tau,color=pp['line_hmmcolor'],alpha=.9)
			window.plot.ax.axvline(x=window.hmm.beta,color=pp['line_hmmcolor'],alpha=.9)

@tryexcept
def plot_indacf(window):
	'''show ACF for individual trace'''
	pp = window.prefs
	from .autocorr import fit_acf
	tau = pp['time_dt']

	ind = pp['acorr_ind']
	if ind < 0:
		pp['acorr_ind'] = 0
		plot_indacf(window)
		return
	elif ind >= window.ind.y.shape[0]:
		pp['acorr_ind'] = window.ind.y.shape[0] - 1
		plot_indacf(window)
		return
	window.plot.ax.axhline(y=0,color='k',alpha=.5)

	window.plot.ax.plot(window.ind.t*tau, window.ind.y[ind], color='k', alpha=pp['line_alpha'])
	fit = fit_acf(window.ind.t,window.ind.y[ind],pp['tc_fit_ymin'],pp['fit_biexp'])
	window.plot.ax.plot(window.ind.t*tau,fit(window.ind.t),color='r',alpha=pp['line_alpha'])

	window.plot.ax.set_xscale('linear')
	if pp['time_scale'] == 'log':
		window.plot.ax.set_xscale('log')
		if pp['time_min'] < pp['time_dt']:
			pp['time_min'] = pp['time_dt']
	window.plot.ax.set_xlim(pp['time_min'],pp['time_max'])
	window.plot.ax.set_ylim(pp['acorr_min'],pp['acorr_max'])

	fit.tau = float(tau)
	window.plot.ax.set_title(r"$t_c=%.3f: $"%(fit.calc_tc()) + str(fit),fontsize=pp['label_fontsize'])
