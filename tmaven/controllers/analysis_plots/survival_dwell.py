import numpy as np
import logging
logger = logging.getLogger(__name__)

from .base import controller_base_analysisplot
from ..modeler.dwells import (single_exp_surv, double_exp_surv, triple_exp_surv, stretched_exp_surv,
							 single_exp_hist, double_exp_hist, triple_exp_hist, stretched_exp_hist)

class controller_survival_dwell(controller_base_analysisplot):
	def __init__(self,maven):
		super().__init__(maven)
		self.defaults()

	def defaults(self):
		self.prefs.add_dictionary({
			'fig_height':2.5,
			'fig_width':3.5,
			'subplots_top':0.95,
			'subplots_right':0.95,
			'subplots_left':0.2,
			'subplots_bottom':0.2,
			'subplots_hspace':0.0,
			'subplots_wspace':0.0,
			'axes_topright':True,
			'xlabel_offset':-0.15,
			'ylabel_offset':-0.2,

			'dwell_nbins':51,
			'dwell_nticks':6,
			'dwell_state':0,
			'dwell_force_xmax': False,
			'dwell_min': 0.,
			'dwell_max': 100,

			'hist_on':True,
			'hist_type':'bar',
			'hist_color':'tab:red',
			'hist_edgecolor':'tab:black',
			'hist_log_y':True,
			'hist_force_ymax':False,
			'hist_ymax':5.0,
			'hist_ymin':0.0,
			'hist_nticks':5,

			'survival_on':True,
			'survival_color':'steelblue',
			'survival_marker': '.',
			'survival_mark_size':3,
			'survival_edgecolor': 'black',

			'model_on': False,
			'model_color': 'tab:black',
			'model_ls': '--',

			'textbox_x':0.965,
			'textbox_y':0.9,
			'textbox_offset':0.175,
			'textbox_fontsize':7.0,
			'textbox_nmol':True,
			'xlabel_text':r'Dwells (s)',
			'ylabel_text':r'Probability'})

	def plot(self,fig,ax):
		## Decide if we should be plotting at all
		if self.maven.modeler.model is None:
			logger.error('No model')
			error_dwell = True
		elif self.maven.modeler.model.dwells is None:
			logger.error('Incorrect model')
			error_dwell = True
		elif self.prefs['dwell_state'] > self.maven.modeler.model.nstates - 1:
			logger.error('State number out of bounds')
			error_dwell = True
		else:
			self.d = np.array(self.maven.modeler.model.dwells[str(self.prefs['dwell_state'])])
			tau, self.surv = self.maven.modeler.get_survival_dwells(self.prefs['dwell_state'])
			error_dwell = False

		if error_dwell:
			self.d = np.array([])
			tau =  np.array([])
			self.surv = np.array([])

		ax.cla()
		self.fix_ax(fig,ax)

		d = self.d
		surv = self.surv

		## Plot Histogram
		from matplotlib import colors
		if self.prefs['survival_on']:
			color = self.prefs['survival_color']
			if not colors.is_color_like(color):
				color = 'steelblue'
			ecolor = self.prefs['survival_edgecolor']
			if not colors.is_color_like(ecolor):
				ecolor = 'black'
			ax.plot(tau, surv, color=color, marker=self.prefs['survival_marker'], ls = "", ms = self.prefs['survival_mark_size'])

			if self.prefs['model_on']:
				if not self.maven.modeler.model.rates is None:
					if self.maven.modeler.model.rate_type == "Transition Matrix":
						#d_range=np.arange(d.min(),d.max())
						k = self.maven.modeler.model.rates[self.prefs['dwell_state']].sum()
						self.k = k
						self.a = 1
						decay = np.exp(-tau*k)
						self.beta = None
					elif self.maven.modeler.model.rate_type == "Dwell Analysis":
						rate = self.maven.modeler.model.rates[self.prefs['dwell_state']]
						self.k = rate['ks']
						self.a = rate['As']
						if 'betas' in rate:
							self.beta = rate['betas']
							decay = stretched_exp_surv(tau, self.k, self.beta, self.a)
						elif len(self.k) == 1:
							self.beta = None
							decay = single_exp_surv(tau, self.k, self.a)
						elif len(self.k) == 2:
							self.beta = None
							decay = double_exp_surv(tau, self.k[0], self.k[1], self.a[0], self.a[1])
						elif len(self.k) == 3:
							self.beta = None
							decay = triple_exp_surv(tau, self.k[0], self.k[1], self.k[1],self.a[0]/self.a.sum(), self.a[1]/self.a.sum(),self.a.sum())

					color = self.prefs['model_color']
					if not colors.is_color_like(color):
						color = 'black'
					ax.plot(tau,decay,color=color,ls =self.prefs['model_ls'])

			if self.prefs['hist_log_y']:
				ax.set_yscale('log')

		elif self.prefs['hist_on']:
			color = self.prefs['hist_color']
			if not colors.is_color_like(color):
				color = 'red'
			ecolor = self.prefs['hist_edgecolor']
			if not colors.is_color_like(ecolor):
				ecolor = 'black'
			try:
				hist_range=(d.min(), d.max())
			except:
				hist_range = (0.,100.)
			self.hist_y, self.hist_x = ax.hist(d,bins=self.prefs['dwell_nbins'],
				range=hist_range, #(d.min(), d.max()),
				histtype=self.prefs['hist_type'], alpha=.8, density=True,
				color=color, edgecolor=ecolor, log=self.prefs['hist_log_y'])[:2]
			if self.prefs['hist_log_y']:
				ax.set_yscale('log')



			if self.prefs['model_on']:
				if not self.maven.modeler.model.rates is None:
					drange=np.arange(d.min(),d.max())
					if self.maven.modeler.model.rate_type == "Transition Matrix":
						#d_range=np.arange(d.min(),d.max())
						k = self.maven.modeler.model.rates[self.prefs['dwell_state']].sum()
						self.k = k
						self.a = 1
						decay = k*np.exp(-drange*k)
						self.beta = None
					elif self.maven.modeler.model.rate_type == "Dwell Analysis":
						rate = self.maven.modeler.model.rates[self.prefs['dwell_state']]
						self.k = rate['ks']
						self.a = rate['As']
						if 'betas' in rate:
							self.beta = rate['betas']
							decay = stretched_exp_hist(drange, self.k, self.beta, self.a)
						elif len(self.k) == 1:
							self.beta = None
							decay = single_exp_hist(drange, self.k, self.a)
						elif len(self.k) == 2:
							self.beta = None
							decay = double_exp_hist(drange, self.k[0], self.k[1], self.a[0]/self.a.sum(), self.a.sum())
						elif len(self.k) == 3:
							self.beta = None
							decay = triple_exp_hist(drange, self.k[0], self.k[1], self.k[1],self.a[0]/self.a.sum(), self.a[1]/self.a.sum(),self.a.sum())

					color = self.prefs['model_color']
					if not colors.is_color_like(color):
						color = 'black'

					ax.plot(drange,decay,color=color,ls =self.prefs['model_ls'])

		self.garnish(fig,ax)
		fig.canvas.draw()

	def garnish(self,fig,ax):
		## Fix up the plot
		ylim = ax.get_ylim()
		xlim = ax.get_xlim()
		ax.set_xlim(0,xlim[1])
		ax.set_ylim(*ylim) ## incase modeling gave crazy results

		if self.prefs['hist_force_ymax']:
			ax.set_ylim(self.prefs['hist_ymin'], self.prefs['hist_ymax'])
			ticks = self.best_ticks(self.prefs['hist_ymin'], self.prefs['hist_ymax'], self.prefs['hist_nticks'])
		else:
			if not self.prefs['hist_log_y']:
				ticks = self.best_ticks(0,ax.get_ylim()[1], self.prefs['hist_nticks'])
			else:
				try:
					ticks = self.best_ticks(10**(-np.floor(np.log10(len(self.d)))),ax.get_ylim()[1], self.prefs['hist_nticks'])
				except:
					ticks = self.best_ticks(0,ax.get_ylim()[1], self.prefs['hist_nticks'])
			ax.set_yticks(ticks)
		ticks = self.best_ticks(0,xlim[1],self.prefs['dwell_nticks'])
		ax.set_xticks(ticks)

		dpr = self.devicePixelRatio()
		fontdict = {'family': self.prefs['font'],
			'size': self.prefs['label_fontsize']/dpr,
			'va':'top'}
		ax.set_xlabel(self.prefs['xlabel_text'], fontdict=fontdict)
		ax.set_ylabel(self.prefs['ylabel_text'], fontdict=fontdict)
		ax.yaxis.set_label_coords(self.prefs['ylabel_offset'], 0.5)
		ax.xaxis.set_label_coords(0.5, self.prefs['xlabel_offset'])
		bbox_props = dict(boxstyle="square", fc="w", alpha=1.0,lw=1./dpr)
		lstr = 'N = %d'%(self.d.shape[0])
		ax.annotate(lstr,xy=(self.prefs['textbox_x'], self.prefs['textbox_y']),
			xycoords='axes fraction', ha='right', color='k',
			bbox=bbox_props, fontsize=self.prefs['textbox_fontsize']/dpr,
			family=self.prefs['font'])

		if self.prefs['model_on']:
			print(self.beta)
			if not self.beta is None:
				lstr2 ='A = {} \nk = {}\n'.format(np.around(self.a, decimals = 3),np.around(self.k, decimals = 3)) + r'$\beta$ = {}'.format(np.around(self.beta, decimals = 3))
			else:
				lstr2 = 'A = {} \nk = {}'.format(np.around(self.a, decimals = 3), np.around(self.k, decimals = 3))
			ax.annotate(lstr2,xy=(self.prefs['textbox_x'], self.prefs['textbox_y'] - self.prefs['textbox_offset']),
				xycoords='axes fraction', ha='right', color='k',
				bbox=bbox_props, fontsize=self.prefs['textbox_fontsize']/dpr,
				family=self.prefs['font'])
