import numpy as np
import logging
logger = logging.getLogger(__name__)

from .base import controller_base_analysisplot
from ..modeler.fxns.exponentials import (single_exp_surv, double_exp_surv, triple_exp_surv, stretched_exp_surv,
							 single_exp_hist, double_exp_hist, triple_exp_hist, stretched_exp_hist)

class controller_survival_dwell(controller_base_analysisplot):
	def __init__(self,maven):
		super().__init__(maven)
		self.defaults()

	def defaults(self):
		self.prefs.add_dictionary({
			'fig_height':4.0,
			'fig_width':4.0,
			'subplots_top':0.95,
			'subplots_right':0.95,
			'subplots_left':0.3,
			'subplots_bottom':0.3,
			'subplots_hspace':0.0,
			'subplots_wspace':0.0,
			'axes_topright':True,
			'xlabel_offset':-0.5,
			'ylabel_offset':-0.25,

			'dwell_nbins':51,
			'dwell_nticks':6,
			'dwell_state':0,
			'dwell_force_xmax': False,
			'dwell_max': 100,
			'time_dt':1.0,

			'hist_on':False,
			'hist_type':'bar',
			'hist_color':'tab:red',
			'hist_edgecolor':'tab:black',
			'hist_log':False,
			'hist_force_y':False,
			'hist_ymax':5.0,
			'hist_ymin':0.0,
			'hist_nticks':5,

			'survival_on':True,
			'survival_color':'steelblue',
			'survival_marker': '.',
			'survival_mark_size':3,
			'survival_edgecolor': 'black',

			'model_on': True,
			'model_color': 'tab:black',
			'model_ls': '--',
			'model_lw':1.,

			'residual_heightpercent':25,
			'residual_padpercent':5,
			'residual_nticks':2,
			'residual_ylabel_text': 'Residual',
			'residual_force_y':False,
			'residual_ymin':-0.1,
			'residual_ymax':0.1,
			'residual_zero_alpha':0.15,
			'residual_lw':1.,

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
			dt = np.float64(self.prefs['time_dt'])
			d = np.array(self.maven.modeler.model.dwells[str(self.prefs['dwell_state'])])
			d = d*dt
			if d.size == 0:
				error_dwell = True
			else:
				tau, surv = self.maven.modeler.get_survival_dwells(self.prefs['dwell_state'])
				tau = tau*dt
				error_dwell = False
				drange=np.arange(d.min(),d.max())

		if error_dwell:
			d = np.array([])
			tau =  np.array([])
			surv = np.array([])
			drange = np.array([])

		if len(fig.axes)>1:
			[aa.remove() for aa in fig.axes[1:]]
		ax.cla()
		self.fix_ax(fig,ax)

		self.d = d
		self.surv = surv
		self.hist_x = drange
		self.model = None

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
			bin_height, bin_edge = ax.hist(d, bins=self.prefs['dwell_nbins'],
				range=hist_range, #(d.min(), d.max()),
				histtype=self.prefs['hist_type'], alpha=.8, density=True,
				color=color, edgecolor=ecolor, log=self.prefs['hist_log'])[:2]

			self.hist_x = bin_edge[1:] - 0.5*(bin_edge[1] - bin_edge[0])
			self.hist_y = bin_height.astype('double')

		try:
			if self.prefs['model_on'] and not self.maven.modeler.model.rates is None:
				if self.maven.modeler.model.rate_type == "Transition Matrix":
					k = self.maven.modeler.model.rates[self.prefs['dwell_state']].sum()
					k = k/dt
					self.k = k
					self.a = 1
					self.beta = None

					decay_surv = np.exp(-k*tau)
					decay_hist = k*np.exp(-self.hist_x*k)

				elif self.maven.modeler.model.rate_type == "Dwell Analysis":
					
					rate = self.maven.modeler.model.rates[self.prefs['dwell_state']]
					k = rate['ks']
					k = k/dt
					self.k = k
					self.a = rate['As']
					
					if 'betas' in rate:
						self.beta = rate['betas']
						
						decay_surv = stretched_exp_surv(tau, self.k, self.beta, self.a)
						decay_hist = stretched_exp_hist(self.hist_x, self.k, self.beta, self.a)

					elif len(self.k) == 1:
						self.beta = None
						
						decay_surv = single_exp_surv(tau, self.k, self.a)
						decay_hist = single_exp_hist(self.hist_x, self.k, self.a)

					elif len(self.k) == 2:
						self.beta = None
						
						decay_surv = double_exp_surv(tau, self.k[0], self.k[1], self.a[0], self.a[1])
						decay_hist = double_exp_hist(self.hist_x, self.k[0], self.k[1], self.a[0], self.a[1])

					elif len(self.k) == 3:
						self.beta = None
						
						decay_surv = triple_exp_surv(tau, self.k[0], self.k[1], self.k[2], self.a[0], self.a[1],self.a[2])
						decay_hist = triple_exp_hist(self.hist_x, self.k[0], self.k[1], self.k[2], self.a[0], self.a[1],self.a[2])

				color = self.prefs['model_color']
				if not colors.is_color_like(color):
					color = 'black'
					
				if self.prefs['survival_on']:
					self.model = decay_surv
					ax.plot(tau,decay_surv,color=color,ls =self.prefs['model_ls'])
				elif self.prefs['hist_on']:
					self.model = decay_hist
					ax.plot(self.hist_x,decay_hist,color=color,ls =self.prefs['model_ls'])
		except:
			pass

		from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
		ax_divider = make_axes_locatable(ax)
		rax = ax_divider.append_axes("bottom", size="%d%%"%(self.prefs['residual_heightpercent']), pad="%d%%"%(self.prefs['residual_padpercent']))
		fig.add_axes(rax)
		rax.cla()

		if self.prefs['survival_on']:
			color = self.prefs['survival_color']
			if not colors.is_color_like(color):
				color = 'steelblue'
		elif self.prefs['hist_on']:
			color = self.prefs['hist_color']
			if not colors.is_color_like(color):
				color = 'red'
		else:
			color = 'black'

		rax.axhline(y=0, color=color, alpha=self.prefs['residual_zero_alpha'])
		
		if self.model is not None:
			if self.prefs["model_on"]:
				color = self.prefs['model_color']
				if not colors.is_color_like(color):
					color = 'black'

				if self.prefs["survival_on"]:
					residuals = self.surv - self.model
					rax.plot(tau, residuals, color=color,ls =self.prefs['model_ls'],lw=self.prefs['model_lw'])
				elif self.prefs["hist_on"]:
					residuals = self.hist_y - self.model
					rax.plot(self.hist_x, residuals, color=color,ls =self.prefs['model_ls'],lw=self.prefs['residual_lw'])

		self.fix_ax(fig,ax)
		self.garnish(fig,ax,rax)
		fig.canvas.draw()

	def garnish(self,fig,ax,rax):
		## Fix up the plot

		if self.prefs['hist_log']:
			ax.set_yscale('log')
			
		ylim_ax = ax.get_ylim()
		ax.set_ylim(*ylim_ax) ## incase modeling gave crazy results

		xlim_ax = ax.get_xlim()
		ax.set_xlim(0, xlim_ax[1])

		ylim_rax = rax.get_ylim()
		rax.set_ylim(*ylim_rax) ## incase modeling gave crazy results

		xlim_rax = rax.get_xlim()
		rax.set_xlim(0,xlim_rax[1])
		

		if self.prefs['hist_force_y']:
			ax.set_ylim(self.prefs['hist_ymin'], self.prefs['hist_ymax'])
			if not self.prefs['hist_log']:
				ticks_ax = self.best_ticks(self.prefs['hist_ymin'], self.prefs['hist_ymax'], self.prefs['hist_nticks'])
			else:
				ticks_ax = 10**self.best_ticks(np.log10(self.prefs['hist_ymin']), 
								   			  np.log10(self.prefs['hist_ymax']), self.prefs['hist_nticks'])

		else:
			if not self.prefs['hist_log']:
				ticks_ax = self.best_ticks(0,ylim_ax[1], self.prefs['hist_nticks'])
			else:
				ticks_ax = 10**self.best_ticks(np.log10(ylim_ax[0]),np.log10(ylim_ax[1]), self.prefs['hist_nticks'])

		ax.set_yticks(ticks_ax)

		if self.prefs['residual_force_y']:
			rax.set_ylim(self.prefs['residual_ymin'], self.prefs['residual_ymax'])
			ticks_rax = self.best_ticks(self.prefs['residual_ymin'], self.prefs['residual_ymax'], self.prefs['residual_nticks'])
		else:
			ticks_rax = self.best_ticks(ylim_rax[0],ylim_rax[1], self.prefs['residual_nticks'])
		rax.set_yticks(ticks_rax)


		if self.prefs['dwell_force_xmax']:
			ax.set_xlim(0,self.prefs['dwell_max'])
			rax.set_xlim(0,self.prefs['dwell_max'])
			ticks = self.best_ticks(0,self.prefs['dwell_max'],self.prefs['dwell_nticks'])
		else:
			ticks = self.best_ticks(0,xlim_ax[1],self.prefs['dwell_nticks'])
		
		ax.set(xticks = [], xticklabels=[])
		rax.set_xticks(ticks)

		dpr = self.devicePixelRatio()
		fontdict = {'family': self.prefs['font'],
			'size': self.prefs['label_fontsize']/dpr,
			'va':'top'}
		
		rax.set_xlabel(self.prefs['xlabel_text'], fontdict=fontdict)
		rax.xaxis.set_label_coords(0.5, self.prefs['xlabel_offset'])

		ax.set_ylabel(self.prefs['ylabel_text'], fontdict=fontdict)
		ax.yaxis.set_label_coords(self.prefs['ylabel_offset'], 0.5)

		rax.set_ylabel(self.prefs['residual_ylabel_text'], fontdict=fontdict)
		rax.yaxis.set_label_coords(self.prefs['ylabel_offset'], 0.5)

		bbox_props = dict(boxstyle="square", fc="w", alpha=1.0,lw=1./dpr)
		lstr = 'N = %d'%(self.d.shape[0])
		ax.annotate(lstr,xy=(self.prefs['textbox_x'], self.prefs['textbox_y']),
			xycoords='axes fraction', ha='right', color='k',
			bbox=bbox_props, fontsize=self.prefs['textbox_fontsize']/dpr,
			family=self.prefs['font'])

		try:
			if self.prefs['model_on'] and not self.maven.modeler.model.rates is None:
				if not self.beta is None:
					lstr2 ='A = {} \nk = {}\n'.format(np.around(self.a, decimals = 3),np.around(self.k, decimals = 3)) + r'$\beta$ = {}'.format(np.around(self.beta, decimals = 3))
				else:
					lstr2 = 'A = {} \nk = {}'.format(np.around(self.a, decimals = 3), np.around(self.k, decimals = 3))
				ax.annotate(lstr2,xy=(self.prefs['textbox_x'], self.prefs['textbox_y'] - self.prefs['textbox_offset']),
					xycoords='axes fraction', ha='right', color='k',
					bbox=bbox_props, fontsize=self.prefs['textbox_fontsize']/dpr,
					family=self.prefs['font'])
		except:
			pass


