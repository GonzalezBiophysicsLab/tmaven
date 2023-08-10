import numpy as np
import logging
from scipy import stats
logger = logging.getLogger(__name__)

from .base import controller_base_analysisplot

class controller_tm_hist(controller_base_analysisplot):
	'''
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
	'''

	def __init__(self,maven):
		super().__init__(maven)
		self.defaults()

	def defaults(self):
		self.prefs.add_dictionary({
			'fig_height':2.5,
			'fig_width':2.5,
			'subplots_top':0.98,
			'subplots_left':0.17,
			'subplots_right':0.97,
			'subplots_bottom':0.22,
			'xlabel_offset':-0.15,
			'ylabel_offset':-0.2,
			'prob_min':-.05,
			'prob_max':1.05,
			'prob_nbins':25,
			'prob_clip_low':-1.,
			'prob_clip_high':2.,
			'prob_nticks':6,
			'hist_on':False,
			'hist_type':'stepfilled',
			'hist_color':'tab:blue',
			'hist_edgecolor':'tab:blue',
			'kde': True,
			'kde_ls': '-',
			'kde_lw': 1,
			'kde_color': 'black',
			'kde_bandwidth': 0.25,
			'hist_log_y':False,
			'hist_force_ymax':False,
			'hist_ymax':5.0,
			'hist_ymin':0.0,
			'hist_nticks':5,

			'states': [0, 1],
			'model_on':True,
			'idealized':False,

			'textbox_x':0.965,
			'textbox_y':0.9,
			'textbox_fontsize':7.0,
			'textbox_nmol':True,
			'xlabel_text':'Transition Prob',
			'ylabel_text':'Density',
		})

	def get_composite_tm(self, states):
		if self.maven.modeler.model is None:
			return None
		elif hasattr(self.maven.modeler.model, 'trace_level'):
			result = self.maven.modeler.model
			init =  states[0]
			init_mean = result.mean[init]
			init_var = result.var[init]
			
			fin = states[1]
			fin_mean = result.mean[fin]
			fin_var = result.var[fin]

			trace_level = result.trace_level

			t_prob = []
			for vb in trace_level.values():
				init_prob = 1./np.sqrt(2.*np.pi*init_var)*np.exp(-.5/init_var*(vb.mean-init_mean)**2.)
				init_vb = np.argmax(init_prob)

				fin_prob = 1./np.sqrt(2.*np.pi*fin_var)*np.exp(-.5/fin_var*(vb.mean-fin_mean)**2.)
				fin_vb = np.argmax(fin_prob)
				t_prob.append(np.float64(vb.norm_tmatrix[init_vb, fin_vb]))

			return np.array(t_prob, dtype=np.float64)
		else:
			return None
			#probs /= probs.sum(1)[:,None]

	def plot(self,fig,ax):
		## Decide if we should be plotting at all
		if not self.maven.data.ncolors == 2:
			logger.error('more than 2 colors not implemented')
			# return

		## Setup
		ax.cla()
		self.fix_ax(fig,ax)

		states = self.prefs['states']
		try:
			self.tp = self.get_composite_tm(states)
			if self.tp is None:
				self.tp = np.array(())
		except:
			self.tp = np.array(())
		tp = self.tp[np.isfinite(self.tp)].flatten()

		## Plot Histogram
		from matplotlib import colors
		if self.prefs['hist_on']:
			color = self.prefs['hist_color']
			if not colors.is_color_like(color):
				color = 'steelblue'
			ecolor = self.prefs['hist_edgecolor']
			if not colors.is_color_like(ecolor):
				ecolor = 'black'
			self.hist_y, self.hist_x = ax.hist(tp,bins=self.prefs['prob_nbins'],
				range=(self.prefs['prob_min'], self.prefs['prob_max']),
				histtype=self.prefs['hist_type'], alpha=.8, density=True,
				color=color, edgecolor=ecolor, log=self.prefs['hist_log_y'])[:2]
			if self.prefs['hist_log_y']:
				ax.set_yscale('log')
		
		if self.prefs['kde']:
			try:
				color = self.prefs['kde_color']
				if not colors.is_color_like(color):
					color = 'black'
				gkde =  stats.gaussian_kde(tp, bw_method=self.prefs['kde_bandwidth'])
				self.pdf = gkde.evaluate(np.linspace(0.,1.,100))
				ax.plot(np.linspace(0.,1.,100), self.pdf, color = color, ls=self.prefs['kde_ls'], lw = self.prefs['kde_lw'])
				
				if self.prefs['hist_log_y']:
					ax.set_yscale('log')
			except:
				pass

		self.garnish(fig,ax)
		fig.canvas.draw()

	def garnish(self,fig,ax):
		## Fix up the plot
		ylim = ax.get_ylim()
		ax.set_xlim(self.prefs['prob_min'], self.prefs['prob_max'])
		ax.set_ylim(*ylim) ## incase modeling gave crazy results
		if not self.prefs['hist_log_y']:
			if self.prefs['hist_force_ymax']:
				ax.set_ylim(self.prefs['hist_ymin'], self.prefs['hist_ymax'])
				ticks = self.best_ticks(self.prefs['hist_ymin'], self.prefs['hist_ymax'], self.prefs['hist_nticks'])
			else:
				ticks = self.best_ticks(0,ax.get_ylim()[1], self.prefs['hist_nticks'])
			ax.set_yticks(ticks)
		ticks = self.best_ticks(self.prefs['prob_min'],self.prefs['prob_max'],self.prefs['prob_nticks'])
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
		lstr = 'N = %d'%(self.tp.shape[0])
		ax.annotate(lstr,xy=(self.prefs['textbox_x'], self.prefs['textbox_y']),
			xycoords='axes fraction', ha='right', color='k',
			bbox=bbox_props, fontsize=self.prefs['textbox_fontsize']/dpr,
			family=self.prefs['font'])
