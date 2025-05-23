import numpy as np
import logging
logger = logging.getLogger(__name__)

from .base import controller_base_analysisplot

class controller_data_hist1d(controller_base_analysisplot):
	'''
	Notes
	-----
	prefs['hist_on'] : bool
		whether to plot histogram or not
	prefs['hist_color'] : str
		color of histogram
	prefs['hist_edgecolor'] : str
		color histogram edges
	prefs['signal_min'] : double
		min FRET value for histogram
	prefs['signal_max'] : double
		max FRET value for histogram
	prefs['signal_nbins'] : int
		number of bins in FRET histogram
	prefs['histtype'] : str
		matplotlib histogram type
	prefs['hist_log'] : bool
		whether to log10 scale the y axis of the histogram
	prefs['hmm_on'] : bool
		whether to plot the HMM overlay
	prefs['gmm_on'] : bool
		whether to plot the GMM overlay
	prefs['hist_auto_ylim'] : bool
		whether to use ymin,ymax values in preferences or autoscale
	prefs['hist_ymin'] : double
		minimum y axis value
	prefs['hist_ymax'] : double
		maximum y axis value
	prefs['hist_nticks'] : int
		number of ticks on y axis
	prefs['signal_nticks'] : int
		number of ticks on x axis
	'''

	def __init__(self,maven):
		super().__init__(maven)
		self.defaults()
		self.hist_x = np.array((0.,.5,1.))
		self.hist_y = np.array((0.,0.,0.))

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
			'xlabel_text':r'E$_{\rm{FRET}}$',
			'ylabel_text':r'Density',

			'hist_on':True,
			'hist_type':'stepfilled',
			'hist_color':'tab:blue',
			'hist_edgecolor':'tab:blue',

			'signal_min':-.25,
			'signal_max':1.25,
			'signal_nbins':151,
			'signal_nticks':6,

			'hist_log':False,
			'hist_auto_ylim':True,
			'hist_ymax':5.0,
			'hist_ymin':0.0,
			'hist_nticks':5,

			'data_clip_low':-1.,
			'data_clip_high':2.,

			'model_on':True,
			'idealized':False,

			'textbox_x':0.965,
			'textbox_y':0.9,
			'textbox_fontsize':7.0,
			'textbox_nmol':True,

			'plot_channel':0,
			'plot_mode': 'smFRET'
		})
	
	def fret_defaults(self):
		self.prefs.add_dictionary({
			'xlabel_text':r'E$_{\rm{FRET}}$',
			'signal_min':-.25,
			'signal_max':1.25,
			'plot_mode':'smFRET'
		})

	def normalized_defaults(self):
		self.prefs.add_dictionary({
			'xlabel_text':r'Normalized Intensity',
			'signal_min':-.25,
			'signal_max':1.25,
			'plot_mode':'ND Normalized'
		})
	
	def raw_defaults(self):
		self.prefs.add_dictionary({
			'xlabel_text':r'Intensity (A.U.)',
			'signal_min':-500,
			'signal_max':10000,
			'plot_mode':'ND Raw'
		})

	def plot(self,fig,ax):
		## Decide if we should be plotting at all
		#if not self.maven.data.ncolors == 2:
			#logger.error('more than 2 colors not implemented')
			# return

		## Setup
		ax.cla()
		self.fix_ax(fig,ax)

		self.plot_mode = self.prefs['plot_mode']

		if self.prefs['idealized']:
			self.dpb = self.get_idealized_data()
			if self.dpb is None:
				self.dpb = np.array(())
		else:
			if self.plot_mode == 'smFRET':
				index = 1
			else:
				index = self.prefs['plot_channel']
				
			try:
				self.dpb = self.get_plot_data()[:,:,index].copy()
			except:
				self.dpb = np.array(())
			
		dpb = self.dpb[np.isfinite(self.dpb)].flatten()

		## Plot Histogram
		from matplotlib import colors
		if self.prefs['hist_on']:
			color = self.prefs['hist_color']
			if not colors.is_color_like(color):
				color = 'steelblue'
			ecolor = self.prefs['hist_edgecolor']
			if not colors.is_color_like(ecolor):
				ecolor = 'black'
			self.hist_y, self.hist_x = ax.hist(dpb,bins=self.prefs['signal_nbins'],
				range=(self.prefs['signal_min'], self.prefs['signal_max']),
				histtype=self.prefs['hist_type'], alpha=.8, density=True,
				color=color, edgecolor=ecolor, log=self.prefs['hist_log'])[:2]
			if self.prefs['hist_log']:
				ax.set_yscale('log')

		if self.prefs['model_on']:
			m = self.maven.modeler.model
			if not self.maven.modeler.model is None:
				try:
					x = np.linspace(self.prefs['signal_min'],self.prefs['signal_max'],1001)
					y = np.zeros_like(x)
					n = m.mean.size
					for i in range(n):
						yi = m.frac[i]*1./np.sqrt(2.*np.pi*m.var[i])*np.exp(-.5/m.var[i]*(x-m.mean[i])**2.)
						y += yi
						ax.plot(x,yi,color='k',lw=1,alpha=.8,ls='--')
					ax.plot(x,y,color='k',lw=2,alpha=.8)
				except:
					pass

		self.garnish(fig,ax)
		fig.canvas.draw()

	def garnish(self,fig,ax):
		## Fix up the plot
		# ymin,ymax = ax.get_ylim()
		ax.set_xlim(self.prefs['signal_min'], self.prefs['signal_max'])
		# ax.set_ylim(*ylim) ## incase modeling gave crazy results
		if self.hist_y.sum() > 0:
			if not self.prefs['hist_auto_ylim']:
				ax.set_ylim(self.prefs['hist_ymin'], self.prefs['hist_ymax'])
			else:
				ymin = self.hist_y[self.hist_y>0].min() if np.sum(self.hist_y>0) > 0 else 0
				ymax = self.hist_y.max()*1.1 +1e-6
				ax.set_ylim(ymin,ymax)
		if not self.prefs['hist_log']:
			if not self.prefs['hist_auto_ylim']:
				ticks = self.best_ticks(self.prefs['hist_ymin'], self.prefs['hist_ymax'], self.prefs['hist_nticks'])
			else:
				ticks = self.best_ticks(0,ax.get_ylim()[1], self.prefs['hist_nticks'])
			ax.set_yticks(ticks)

		ticks = self.best_ticks(self.prefs['signal_min'],self.prefs['signal_max'],self.prefs['signal_nticks'])
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
		lstr = 'N = %d'%(self.dpb.shape[0])
		ax.annotate(lstr,xy=(self.prefs['textbox_x'], self.prefs['textbox_y']),
			xycoords='axes fraction', ha='right', color='k',
			bbox=bbox_props, fontsize=self.prefs['textbox_fontsize']/dpr,
			family=self.prefs['font'])
