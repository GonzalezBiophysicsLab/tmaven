import numpy as np
import logging
logger = logging.getLogger(__name__)

from .base import controller_base_analysisplot

class controller_data_tdp(controller_base_analysisplot):
	def __init__(self,maven):
		super().__init__(maven)
		self.defaults()

	def defaults(self):
		self.prefs.add_dictionary({
			'fig_height':4.0,
			'fig_width':4.0,
			'subplots_top':0.95,
			'subplots_right':0.83,
			'subplots_left':0.18,
			'axes_topright':True,
			'xlabel_offset':-0.2,
			'ylabel_offset':-0.26,

			'colorbar_widthpercent':5,
			'colorbar_padpercent':2,

			'signal_min':-.25,
			'signal_max':1.25,
			'signal_nbins':101,
			'signal_nticks':7,

			'hist_smoothx':1.,
			'hist_smoothy':1.,
			'hist_interp_res':0,
			'hist_rawsignal':True,
			'hist_normalize':True,
			'hist_log':True,

			'color_cmap':'jet',
			'color_floorcolor':r'#FFFFCC',
			'color_ceiling':.85,
			'color_floor':0.0001,
			'color_nticks':5,
			'color_xloc':.75,
			'color_dblfloor':.01,
			'color_dblfloorcolor':'white',
			'color_dbl':True,
			'color_decimals':3,

			'xlabel_rotate':45.,
			'ylabel_rotate':0.,
			'xlabel_text':r'Initial E$_{\rm{FRET}}$',
			'ylabel_text':r'Final E$_{\rm{FRET}}$',
			'xlabel_decimals':2,
			'ylabel_decimals':2,
			'textbox_x':0.95,
			'textbox_y':0.93,
			'textbox_fontsize':8,
			'textbox_nmol':True,

			'nskip':2,

			'plot_channel':0,
			'plot_mode': 'smFRET'
		})

	def fret_defaults(self):
		self.prefs.add_dictionary({
			'xlabel_text':r'Initial E$_{\rm{FRET}}$',
			'ylabel_text':r'Final E$_{\rm{FRET}}$',
			'signal_min':-.25,
			'signal_max':1.25,
			'plot_mode':'smFRET'
		})

	def normalized_defaults(self):
		self.prefs.add_dictionary({
			'xlabel_text':r'Initial Normalized Intensity',
			'ylabel_text':r'Final Normalized Intensity',
			'signal_min':-.25,
			'signal_max':1.25,
			'plot_mode':'ND Normalized'
		})
	
	def raw_defaults(self):
		self.prefs.add_dictionary({
			'xlabel_text':r'Initial Intensity (A.U.)',
			'ylabel_text':r'Final Intensity (A.U.)',
			'signal_min':-500,
			'signal_max':10000,
			'plot_mode':'ND Raw'
		})


	def plot(self,fig,ax):
		''' Plots transition density plot in ax
		plot style prefs are very similar to those for 2D hist (eg double floor)
		'''
		## Decide if we should be plotting at all
		if not self.maven.data.ncolors == 2:
			logger.error('more than 2 colors not implemented')
			# return

		## Setup
		if len(fig.axes)>1:
			[aa.remove() for aa in fig.axes[1:]]
		ax.cla()
		self.fix_ax(fig,ax)
		self.plot_mode = self.prefs['plot_mode']

		## This is protected and shouldn't crash
		d1,d2,N = self.get_neighbor_data()
		x,y,z = self.gen_histogram(d1,d2)

		### Plotting
		cm = self.colormap()
		if self.prefs['color_ceiling'] == self.prefs['color_floor']:
			self.prefs['color_floor'] = 0.0
			self.prefs['color_ceiling'] = np.ceil(z.max())
		vmin = self.prefs['color_floor']
		vmax = self.prefs['color_ceiling']


		if self.prefs['hist_log']:
			from matplotlib.colors import LogNorm
			pc = ax.imshow(z.T, cmap=cm, origin='lower',interpolation='none',
				  extent=[x.min(),x.max(),x.min(),x.max()],
				  norm = LogNorm(vmin=np.max((1./(1+d1.size),vmin)),vmax=vmax))
		else:
			pc = ax.imshow(z.T, cmap=cm, origin='lower',interpolation='none',
				  extent=[x.min(),x.max(),x.min(),x.max()],vmin=vmin,vmax=vmax)

		# for pcc in pc.collections:
			# pcc.set_edgecolor("face")

		### Colorbar
		ext='neither'
		if vmin > z.min():
			ext = 'min'
		from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
		ax_divider = make_axes_locatable(ax)
		cax = ax_divider.append_axes("right", size="%d%%"%(self.prefs['colorbar_widthpercent']), pad="%d%%"%(self.prefs['colorbar_padpercent']))
		cb = fig.colorbar(pc,extend=ext,cax=cax)

		if not self.prefs['hist_log']:
			cbticks = np.linspace(vmin,vmax,self.prefs['color_nticks'])
			cbticks = np.array(self.best_ticks(0,self.prefs['color_ceiling'],self.prefs['color_nticks']))
		else:
			cbticks = np.logspace(np.log10(self.prefs['color_floor']),np.log10(self.prefs['color_ceiling']),self.prefs['color_nticks'])

		cbticks = cbticks[cbticks > self.prefs['color_floor']]
		cbticks = cbticks[cbticks < self.prefs['color_ceiling']]
		cbticks = np.append(self.prefs['color_floor'], cbticks)
		cbticks = np.append(cbticks, self.prefs['color_ceiling'])

		cb.set_ticks(cbticks)
		cb.set_ticklabels(["{0:.{1}f}".format(cbtick, self.prefs['color_decimals']) for cbtick in cbticks])
		for label in cb.ax.get_yticklabels():
			label.set_family(self.prefs['font'])

		dpr = self.devicePixelRatio()
		cb.ax.yaxis.set_tick_params(labelsize=self.prefs['tick_fontsize']/dpr,
			direction=self.prefs['tick_direction'],
			width=self.prefs['tick_linewidth']/dpr,
			length=self.prefs['tick_length_major']/dpr)
		for asp in ['top','bottom','left','right']:
			cb.ax.spines[asp].set_linewidth(self.prefs['axes_linewidth']/dpr)
		cb.outline.set_linewidth(self.prefs['axes_linewidth']/dpr)
		cb.solids.set_edgecolor('face')
		cb.solids.set_rasterized(True)

		pos = fig.axes[1].get_position()
		fig.axes[1].set_position([self.prefs['color_xloc'],pos.y0,pos.width,pos.height])

		self.d1 = d1
		self.N = N
		self.garnish(fig,ax)
		fig.canvas.draw()

	def garnish(self,fig,ax):
		## Fix up the plot
		dpr = self.devicePixelRatio()
		fs = self.prefs['label_fontsize']/dpr
		font = {'family': self.prefs['font'],
			'size': fs,
			'va':'top'}
		ax.set_xlabel(self.prefs['xlabel_text'],fontdict=font)
		ax.set_ylabel(self.prefs['ylabel_text'],fontdict=font)
		ax.yaxis.set_label_coords(self.prefs['ylabel_offset'], 0.5)
		ax.xaxis.set_label_coords(0.5, self.prefs['xlabel_offset'])

		ax.set_xticks(self.best_ticks(self.prefs['signal_min'],self.prefs['signal_max'],self.prefs['signal_nticks']))
		ax.set_yticks(self.best_ticks(self.prefs['signal_min'],self.prefs['signal_max'],self.prefs['signal_nticks']))

		bbox_props = dict(boxstyle="square", fc="w", alpha=1.0,lw=1./dpr)
		lstr = 'n = %d'%(self.d1.size)
		if self.prefs['textbox_nmol']:
			lstr = 'N = %d'%(self.N)

		ax.annotate(lstr,xy=(self.prefs['textbox_x'],self.prefs['textbox_y']),
			xycoords='axes fraction', ha='right', color='k', bbox=bbox_props,
			fontsize=self.prefs['textbox_fontsize']/dpr)

		from matplotlib import ticker
		fd = {'rotation':self.prefs['xlabel_rotate'], 'ha':'center'}
		if fd['rotation'] != 0: fd['ha'] = 'right'
		ax.set_xticklabels(["{0:.{1}f}".format(x, self.prefs['xlabel_decimals']) for x in ax.get_xticks()], fontdict=fd)
		ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos: "{0:.{1}f}".format(x,self.prefs['xlabel_decimals'])))

		fd = {'rotation':self.prefs['ylabel_rotate']}
		ax.set_yticklabels(["{0:.{1}f}".format(y, self.prefs['ylabel_decimals']) for y in ax.get_yticks()], fontdict=fd)
		ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: "{0:.{1}f}".format(y,self.prefs['ylabel_decimals'])))

	def get_neighbor_data(self):
		''' gets transitions (esp when there is an model_result)

		Returns
		-------
		d1 : np.ndarray
			pre datapoints
		d2 : np.ndarray
			post datapoints
		N : int
			number of traces used in this calculations

		Notes
		-----
		nskip : int
			number of frames to skip for before and after datapoints
		hist_rawsignal : bool
			if true use raw data, otherwise will use idealized

		'''
		#try:
		if 1:
			if self.plot_mode == 'smFRET':
				index = 1
			else:
				index = self.prefs['plot_channel']

			dpb = self.get_plot_data()[:,:,index][np.nonzero(self.maven.data.flag_ons)[0]]
			N = dpb.shape[0]
			nskip = self.prefs['nskip']
			# d = np.array([[dpb[i,:-nskip],dpb[i,nskip:]] for i in range(dpb.shape[0])])
			d1 = dpb[:,:-nskip].copy()
			d2 = dpb[:,nskip:].copy()

			if not self.maven.modeler.model is None:
				v = self.get_idealized_data()[np.nonzero(self.maven.data.flag_ons)[0]]
				if not self.prefs['hist_rawsignal']:
					d1 = v[:,:-nskip].copy()
					d2 = v[:,nskip:].copy()
				jump = np.array([np.abs(v[i,1:]-v[i,:-1])>0 for i in range(v.shape[0])])
				if nskip > 1:
					jump = jump[:,:-(nskip-1)]
				## this mehod below will double/triple/quadruple/... count. don't use it.
				# jump = np.array([np.abs(v[i,nskip:]-v[i,:-nskip]) > 0 for i in range(v.shape[0])])
				d1 = d1[jump]
				d2 = d2[jump]
			d1 = d1.flatten()
			d2 = d2.flatten()
			keep = np.isfinite(d1)*np.isfinite(d2)

			return d1[keep],d2[keep],N
		#except:
		else:
			return np.array(()),np.array(()),0

	def gen_histogram(self,d1,d2):
		''' Makes 2D histogram of data from `get_neighbor_data`

		Returns
		-------
		x : np.ndarray
			(signal_nbins) the x axis bin locations
		y : np.ndarray
			(signal_nbins) the y axis bin locations
		z : np.ndarray
			(signal_nbins,signal_nbins) the histogram

		'''
		from scipy.ndimage import gaussian_filter
		from scipy.interpolate import RectBivariateSpline

		## make histogram
		x = np.linspace(self.prefs['signal_min'],self.prefs['signal_max'],self.prefs['signal_nbins'])
		z,hx,hy = np.histogram2d(d1,d2,bins=[x.size,x.size],range=[[x.min(),x.max()],[x.min(),x.max()]])

		## smooth histogram
		z = gaussian_filter(z,(self.prefs['hist_smoothx'],self.prefs['hist_smoothy']))

		## interpolate histogram
		if self.prefs['hist_interp_res'] > 0:
			interspline = RectBivariateSpline(x,x,z.T)
			x = np.linspace(self.prefs['signal_min'],self.prefs['signal_max'],self.prefs['hist_interp_res'])
			z = interspline(x,x).T
			z[z<0] = 0.

		if self.prefs['hist_normalize']:
			z /= z.max()

		return x,x,z
