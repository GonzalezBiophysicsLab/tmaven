import numpy as np
import logging
logger = logging.getLogger(__name__)

from ..prefs import prefs_object

class controller_base_analysisplot(object):
	def __init__(self,maven):
		self.maven = maven
		##These are the default preferences for all plots
		self.prefs = prefs_object()
		self.prefs.add_dictionary({'fig_width':2.5,
			'fig_height':2.5,
			'label_fontsize':10.0,
			'ylabel_offset':-0.165,
			'xlabel_offset':-0.25,
			'font':'Arial',
			'axes_linewidth':1.0,
			'axes_topright':False,
			'tick_fontsize':10.0,
			'tick_length_minor':2.0,
			'tick_length_major':4.0,
			'tick_linewidth':1.0,
			'tick_direction':'out',
			'subplots_left':0.125,
			'subplots_right':0.97,
			'subplots_top':0.97,
			'subplots_bottom':0.155,
			'subplots_hspace':0.04,
			'subplots_wspace':0.03,
			'color_cmap':'jet',
			'color_floorcolor':r'#FFFFCC',
			'color_dblfloorcolor':'white',
			'color_dbl':True,
			'color_ceiling':0.8,
			'color_floor':0.05,
			'color_nticks':5,
			'color_dblfloor':.2,
			'plot_channel':0})

	def plot(self,fig,ax):
		## override this
		pass

	def fix_ax(self,fig,ax):
		fig.set_figwidth(self.prefs['fig_width'])
		fig.set_figheight(self.prefs['fig_height'])

		fig.subplots_adjust(left   = self.prefs['subplots_left'],
			right  = self.prefs['subplots_right'],
			top    = self.prefs['subplots_top'],
			bottom = self.prefs['subplots_bottom'],
			hspace = self.prefs['subplots_hspace'],
			wspace = self.prefs['subplots_wspace'])

		dpr = self.devicePixelRatio()
		for aa in fig.axes:
			for asp in ['top','bottom','left','right']:
				aa.spines[asp].set_linewidth(self.prefs['axes_linewidth']/dpr)
				if asp in ['top','right']:
					aa.spines[asp].set_visible(self.prefs['axes_topright'])

				tickdirection = self.prefs['tick_direction']
				if not tickdirection in ['in','out']:
					tickdirection = 'in'
				aa.tick_params(labelsize=self.prefs['tick_fontsize']/dpr,
					axis='both',
					direction=tickdirection,
					width=self.prefs['tick_linewidth']/dpr,
					length=self.prefs['tick_length_minor']/dpr)
				aa.tick_params(axis='both',
					which='major',
					length=self.prefs['tick_length_major']/dpr)
				for label in aa.get_xticklabels():
					label.set_family(self.prefs['font'])
				for label in aa.get_yticklabels():
					label.set_family(self.prefs['font'])


		fig.canvas.draw()

	def cla(self,fig,ax):
		for a in fig.axes:
			a.cla()
		self.fix_ax(fig,ax)

	def best_ticks(self,ymin,ymax,nticks):
		m = nticks
		if m <= 0: return ()
		if ymax <= ymin: return ()
		delta = ymax-ymin

		d = 10.0**(np.floor(np.log10(delta)))
		ind = np.arange(1,10)
		ind = np.concatenate((1./ind[::-1],ind))
		di = d*ind
		for i in range(ind.size):
			if np.floor(delta/di[i]) < m:
				s = di[i]
				break
		y0 = np.ceil(ymin/s)*s
		delta = ymax - y0

		d = 10.0**(np.floor(np.log10(delta)))
		ind = np.arange(1,10)
		ind = np.concatenate((1./ind[::-1],ind))
		di = d*ind
		for i in range(ind.size):
			if np.floor(delta/di[i]) < m:
				s = di[i]
				break
		y0 = np.ceil(ymin/s)*s
		delta = ymax - y0
		n = np.floor(delta/s+1e-10)
		return y0 + np.arange(n+1)*s

	def get_plot_data(self):
		''' Get the data for plotting

		Has to be in a toggled class. Removes photobleaching pre and post times

		Returns
		-------
		dpb : np.ndarray
		 	fret (nmol toggled, ntime, ncolors)
		'''

		if self.plot_mode in ["ND Relative", "smFRET"]:
			dpb = self.maven.calc_relative()
		else:
			dpb = self.maven.data.corrected

		for i in range(self.maven.data.nmol): ## photobleach molecules
			dpb[i,:self.maven.data.pre_list[i]] = np.nan
			dpb[i,self.maven.data.post_list[i]:] = np.nan
		mask = self.maven.selection.get_toggled_mask() ## only chosen classes
		dpb = dpb[mask]
		return dpb

	def get_idealized_data(self):
		''' Get toggled idealized data for plotting
		this might be busted by not considering flag_ons?
		'''
		if self.maven.modeler.model is None:
			return None
		return self.maven.modeler.model.idealized.copy()
		
	def get_chain_data(self):
		''' Get toggled idealized data for plotting
		this might be busted by not considering flag_ons?
		'''
		if self.maven.modeler.model is None:
			return None
		return self.maven.modeler.model.chain.copy()
	
	def devicePixelRatio(self):
		## maybe look this up from the figure?
		return 1.

	def colormap(self):
		'''Get a matplotlib colormap from string in prefs['color_cmap']'''
		import matplotlib.pyplot as plt
		if self.prefs['color_dbl']:
			cm = self.double_floor_cmap()
		else:
			try:
				import colorcet as cc
				cm = cc.cm.__getattr__(self.prefs['color_cmap'])
			except:
				try:
					cm = plt.cm.__dict__[self.prefs['color_cmap']]
				except:
					cm = plt.cm.rainbow
					self.prefs['color_cmap'] = 'rainbow'
			try:
				cm.set_under(self.prefs['color_floorcolor'])
			except:
				cm.set_under('w')
		return cm

	def double_floor_cmap(self):
		''' Creates a double floored colormap

		color_floor : double
			first floor of colormap
		color_dblfloor : double
			middle floor of colormap
		color_ceiling : double
			top of colormap
		color_cmap : str
			name of matplotlib colormap to use between middle and ceiling
		color_dblfloorcolor : str
			color of area between first and middle floor
		color_floorcolor : str
			color of area below first floor
		'''
		from matplotlib import colors
		import matplotlib.pyplot as plt

		try:
			colorbar_bins = 1000
			cutoff1 = self.prefs['color_floor']
			cutoff2 = self.prefs['color_dblfloor']
			cutoff3 = self.prefs['color_ceiling']
			if cutoff2 < cutoff1:
				cutoff2 = cutoff1+1e-6
			if cutoff3 < cutoff2:
				cutoff3 = cutoff2+1e-6

			cmap_dbl = plt.get_cmap(self.prefs['color_cmap'], int(colorbar_bins * (cutoff3 - cutoff2)+1))
			part_map = np.array([cmap_dbl(i)[:3] for i in range(cmap_dbl.N)]).T

			part_floor = np.zeros((3,int((cutoff2 - cutoff1)*colorbar_bins))) + np.array(colors.to_rgb(self.prefs['color_dblfloorcolor']))[:,None]

			cmap_list = np.hstack((part_floor,part_map)).T
			cmap_final = colors.ListedColormap(cmap_list, name = 'cmap_final')

			start_beige = self.prefs['color_floorcolor']
			end_red = part_map[:,-1]
			cmap_final.set_under(start_beige)
			cmap_final.set_over(end_red)
		except:
			cmap_final = plt.cm.rainbow

		return cmap_final
