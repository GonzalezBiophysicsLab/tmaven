import numpy as np
import logging
logger = logging.getLogger(__name__)

from .base import controller_base_analysisplot

class controller_model_vbstates(controller_base_analysisplot):
	def __init__(self,maven):
		super().__init__(maven)
		self.defaults()

	def defaults(self):
		self.prefs.add_dictionary({
			'bar_color':'steelblue',
			'bar_edgecolor':'black',

			'states_low':1,
			'states_high':10,

			'count_nticks':5,

			'xlabel_text':r'N$_{\rm{States}}$',
			'ylabel_nticks':4,
			'ylabel_text':r'$N_{\rm trajectories}$',

			'textbox_x':0.96,
			'textbox_y':0.93,
			'textbox_fontsize':8.0,
			'textbox_nmol':True,

			'fig_width':2.0,
			'fig_height':2.0,
			'subplots_top':0.97,
			'subplots_left':0.25,
			'subplots_bottom':0.3,
		})

	def plot(self,fig,ax):
		''' Plot the distribution of states for traces modeled with vbFRET

		After running vbFRET with model selection on a collection of traces, each trace will have a different number of states. This creates a bar graph of the number of traces modeled to have a certain number of states (Max evidence).

		Notes
		-----
		states_low : int
			lowest number of states in bar graph
		states_high : int
			highest number of states in bar graph
		bar_color : str
			color of bars
		bar_edgecolor : str
			color of bar edges
		'''
		## Decide if we should be plotting at all

		## Setup
		ax.cla()
		self.fix_ax(fig,ax)

		try:
			self.N = 0
			if not self.maven.modeler.model is None:
				if self.maven.modeler.model.type[:3] == 'vb ':
					ns = np.arange(self.prefs['states_low'],self.prefs['states_high']+1).astype('i')
					nstates = np.array([r.mu.size for r in self.maven.modeler.model.hmms])
					y = np.array([np.sum(nstates == i) for i in ns])

					from matplotlib import colors
					bcolor = self.prefs['bar_color']
					if not colors.is_color_like(bcolor):
						bcolor = 'steelblue'
					ecolor = self.prefs['bar_edgecolor']
					if not colors.is_color_like(ecolor):
						ecolor = 'black'
					ax.bar(ns,y,width=1.0,color=bcolor,edgecolor=ecolor)

				ax.set_xticks(ns)
				ylim = ax.get_ylim()
				ticks = self.best_ticks(0.,ylim[1],self.prefs['count_nticks'])
				ax.set_yticks(ticks)
				self.N = y.sum()
		except:
			pass

		self.garnish(fig,ax)
		fig.canvas.draw()

	def garnish(self,fig,ax):
		dpr =self.devicePixelRatio()
		ax.set_xlabel(self.prefs['xlabel_text'],fontsize=self.prefs['label_fontsize']/dpr,labelpad=self.prefs['xlabel_offset'])
		ax.set_ylabel(self.prefs['ylabel_text'],fontsize=self.prefs['label_fontsize']/dpr,labelpad=self.prefs['ylabel_offset'])

		bbox_props = dict(boxstyle="square", fc="w", alpha=1.0,lw=self.prefs['axes_linewidth']/dpr)
		lstr = 'N = %d'%(int(self.N))

		ax.annotate(lstr,xy=(self.prefs['textbox_x'],self.prefs['textbox_y']),xycoords='axes fraction',ha='right',color='k',bbox=bbox_props,fontsize=self.prefs['textbox_fontsize']/dpr)
