import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import cnames
from matplotlib import colors
from PyQt5.QtWidgets import QApplication

default_prefs = {
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
}

def plot(window):
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
	pp = window.prefs
	window.plot.cla()
	window.plot.resize_figure()
	QApplication.instance().processEvents()

	dpr = window.plot.devicePixelRatio()

	if not window.gui.maven.modeler.model is None:
		if window.gui.maven.modeler.model.type == 'vb':
			ns = np.arange(pp['states_low'],pp['states_high']+1).astype('i')
			nstates = np.array([r.mu.size for r in window.gui.maven.modeler.model.results])
			y = np.array([np.sum(nstates == i) for i in ns])

			bcolor = pp['bar_color']
			if not colors.is_color_like(bcolor):
				bcolor = 'steelblue'
			ecolor = pp['bar_edgecolor']
			if not colors.is_color_like(ecolor):
				ecolor = 'black'
			window.plot.ax.bar(ns,y,width=1.0,color=bcolor,edgecolor=ecolor)

		window.plot.ax.set_xticks(ns)
		ylim = window.plot.ax.get_ylim()
		ticks = window.plot.best_ticks(0.,ylim[1],pp['count_nticks'])
		window.plot.ax.set_yticks(ticks)

		window.plot.ax.set_xlabel(pp['xlabel_text'],fontsize=pp['label_fontsize']/dpr,labelpad=pp['xlabel_offset'])
		window.plot.ax.set_ylabel(pp['ylabel_text'],fontsize=pp['label_fontsize']/dpr,labelpad=pp['ylabel_offset'])


		bbox_props = dict(boxstyle="square", fc="w", alpha=1.0,lw=pp['axes_linewidth']/dpr)
		lstr = 'N = %d'%(int(y.sum()))

		window.plot.ax.annotate(lstr,xy=(pp['textbox_x'],pp['textbox_y']),xycoords='axes fraction',ha='right',color='k',bbox=bbox_props,fontsize=pp['textbox_fontsize']/dpr)

		window.plot.draw()
