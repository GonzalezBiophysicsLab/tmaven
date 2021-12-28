import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from PyQt5.QtWidgets import  QComboBox
from PyQt5.QtWidgets import QApplication


default_prefs = {
	'bar_color':'steelblue',
	'bar_edgecolor':'black',

# 	'states_low':1,
# 	'states_high':10,
#
# 	'count_nticks':5,
#
	'xlabel_rotate':45.,
# 	'xlabel_text':r'N$_{\rm{States}}$',
# 	'ylabel_nticks':4,
# 	'ylabel_text':r'$N_{\rm trajectories}$',
#
# 	'textbox_x':0.96,
# 	'textbox_y':0.93,
# 	'textbox_fontsize':8.0,
# 	'textbox_nmol':True,
#
# 	'fig_width':2.0,
# 	'fig_height':2.0,
# 	'subplots_top':0.97,
# 	'subplots_left':0.2
}


def setup(window):
	''' setup the plot window'''
	window.combo_plot = QComboBox()
	window.combo_plot.addItems(['Length','PB Time','Intensity','Variance'])
	window.combo_color = QComboBox()
	colors = ['Sum']
	colors = colors + ['Color %d'%(i+1) for i in range(window.gui.maven.data.ncolors)]
	colors = colors + ['Relative %d'%(i+1) for i in range(window.gui.maven.data.ncolors)]
	window.combo_color.addItems(colors)

	window.buttonbox.insertWidget(2,window.combo_plot)
	window.buttonbox.insertWidget(3,window.combo_color)
	window.combo_plot.setCurrentIndex(0)

def plot(window):
	'''Draw the plots as a function of data source

	This function largely draws everything but the data, then calls a separate function to draw the data

	Notes
	-----
	length
		plots the "lengths" (post point - pre point)

	PB Time
		plots the photobleaching time (post point)

	Intensity
		plots the mean of `window.gui.maven.data.corrected` for specified color option
	Variance
		plots the variance of `window.gui.maven.data.corrected` for specified color option
	'''

	pp = window.prefs
	window.plot.cla()
	window.plot.resize_figure()
	QApplication.instance().processEvents()

	dpr = window.plot.devicePixelRatio()

	ns = len(window.gui.maven.smd.source_names)
	masks = np.zeros((ns,window.gui.maven.data.nmol),dtype='bool')
	class_mask = window.gui.maven.selection.get_toggled_mask()
	for i in range(ns):
		masks[i] = np.bitwise_and(np.array([s == i for s in np.arange(window.gui.maven.data.nmol)],dtype='bool'),class_mask)

	method = window.combo_plot.currentText()
	color = window.combo_color.currentText()
	if method == 'Length':
		plot_length(window,masks,color)
	elif method == 'PB Time':
		plot_pbtime(window,masks,color)
	elif method == 'Intensity':
		plot_mu(window,masks,color)
	elif method == 'Variance':
		plot_var(window,masks,color)

	# fontdict = {'rotation':pp['xlabel_rotate'], 'ha':'right'}
	# window.plot.ax.set_xticklabels(np.arange(ns),window.gui.maven.smd.source_names,fontdict=fontdict)

	# if not window.gui.maven.modeler.model is None:
# 		if window.gui.maven.modeler.model.type == 'vb':
# 			ns = np.arange(pp['states_low'],pp['states_high']+1).astype('i')
# 			nstates = np.array([r.mu.size for r in window.gui.maven.modeler.model.results])
# 			y = np.array([np.sum(nstates == i) for i in ns])
#
# 			try:
# 				bcolor = 'steelblue'
# 				ecolor = 'black'
# 				if list(cnames.keys()).count(pp['bar_color']) > 0:
# 					bcolor = pp['bar_color']
# 				if list(cnames.keys()).count(pp['bar_edgecolor']) > 0:
# 					ecolor = pp['bar_edgecolor']
# 				window.plot.bar(ns,y,width=1.0,color=bcolor,edgecolor=ecolor)
# 			except:
# 				pass
#
# 		window.plot.ax.set_xticks(ns)
# 		ylim = window.plot.ax.get_ylim()
# 		ticks = window.plot.best_ticks(0.,ylim[1],pp['count_nticks'])
# 		window.plot.ax.set_yticks(ticks)
#
# 		window.plot.ax.set_xlabel(pp['xlabel_text'],fontsize=pp['label_fontsize']/dpr,labelpad=popplot.prefs['xlabel_offset'])
# 		window.plot.ax.set_ylabel(pp['ylabel_text'],fontsize=pp['label_fontsize']/dpr,labelpad=popplot.prefs['ylabel_offset'])
#
#
# 		bbox_props = dict(boxstyle="square", fc="w", alpha=1.0,lw=pp['axes_linewidth']/dpr)
# 		lstr = 'N = %d'%(int(y.sum()))
#
# 		window.plot.ax.annotate(lstr,xy=(pp['textbox_x'],pp['textbox_y']),xycoords='axes fraction',ha='right',color='k',bbox=bbox_props,fontsize=pp['textbox_fontsize']/dpr)
#
	window.plot.draw()

def color_interpreter(window,color):
	'''Parses color option from combo_color'''
	nc = window.gui.maven.data.ncolors
	if color == 'Sum':
		return window.gui.maven.data.corrected.sum(-1)
	elif color.startswith('Color'):
		channel  = int(color.split(' ')[-1]) - 1
		return window.gui.maven.data.corrected[:,:,channel]
	elif color.startswith('Relative'):
		channel  = int(color.split(' ')[-1]) - 1
		return window.gui.maven.data.calc_fret()[:,:,channel]

def plot_length(window,masks,color):
	'''plots the lengths (post point - pre point)'''
	ns = masks.shape[0]
	datas = [None,]*ns
	for i in range(ns):
		datas[i] = window.gui.maven.data.post_list[masks[i]] - window.gui.maven.data.pre_list[masks[i]]
	window.plot.ax.violinplot(datas,positions=np.arange(ns),showmeans=True)

def plot_pbtime(window,masks,color):
	''' plots the photobleaching time (post point)'''
	ns = masks.shape[0]
	datas = [None,]*ns
	for i in range(ns):
		datas[i] = window.gui.maven.data.post_list[masks[i]]
	window.plot.ax.violinplot(datas,positions=np.arange(ns),showmeans=True)

def plot_mu(window,masks,color):
	''' plots the average of the intensity data'''
	pp = window.prefs
	ns = masks.shape[0]
	datas = [None,]*ns
	data = color_interpreter(window,color)
	for i in range(ns):
		nz = np.nonzero(masks[i])[0]
		ni = nz.size
		datas[i] = np.zeros(ni)
		for j in range(ni):
			xij = data[nz[j],window.gui.maven.data.pre_list[nz[j]]:window.gui.maven.data.post_list[nz[j]]]
			datas[i][j] = np.mean(xij)
	window.plot.ax.violinplot(datas,positions=np.arange(ns),showmeans=True)

def plot_var(window,masks,color):
	'''plots the variance of the intensity data'''
	pp = window.prefs
	ns = masks.shape[0]
	datas = [None,]*ns
	data = color_interpreter(window,color)
	for i in range(ns):
		nz = np.nonzero(masks[i])[0]
		ni = nz.size
		datas[i] = np.zeros(ni)
		for j in range(ni):
			xij = data[nz[j],window.gui.maven.data.pre_list[nz[j]]:window.gui.maven.data.post_list[nz[j]]]
			datas[i][j] = np.var(xij)
	window.plot.ax.violinplot(datas,positions=np.arange(ns),showmeans=True)
