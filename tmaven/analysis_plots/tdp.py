import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import ticker
from PyQt5.QtWidgets import QApplication

default_prefs = {
'fig_height':4.0,
'fig_width':4.0,
'subplots_top':0.95,
'subplots_right':0.83,
'subplots_left':0.18,
'axes_topright':True,
'xlabel_offset':-0.24,
'ylabel_offset':-0.34,

'fret_min':-.25,
'fret_max':1.25,
'fret_nbins':101,
'fret_nticks':7,

'hist_smoothx':1.,
'hist_smoothy':1.,
'hist_interp_res':800,
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
}


def get_neighbor_data(window):
	''' gets transitions (esp when there is an model_result)

	Parameters
	----------
	window : popout_plot_container
		the `gui.plots.tdp` container

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
	fpb = window.gui.analysis_plots.get_plot_fret()[:,:,1]
	N = fpb.shape[0]
	nskip = window.prefs['nskip']
	d = np.array([[fpb[i,:-nskip],fpb[i,nskip:]] for i in range(fpb.shape[0])])

	if not window.gui.maven.modeler.model is None:
		v = window.gui.analysis_plots.get_idealized_data(signal=True)
		# vv = np.array([[v[i,:-nskip],v[i,nskip:]] for i in range(v.shape[0])])
		vv = np.array([[v[i,:-1],v[i,1:]] for i in range(v.shape[0])])[:,:,:-nskip]

		for i in range(d.shape[0]):
			# d[i,:,vv[i,0]==vv[i,1]] = np.array((np.nan,np.nan))
			d[i,:,:-1][:,vv[i,0]==vv[i,1]] = np.nan
			d[i,:,-nskip:] = np.nan
			if not window.prefs['hist_rawsignal']:
				xx = np.nonzero(vv[i,0]!=vv[i,1])[0]
				d[i,0,xx] = v[i,xx]
				d[i,1,xx] = v[i,xx+1]

	d1 = d[:,0].flatten()
	d2 = d[:,1].flatten()
	cut = np.isfinite(d1)*np.isfinite(d2)

	return d1[cut],d2[cut],N

def gen_histogram(window,d1,d2):
	''' Makes 2D histogram of data from `get_neighbor_data`

	Parameters
	----------
	window : popout_plot_container
		the `gui.plots.tdp` container

	Returns
	-------
	x : np.ndarray
		(fret_nbins) the x axis bin locations
	y : np.ndarray
		(fret_nbins) the y axis bin locations
	z : np.ndarray
		(fret_nbins,fret_nbins) the histogram

	'''
	from scipy.ndimage import gaussian_filter
	from scipy.interpolate import interp2d

	p = window.prefs

	## make histogram
	x = np.linspace(p['fret_min'],p['fret_max'],p['fret_nbins'])
	z,hx,hy = np.histogram2d(d1,d2,bins=[x.size,x.size],range=[[x.min(),x.max()],[x.min(),x.max()]])

	## smooth histogram
	z = gaussian_filter(z,(p['hist_smoothx'],p['hist_smoothy']))

	## interpolate histogram
	f =  interp2d(x,x,z, kind='cubic')
	x = np.linspace(p['fret_min'],p['fret_max'],p['hist_interp_res'])
	z = f(x,x)
	z[z<0] = 0.

	if p['hist_normalize']:
		z /= z.max()

	return x,x,z

def colormap(window):
	if window.prefs['color_dbl']:
		cm = double_floor_cmap(window)
	else:
		try:
			import colorcet as cc
			cm = cc.cm.__getattr__(window.prefs['color_cmap'])
		except:
			try:
				cm = plt.cm.__dict__[window.prefs['color_cmap']]
			except:
				cm = plt.cm.rainbow
				window.prefs['color_cmap'] = 'rainbow'
		try:
			cm.set_under(prefs['color_floorcolor'])
		except:
			cm.set_under('w')
	return cm

def double_floor_cmap(window):
	from matplotlib import colors

	try:
		colorbar_bins = 1000
		cutoff1 = window.prefs['color_floor']
		cutoff2 = window.prefs['color_dblfloor']
		cutoff3 = window.prefs['color_ceiling']
		if cutoff2 < cutoff1:
			cutoff2 = cutoff1+1e-6
		if cutoff3 < cutoff2:
			cutoff3 = cutoff2+1e-6

		cmap_dbl = plt.cm.get_cmap(window.prefs['color_cmap'], int(colorbar_bins * (cutoff3 - cutoff2)+1))
		part_map = np.array([cmap_dbl(i)[:3] for i in range(cmap_dbl.N)]).T

		part_floor = np.zeros((3,int((cutoff2 - cutoff1)*colorbar_bins))) + np.array(colors.to_rgb(window.prefs['color_dblfloorcolor']))[:,None]

		cmap_list = np.hstack((part_floor,part_map)).T
		cmap_final = colors.ListedColormap(cmap_list, name = 'cmap_final')

		start_beige = window.prefs['color_floorcolor']
		end_red = part_map[:,-1]
		cmap_final.set_under(start_beige)
		cmap_final.set_over(end_red)

	except:
		cmap_final = plt.cm.rainbow

	return cmap_final

def plot(window):
	''' Plots transition density plot in window.ax

	Parameters
	----------
	window : popout_plot_container
		the `gui.plots.tdp` container

	Notes
	-----
	plot style prefs are very similar to those for 2D hist (eg double floor)

	'''

	pp = window.prefs
	window.plot.cla()
	window.plot.resize_figure()
	QApplication.instance().processEvents()

	dpr = window.plot.devicePixelRatio()

	if window.gui.maven.data.ncolors != 2 or window.gui.maven.data.nmol == 0:
		return

	d1,d2,N = get_neighbor_data(window)
	x,y,z = gen_histogram(window,d1,d2)

	### Plotting
	cm = colormap(window)

	if pp['color_ceiling'] == pp['color_floor']:
		pp['color_floor'] = 0.0
		pp['color_ceiling'] = np.ceil(z.max())

	vmin = pp['color_floor']
	vmax = pp['color_ceiling']

	if pp['hist_log']:
		pc = window.plot.ax.imshow(z.T, cmap=cm, origin='lower',interpolation='none',extent=[x.min(),x.max(),x.min(),x.max()],vmin=np.max((1./d1.size,vmin)),vmax=vmax,norm = LogNorm())
	else:
		pc = window.plot.ax.imshow(z.T, cmap=cm, origin='lower',interpolation='none',extent=[x.min(),x.max(),x.min(),x.max()],vmin=vmin,vmax=vmax)

	# for pcc in pc.collections:
		# pcc.set_edgecolor("face")

	### Colorbar
	ext='neither'
	if vmin > z.min():
		ext = 'min'
	if len(window.plot.figure.axes) == 1:
		cb = window.plot.figure.colorbar(pc,extend=ext)
	else:
		window.plot.figure.axes[1].cla()
		cb = window.plot.figure.colorbar(pc,cax=window.plot.figure.axes[1],extend=ext)

	if not pp['hist_log']:
		cbticks = np.linspace(vmin,vmax,pp['color_nticks'])
		cbticks = np.array(window.plot.best_ticks(0,pp['color_ceiling'],pp['color_nticks']))
	else:
		cbticks = np.logspace(np.log10(pp['color_floor']),np.log10(pp['color_ceiling']),pp['color_nticks'])

	cbticks = cbticks[cbticks > pp['color_floor']]
	cbticks = cbticks[cbticks < pp['color_ceiling']]
	cbticks = np.append(pp['color_floor'], cbticks)
	cbticks = np.append(cbticks, pp['color_ceiling'])

	cb.set_ticks(cbticks)
	cb.set_ticklabels(["{0:.{1}f}".format(cbtick, pp['color_decimals']) for cbtick in cbticks])
	for label in cb.ax.get_yticklabels():
		label.set_family(pp['font'])

	cb.ax.yaxis.set_tick_params(labelsize=pp['tick_fontsize']/dpr,direction=pp['tick_direction'],width=pp['tick_linewidth']/dpr,length=pp['tick_length_major']/dpr)
	for asp in ['top','bottom','left','right']:
		cb.ax.spines[asp].set_linewidth(pp['axes_linewidth']/dpr)
	cb.outline.set_linewidth(pp['axes_linewidth']/dpr)
	cb.solids.set_edgecolor('face')
	cb.solids.set_rasterized(True)

	pos = window.plot.figure.axes[1].get_position()
	window.plot.figure.axes[1].set_position([window.prefs['color_xloc'],pos.y0,pos.width,pos.height])

	####################################################
	####################################################

	fs = pp['label_fontsize']/dpr
	font = {
		'family': pp['font'],
		'size': fs,
		'va':'top'
	}
	window.plot.ax.set_xlabel(pp['xlabel_text'],fontdict=font)
	window.plot.ax.set_ylabel(pp['ylabel_text'],fontdict=font)
	window.plot.ax.yaxis.set_label_coords(pp['ylabel_offset'], 0.5)
	window.plot.ax.xaxis.set_label_coords(0.5, pp['xlabel_offset'])

	window.plot.ax.set_xticks(window.plot.best_ticks(pp['fret_min'],pp['fret_max'],pp['fret_nticks']))
	window.plot.ax.set_yticks(window.plot.best_ticks(pp['fret_min'],pp['fret_max'],pp['fret_nticks']))

	bbox_props = dict(boxstyle="square", fc="w", alpha=1.0,lw=1./dpr)
	lstr = 'n = %d'%(d1.size)
	if pp['textbox_nmol']:
		lstr = 'N = %d'%(N)

	window.plot.ax.annotate(lstr,xy=(pp['textbox_x'],pp['textbox_y']),xycoords='axes fraction', ha='right', color='k', bbox=bbox_props, fontsize=pp['textbox_fontsize']/dpr)

	fd = {'rotation':pp['xlabel_rotate'], 'ha':'center'}
	if fd['rotation'] != 0: fd['ha'] = 'right'
	window.plot.ax.set_xticklabels(["{0:.{1}f}".format(x, pp['xlabel_decimals']) for x in window.plot.ax.get_xticks()], fontdict=fd)
	window.plot.ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos: "{0:.{1}f}".format(x,pp['xlabel_decimals'])))

	fd = {'rotation':pp['ylabel_rotate']}
	window.plot.ax.set_yticklabels(["{0:.{1}f}".format(y, pp['ylabel_decimals']) for y in window.plot.ax.get_yticks()], fontdict=fd)
	window.plot.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: "{0:.{1}f}".format(y,pp['ylabel_decimals'])))

	window.plot.draw()
