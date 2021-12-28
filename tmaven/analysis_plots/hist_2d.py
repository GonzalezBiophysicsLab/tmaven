from PyQt5.QtWidgets import QInputDialog,QApplication
import numpy as np
import numba as nb
import matplotlib.pyplot as plt


default_prefs = {
	'fig_height':2.500000,
	'fig_width':2.500000,
	'subplots_wspace':0.030000,
	'subplots_hspace':0.040000,
	'subplots_bottom':0.170000,
	'subplots_top':0.970000,
	'subplots_right':0.89,
	'subplots_left':0.22,
	'axes_topright':'True',

	'tick_fontsize':10.000000,
	'xlabel_text':r'Time (s)',
	'ylabel_text':r'E$_{\rm{FRET}}$',
	'label_fontsize':10.000000,
	'xlabel_offset':-0.15,
	'ylabel_offset':-0.40,

	'textbox_x':0.96,
	'textbox_y':0.93,
	'textbox_fontsize':7.,
	'textbox_nmol':True,

	# 'time_min':0,
	# 'time_max':200,
	'time_nbins':200,
	'time_shift':0.0,
	'time_nticks':5,
	'time_dt':1.,

	'sync_start':True,
	'sync_postsync':True,
	'sync_preframe':50,
	'sync_hmmstate_1':0,
	'sync_hmmstate_2':-1,
	# 'sync_LHS':False,
	'sync_singledwell':True,
	# 'sync_ignoreends':True,

	'fret_min':-.2,
	'fret_max':1.2,
	'fret_nbins':61,
	'fret_nticks':7,

	'hist_smooth_med':True,
	'hist_smoothx':.5,
	'hist_smoothy':.5,
	'hist_normalizeframe':False,
	'hist_interp_res':0,
	'hist_log':False,


	'color_cmap':'jet',
	'color_floorcolor':r'#FFFFCC',
	'color_dblfloorcolor':'white',
	'color_dbl':True,
	'color_ceiling':0.8,
	'color_floor':0.05,
	'color_nticks':5,
	'color_dblfloor':.2
}


@nb.njit
def gen_jumplist(viterbis,vi,vj):
	''' find jumps for synchronization point '''
	njumps = 0
	for i in range(viterbis.shape[0]):
		for t in range(viterbis.shape[1]-1):
			vit = viterbis[i,t]
			vjt = viterbis[i,t+1]
			if (not np.isnan(vit)) and (not np.isnan(vjt)) and vit != vjt:
				njumps += 1

	jumplist = np.zeros((njumps,4),dtype=type(1))
	current = 0
	for i in range(viterbis.shape[0]):
		for t in range(viterbis.shape[1]-1):
			vit = viterbis[i,t]
			vjt = viterbis[i,t+1]
			if (not np.isnan(vit)) and (not np.isnan(vjt)) and vit != vjt:
				jumplist[current] = np.array((i,t,vit,vjt))
				current += 1

	if vi < 0 and vj < 0:
		return jumplist,np.arange(jumplist.shape[0])
	elif vi < 0 and vj >= 0:
		return jumplist,np.nonzero(jumplist[:,3]==vj)[0]
	elif vi >= 0 and vj < 0:
		return jumplist,np.nonzero(jumplist[:,2]==vi)[0]
	else:
		return jumplist,np.nonzero(np.bitwise_and(jumplist[:,2]==vi,jumplist[:,3]==vj))[0]

@nb.njit
def gen_sync_list_fixed(viterbis,ii,jj,npre,npost):
	''' find jumps for synchronization point '''
	jumplist,jumpind = gen_jumplist(viterbis,ii,jj)
	synclist = np.zeros((jumpind.size,4),dtype=type(ii))
	for i in range(jumpind.size):
		synclist[i,0] = jumplist[jumpind[i],0]
		synclist[i,1] = jumplist[jumpind[i],1] - npre
		synclist[i,2] = jumplist[jumpind[i],1]
		synclist[i,3] = jumplist[jumpind[i],1] + npost
	return synclist

@nb.njit
def gen_sync_list_single(viterbis,ii,jj):
	''' find jumps for synchronization point '''
	jumplist,jumpind = gen_jumplist(viterbis,ii,jj)
	synclist = np.zeros((jumpind.size,4),dtype=type(ii))
	synclist[:,3] = viterbis.shape[1] - 1
	for i in range(jumpind.size):
		synclist[i,0] = jumplist[jumpind[i],0]
		if jumpind[i]-1 >= 0:
			if jumplist[jumpind[i]-1,0] == jumplist[jumpind[i],0]:
				synclist[i,1] = jumplist[jumpind[i]-1,1] + 1
		synclist[i,2] = jumplist[jumpind[i],1]
		if jumpind[i]+1 < jumplist.shape[0]:
			if jumplist[jumpind[i]+1,0] == jumplist[jumpind[i],0]:
				synclist[i,3] = jumplist[jumpind[i]+1,1]
	return synclist

# @nb.njit
# def gen_sync_list_recurr(viterbis,ii,jj):
# 	jumplist,jumpind = gen_jumplist(viterbis,ii,jj)
# 	synclist = np.zeros((jumpind.size,4),dtype=type(ii))
# 	synclist[:,3] = viterbis.shape[1] - 1
# 	offset = 1
# 	for i in range(jumpind.size):
# 		synclist[i,0] = jumplist[jumpind[i],0]
# 		if i > 0:
# 			if jumplist[jumpind[i-1]-offset,0] == jumplist[jumpind[i],0]:
# 				synclist[i,1] = jumplist[jumpind[i-1]-offset,1] + 1
# 		synclist[i,2] = jumplist[jumpind[i],1]
# 		if i+1 < jumpind.size and jumpind[i+1]+1 < jumplist.shape[0]:
# 			if jumplist[jumpind[i+1]+offset,0] == jumplist[jumpind[i],0]:
# 				synclist[i,3] = jumplist[jumpind[i+1]+offset,1]
# 	return synclist

# @nb.njit
# def gen_firstpassage_list(viterbis,vi,vj):
# 	nmol,nt = viterbis.shape
#
# 	npassages = 0
# 	for nmoli in range(nmol):
# 		recording = False
# 		for nti in range(nt):
# 			if (not recording) and (viterbis[nmoli,nti] == vi):
# 				recording = True
# 			if recording and (viterbis[nmoli,nti] == vj):
# 				recording = False
# 				npassages += 1
#
# 	out = np.zeros((npassages,3),dtype=type(vi))
#
# 	npi = 0
# 	for nmoli in range(nmol):
# 		recording = False
# 		for nti in range(nt-1):
# 			if (not recording) and (viterbis[nmoli,nti] == vi):
# 				recording = True
# 				start = nti
# 			if recording and (viterbis[nmoli,nti] == vj) and (viterbis[nmoli,nti+1] != vj):
# 				end = nti
# 				recording = False
# 				out[npi] = np.array((nmoli,start,end))
# 				npi += 1
# 		if recording:
# 			out[npi] = np.array((nmoli,start,nt-1))
# 			npi += 1
# 	return out

@nb.njit
def histogram_sync_list(syncs,data,xbins,xsync,ymin,ymax,ybins):
	''' calculate synchronzied 2D histogram from data'''
	out = np.zeros((xbins+1,ybins))
	for synci in range(syncs.shape[0]):
		sync = syncs[synci]
		for t in range(sync[1],sync[3]):
			x = t - sync[2] + xsync
			if x > xbins:
				break
			elif x >= 0:
				d = data[sync[0],t]
				if (not np.isnan(d)) and (d >= ymin) and (d < ymax):
					yind = int(((d-ymin)/(ymax-ymin))*(ybins))
					out[x,yind] += 1
	return out

@nb.njit
def histogram_raw(data,xbins,xsync,ymin,ymax,ybins):
	''' calculate 2D histogram from data'''
	out = np.zeros((xbins+1,ybins))
	for nmoli in range(data.shape[0]):
		for x in range(xbins):
			t = x+xsync
			if t < 0 or t > data.shape[1]:
				break
			else:
				d = data[nmoli,t]
				if (not np.isnan(d)) and (d >= ymin) and (d < ymax):
					yind = int(((d-ymin)/(ymax-ymin))*(ybins))
					out[x,yind] += 1
	return out

# @nb.njit
# def histogram_firstpassage_list(fplist,data,xbins,xsync,ymin,ymax,ybins):
# 	out = np.zeros((xbins+1,ybins))
# 	for fpi in range(fplist.shape[0]):
# 		fp = fplist[fpi]
# 		for x in range(xbins):
# 			t = fp[1]-xsync+x
# 			if t > fp[2]-1:
# 				break
# 			else:
# 				d = data[fp[0],t]
# 				if np.isnan(d) and d >= ymin and d < ymax:
# 					yind = int(((d-ymin)/(ymax-ymin))*(ybins))
# 					out[x,yind] += 1
# 	return out


def interpolate_histogram(window,z):
	'''interpolate/smooth the histogram `z`'''
	from scipy.ndimage import gaussian_filter,uniform_filter,median_filter
	from scipy.interpolate import interp2d

	pp = window.prefs

	# smooth histogram
	x = np.linspace(0.,z.shape[0]-1,z.shape[0])
	y = np.linspace(pp['fret_min'],pp['fret_max'],z.shape[1])
	if pp['hist_smooth_med']:
		z = median_filter(z,(3,3))
	z = gaussian_filter(z,(window.prefs['hist_smoothx'],window.prefs['hist_smoothy']))

	## interpolate histogram - interp2d is backwards...
	if pp['hist_interp_res'] > 0:
		f =  interp2d(y,x,z, kind='cubic')
		x = np.linspace(0.,z.shape[0]-1,pp['hist_interp_res'])#*window.prefs['time_dt']
		y = np.linspace(pp['fret_min'],pp['fret_max'],pp['hist_interp_res']+1)
		z = f(y,x)
		z[z<0] = 0.

	if pp['hist_normalizeframe']:
		z /= np.nanmax(z,axis=1)[:,None]+1
	z /= np.nanmax(z)

	if pp['hist_log']:
		z = np.log10(z)

	return x,y,z

@nb.njit
def sync_start(fret,pre,post):
	for i in range(fret.shape[0]):
		if pre[i] < post[i]+1:
			yy = fret[i,pre[i]:post[i]+1].copy()
			fret[i] *= np.nan
			fret[i,:yy.size] = yy
	return fret

def get_data(window):
	'''Prepare data for making 2D histogram

	Deals with synchronization settings too

	Parameters
	----------
	window : popout_plot_container
		the `gui.plots.hist2d` container

	Notes
	-----
	sync_postsync : bool
		whether to postsynchronize traces if model_result is available
	sync_hmmstate_1 : int
		* all traces leaving this state number on LHS of the synchronization point
		* a value of -1 means all states
	sync_hmmstate_2 : int
		* all traces entering this state number on RHS of the synchronization point
		* a value of -1 means all states
	sync_singledwell : bool
		if True only include data once when synchronizing
	sync_preframe : int
		number of frames on the LHS of the synchronization point

	'''
	pp = window.prefs
	fpb = window.gui.analysis_plots.get_plot_fret()[:,:,1].copy()
	flag = False

	if (not window.gui.maven.modeler.model is None) and window.prefs['sync_postsync']:
		flag = True

	if flag: ## postsync time
		viterbis = window.gui.analysis_plots.get_idealized_data()

		vi = pp['sync_hmmstate_1']
		vj = pp['sync_hmmstate_2']
		if pp['sync_singledwell']:
			synclist = gen_sync_list_single(viterbis,vi,vj)
		else:
			dt = pp['sync_preframe']
			n = pp['time_nbins']
			synclist = gen_sync_list_fixed(viterbis,vi,vj,dt,n-dt+1)

		nmol = np.unique(synclist[:,0]).size
		npoints = synclist.shape[0]
		out = histogram_sync_list(synclist, fpb, pp['time_nbins'], pp['sync_preframe'], pp['fret_min'], pp['fret_max'], pp['fret_nbins'])

	else: ## not post-sync
		if window.prefs['sync_start']:
			fpb = sync_start(fpb,window.gui.maven.data.pre_list,window.gui.maven.data.post_list)

		nmol = fpb.shape[0] - np.all(np.isnan(fpb),axis=1).sum()
		npoints = nmol
		out = histogram_raw(fpb, pp['time_nbins'], 0, pp['fret_min'], pp['fret_max'], pp['fret_nbins'])
	return out,nmol,npoints

def colormap(window):
	'''Get a matplotlib colormap from string in prefs['color_cmap']'''

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
	''' Creates a double floored colormap

	Parameters
	----------
	window : popout_plot_container
		the `gui.plots.hist2d` container

	Notes
	-----
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
	''' Plots 2D histogram (heat map) in window.ax

	Parameters
	----------
	window : popout_plot_container
		the `gui.plots.hist2d` container

	Notes
	-----
	color_floor : double
		lowest floor value for colomap
	color_dblfloor : double
		2nd floor value for colomap
	color_ceiling : double
		highest value (ceiling) for colomap
	color_nticks : int
		number of ticks in colorbar
	time_dt : double
		length of each frame
	sync_preframe : double
		number of frames before synchronization point
	sync_postsync : bool
		whether to post synchronize plot (if `window.gui.maven.modeler.model` available)
	time_shift : double
		value added to all numbers on xaxis (time)
	time_nbins : int
		number of 2d histogram bins in the time dimension (x)
	time_nticks : int
		number of ticks on x axis
	fret_min : double
		min of y axis
	fret_max : double
		max of y axis
	fret_nticks : int
		number of ticks on y axis

	'''
	if window.gui.maven.data.nmol == 0:
		return
	pp = window.prefs
	window.plot.resize_figure()
	window.plot.cla()
	QApplication.instance().processEvents()


	dpr = window.plot.devicePixelRatio()

	hist,nmol,npoints = get_data(window)
	x,y,z = interpolate_histogram(window,hist)

	### Plotting
	cm = colormap(window)

	vmin = pp['color_floor']
	vmax = pp['color_ceiling']

	## imshow is backwards
	tau = pp['time_dt']
	if pp['sync_postsync']:
		tmin = (0-pp['sync_preframe'])*tau + pp['time_shift']
		tmax = (pp['time_nbins']-pp['sync_preframe'])*tau + pp['time_shift']
	else:
		tmin = pp['time_shift']
		tmax = pp['time_nbins']*tau + pp['time_shift']

	pc = window.plot.ax.imshow(z.T, cmap=cm, origin='lower',interpolation='none',extent=[tmin,tmax,y.min(),y.max()],aspect='auto',vmin=vmin,vmax=vmax)

	# for pcc in pc.collections:
		# pcc.set_edgecolor("face")

	## Colorbar
	ext='neither'
	if vmin > z.min():
		ext = 'min'
	if len(window.plot.figure.axes) == 1:
		cb = window.plot.figure.colorbar(pc,extend=ext)
	else:
		window.plot.figure.axes[1].cla()
		cb = window.plot.figure.colorbar(pc,cax=window.plot.figure.axes[1],extend=ext)

	cbticks = np.linspace(0,vmax,pp['color_nticks'])
	cbticks = np.array(window.plot.best_ticks(0,pp['color_ceiling'],pp['color_nticks']))
	cbticks = cbticks[cbticks > pp['color_floor']]
	cbticks = cbticks[cbticks < window.prefs['color_ceiling']]
	cbticks = np.append(pp['color_floor'], cbticks)
	cbticks = np.append(cbticks, pp['color_ceiling'])
	cb.set_ticks(cbticks)
	cb.set_ticklabels(["%.2f"%(cbt) for cbt in cbticks])
	for label in cb.ax.get_yticklabels():
		label.set_family(pp['font'])

	cb.ax.yaxis.set_tick_params(labelsize=pp['tick_fontsize']/dpr,direction=pp['tick_direction'],width=pp['tick_linewidth']/dpr,length=pp['tick_length_major']/dpr)
	for asp in ['top','bottom','left','right']:
		cb.ax.spines[asp].set_linewidth(pp['axes_linewidth']/dpr)
	cb.outline.set_linewidth(pp['axes_linewidth']/dpr)
	cb.solids.set_edgecolor('face')
	cb.solids.set_rasterized(True)

	####################################################
	####################################################

	tticks = window.plot.best_ticks(tmin,tmax,pp['time_nticks'])
	fticks = window.plot.best_ticks(pp['fret_min'],pp['fret_max'],pp['fret_nticks'])
	window.plot.ax.set_xticks(tticks)
	window.plot.ax.set_yticks(fticks)

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

	bbox_props = dict(boxstyle="square", fc="w", alpha=1.0,lw=1./dpr)
	lstr = 'n = %d'%(npoints)
	if pp['textbox_nmol']:
		lstr = 'N = %d'%(nmol)

	window.plot.ax.annotate(lstr,xy=(pp['textbox_x'],pp['textbox_y']),xycoords='axes fraction',ha='right',color='k',bbox=bbox_props,fontsize=pp['textbox_fontsize']/dpr)

	window.plot.draw()
