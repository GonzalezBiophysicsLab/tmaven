import numba as nb
import numpy as np
import logging
logger = logging.getLogger(__name__)

from .base import controller_base_analysisplot

class controller_fret_hist2d(controller_base_analysisplot):
	''' Plots 2D histogram (heat map) in window.ax
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
	def __init__(self,maven):
		super().__init__(maven)
		self.defaults()

	def defaults(self):
		self.prefs.add_dictionary({
			'fig_height':2.500000,
			'fig_width':2.500000,
			'subplots_wspace':0.030000,
			'subplots_hspace':0.040000,
			'subplots_bottom':0.170000,
			'subplots_top':0.970000,
			'subplots_right':0.85,
			'subplots_left':0.22,
			'axes_topright':'True',

			'colorbar_widthpercent':5,
			'colorbar_padpercent':2,

			'tick_fontsize':10.000000,
			'xlabel_text':r'Time (s)',
			'ylabel_text':r'E$_{\rm{FRET}}$',
			'label_fontsize':10.000000,
			'xlabel_offset':-0.14,
			'ylabel_offset':-0.34,

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

		})


	def interpolate_histogram(self,z):
		'''interpolate/smooth the histogram `z`'''
		from scipy.ndimage import gaussian_filter,uniform_filter,median_filter
		from scipy.interpolate import interp2d

		# smooth histogram
		x = np.linspace(0.,z.shape[0]-1,z.shape[0])
		y = np.linspace(self.prefs['fret_min'],self.prefs['fret_max'],z.shape[1])
		try:
			if self.prefs['hist_smooth_med']:
				z = median_filter(z,(3,3))
			z = gaussian_filter(z,(self.prefs['hist_smoothx'],self.prefs['hist_smoothy']))

			## interpolate histogram - interp2d is backwards...
			if self.prefs['hist_interp_res'] > 0:
				f =  interp2d(y,x,z, kind='cubic')
				x = np.linspace(0.,z.shape[0]-1,self.prefs['hist_interp_res'])#*self.prefs['time_dt']
				y = np.linspace(self.prefs['fret_min'],self.prefs['fret_max'],self.prefs['hist_interp_res']+1)
				z = f(y,x)
				z[z<0] = 0.

			if self.prefs['hist_normalizeframe']:
				z /= np.nanmax(z,axis=1)[:,None]+1
			z /= np.nanmax(z)

			if self.prefs['hist_log']:
				z = np.log10(z)
		except:
			pass
		return x,y,z

	def get_data(self):
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
		try:
			fpb = self.get_plot_fret()[:,:].copy()
			flag = False

			if (not self.maven.modeler.model is None) and self.prefs['sync_postsync']:
				flag = True

			if flag: ## postsync time
				viterbis = self.get_idealized_data()
				chains = self.get_chain_data()
				vi = self.prefs['sync_hmmstate_1']
				vj = self.prefs['sync_hmmstate_2']

				if self.prefs['sync_singledwell']:
					synclist = gen_sync_list_single(chains,vi,vj)
				else:
					dt = self.prefs['sync_preframe']
					n = self.prefs['time_nbins']
					synclist = gen_sync_list_fixed(chains,vi,vj,dt,n-dt+1)

				nmol = np.unique(synclist[:,0]).size
				npoints = synclist.shape[0]
				out = histogram_sync_list(synclist, fpb, self.prefs['time_nbins'],
					self.prefs['sync_preframe'], self.prefs['fret_min'],
					self.prefs['fret_max'], self.prefs['fret_nbins'])

			else: ## not post-sync
				if self.prefs['sync_start']:
					fpb = sync_start(fpb,self.maven.data.pre_list, self.maven.data.post_list)
				nmol = fpb.shape[0] - np.all(np.isnan(fpb),axis=1).sum()
				npoints = nmol
				out = histogram_raw(fpb, self.prefs['time_nbins'], 0,
					self.prefs['fret_min'], self.prefs['fret_max'],
					self.prefs['fret_nbins'])
			return out,nmol,npoints

		except:
			return np.zeros((self.prefs['time_nbins'],self.prefs['fret_nbins'])),0,0

	def plot(self,fig,ax):
		## Decide if we should be plotting at all
		if not self.maven.data.ncolors == 2:
			logger.error('more than 2 colors not implemented')

		## Setup
		if len(fig.axes)>1:
			[aa.remove() for aa in fig.axes[1:]]
		self.fix_ax(fig,ax)

		hist,nmol,npoints = self.get_data()
		x,y,z = self.interpolate_histogram(hist)

		### Plotting
		cm = self.colormap()
		vmin = self.prefs['color_floor']
		vmax = self.prefs['color_ceiling']

		## imshow is backwards
		tau = self.prefs['time_dt']
		if self.prefs['sync_postsync']:
			tmin = (0-self.prefs['sync_preframe'])*tau + self.prefs['time_shift']
			tmax = (self.prefs['time_nbins']-self.prefs['sync_preframe'])*tau + self.prefs['time_shift']
		else:
			tmin = self.prefs['time_shift']
			tmax = self.prefs['time_nbins']*tau + self.prefs['time_shift']

		pc = ax.imshow(z.T, cmap=cm, origin='lower',interpolation='none',
			extent=[tmin,tmax,y.min(),y.max()], aspect='auto', vmin=vmin, vmax=vmax)

		# for pcc in pc.collections:
			# pcc.set_edgecolor("face")

		## Colorbar
		ext='neither'
		if vmin > z.min():
			ext = 'min'
		from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
		ax_divider = make_axes_locatable(ax)
		cax = ax_divider.append_axes("right", size="%d%%"%(self.prefs['colorbar_widthpercent']), pad="%d%%"%(self.prefs['colorbar_padpercent']))
		cb = fig.colorbar(pc,extend=ext,cax=cax)

		cbticks = np.linspace(0,vmax,self.prefs['color_nticks'])
		cbticks = np.array(self.best_ticks(0,self.prefs['color_ceiling'],self.prefs['color_nticks']))
		cbticks = cbticks[cbticks > self.prefs['color_floor']]
		cbticks = cbticks[cbticks < self.prefs['color_ceiling']]
		cbticks = np.append(self.prefs['color_floor'], cbticks)
		cbticks = np.append(cbticks, self.prefs['color_ceiling'])
		cb.set_ticks(cbticks)
		cb.set_ticklabels(["%.2f"%(cbt) for cbt in cbticks])
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

		self.npoints = npoints
		self.nmol = nmol
		self.tmin = tmin
		self.tmax = tmax
		self.garnish(fig,ax)

		fig.canvas.draw()

	def garnish(self,fig,ax):
		dpr = self.devicePixelRatio()
		tticks = self.best_ticks(self.tmin,self.tmax,self.prefs['time_nticks'])
		fticks = self.best_ticks(self.prefs['fret_min'],self.prefs['fret_max'],self.prefs['fret_nticks'])
		ax.set_xticks(tticks)
		ax.set_yticks(fticks)
		fs = self.prefs['label_fontsize']/dpr
		font = {'family': self.prefs['font'],
			'size': fs,
			'va':'top'}
		ax.set_xlabel(self.prefs['xlabel_text'],fontdict=font)
		ax.set_ylabel(self.prefs['ylabel_text'],fontdict=font)

		ax.yaxis.set_label_coords(self.prefs['ylabel_offset'], 0.5)
		ax.xaxis.set_label_coords(0.5, self.prefs['xlabel_offset'])

		bbox_props = dict(boxstyle="square", fc="w", alpha=1.0,lw=1./dpr)
		lstr = 'n = %d'%(self.npoints)
		if self.prefs['textbox_nmol']:
			lstr = 'N = %d'%(self.nmol)

		ax.annotate(lstr,xy=(self.prefs['textbox_x'],self.prefs['textbox_y']),
			xycoords='axes fraction',ha='right',color='k',
			bbox=bbox_props,fontsize=self.prefs['textbox_fontsize']/dpr)


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

@nb.njit
def sync_start(fret,pre,post):
	for i in range(fret.shape[0]):
		if pre[i] < post[i]+1:
			yy = fret[i,pre[i]:post[i]+1].copy()
			fret[i] *= np.nan
			fret[i,:yy.size] = yy
	return fret
