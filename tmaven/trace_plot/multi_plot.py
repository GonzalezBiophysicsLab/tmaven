import logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

import matplotlib
matplotlib.use('Qt5Agg') ## forces Qt5 early on...
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
from matplotlib.widgets import  RectangleSelector
import matplotlib.font_manager
from PyQt5.QtWidgets import QSizePolicy,QVBoxLayout,QWidget, QMenu, QAction,QApplication
from PyQt5.QtCore import pyqtSignal,QObject,QSize
from PyQt5.QtGui import QScreen
import numpy as np


class colors(object):
	fret_green =  "#0EA52F" # '#6AD531'
	fret_red =  "#EE0000"  # '#D5415A' 
	fret_blue = "#023BF9"

class multi_canvas(FigureCanvas):

	def __init__(self,gui):
		self.gui = gui

		app = QApplication.instance()
		screen = app.screens()[0]
		self.dpi = screen.physicalDotsPerInch()
		self.dpr = screen.devicePixelRatio()

		matplotlib.rcParams['savefig.format'] = 'pdf'
		matplotlib.rcParams['pdf.fonttype'] = 42
		matplotlib.rcParams['ps.fonttype'] = 42
		matplotlib.rcParams['agg.path.chunksize'] = 10000
		matplotlib.rcParams['figure.facecolor'] = 'white'
		matplotlib.rcParams['savefig.dpi'] = self.dpi*self.dpr
		matplotlib.rcParams['figure.dpi'] = self.dpi*self.dpr

		p = self.gui.maven.prefs
		fig,self.ax = plt.subplots(2,2,gridspec_kw={'width_ratios':[6,1]})
		super().__init__(fig)

		self.plot_mode = 'Relative'

		defpref = self.default_prefs()
		if self.gui.lightdark_mode == 'light':
			defpref['plot.bg_color'] = 'white'
		elif self.gui.lightdark_mode == 'dark':
			defpref['plot.bg_color'] = '#353535'
		self.gui.maven.prefs.add_dictionary(defpref)
		self.gui.preferences_viewer.prefs_model.layoutChanged.emit()

		plt.close(self.figure)
		self.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))
		self.figure.set_dpi(self.dpi*self.dpr)
		self.figure.set_figwidth(p['plot.fig_width'])
		self.figure.set_figheight(p['plot.fig_height'])
		self.ax = self.ax.reshape((2,2))
		
		self.toolbar = NavigationToolbar(self,None)
		self.toolbar.setIconSize(QSize(24,24))
		self.toolbar.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed))
		self.flag_drawing = False

		self.initialize_plots()
		self.mpl_connect('button_press_event', self.mouse_handler)

		self.keyPressEvent = self.gui.keyPressEvent
		self.wheelEvent = self.gui.slider_select.wheelEvent
		self.toolbar.wheelEvent = self.gui.slider_select.wheelEvent
		self.gui.wheelEvent = self.gui.slider_select.wheelEvent
		# self.resize_figure()
		# self.setGeometry(0,0,int(p['plot.fig_width']*self.dpi),int(p['plot.fig_height']*self.dpi))

		# self.resize_figure()
		# self.update_plots(-1)


	def default_prefs(self):
		default_prefs = {
			'plot.time_dt':1.0,
			'plot.axes_linewidth':1.0,
			'plot.axes_topright':False,
			'plot.tick_fontsize':10.0,
			'plot.tick_length_minor':2.0,
			'plot.tick_length_major':4.0,
			'plot.tick_linewidth':1.0,
			'plot.tick_direction':'out',
			'plot.font':'Arial',

			'plot.subplots_left':0.12,
			'plot.subplots_right':0.99,
			'plot.subplots_top':0.98,
			'plot.subplots_bottom':0.12,
			'plot.subplots_hspace':0.04,
			'plot.subplots_wspace':0.03,

			'plot.fig_width':5.0,
			'plot.fig_height':3.5,

			'plot.fret_min':float(-.25),
			'plot.fret_max':float(1.25),
			'plot.intensity_min':float(-1000.0),
			'plot.intensity_max':float(10000.0),
			'plot.channel_colors':[colors.fret_green,colors.fret_red,"cyan","purple"],
			'plot.relative_colors':[colors.fret_green,colors.fret_red,"cyan","purple"],
			# 'plot.fret_color':colors.fret_blue,
			'plot.line_linewidth':0.4,
			'draw_hist_linewidth':1.0,
			'draw_hist_show':True,
			'plot.line_alpha_pb':0.25,

			'plot.line_alpha':0.9,
			'plot.idealized_color':'k',
			'plot.idealized_linewidth':0.7,
			'plot.idealized_alpha':0.9,
			'plot.label_fontsize':10.0,
			'plot.ylabel_offset':-0.16,
			'plot.xlabel_offset':-0.16,

			'plot.xlabel_text1':'Time (s)',
			'plot.xlabel_text2':'Probability',
			'plot.ylabel_text1':r'Intensity (a.u.)',
			'plot.ylabel_text2':r'Relative',
			'plot.fret_pbzero':True,
			'plot.fret_nticks':7,
			'plot.intensity_nticks':6,
			'plot.time_rotate':0.0,
			'plot.time_min':0.0,
			'plot.time_max':1.0,
			'plot.time_nticks':6,
			'plot.time_offset':0.0,
			'plot.time_decimals':0,
			'plot.intensity_decimals':0,
			'plot.fret_decimals':2,
			'normalize_intensities':False,
			'plot.bg_color':'white'
		}
		return default_prefs
	
	def smFRET_prefs(self):
		self.gui.maven.prefs['plot.channel_colors'] = [colors.fret_green,colors.fret_red,"cyan","purple"]
		self.gui.maven.prefs['plot.relative_colors'] = ["none",colors.fret_blue,"none","none"]
		self.gui.maven.prefs['plot.ylabel_text1'] = r'Intensity (A.U.)'
		self.gui.maven.prefs['plot.ylabel_text2'] = r'E$_{\rm{FRET}}$'
		self.update_plots(self.gui.index)

	def ndrelative_prefs(self):
		self.gui.maven.prefs['plot.channel_colors'] = [colors.fret_green,colors.fret_red,"cyan","purple"]
		self.gui.maven.prefs['plot.relative_colors'] = [colors.fret_green,colors.fret_red,"cyan","purple"]
		self.gui.maven.prefs['plot.ylabel_text1'] = r'Intensity (A.U.)'
		self.gui.maven.prefs['plot.ylabel_text2'] = r'Relative'
		self.update_plots(self.gui.index)
	
	def ndraw_prefs(self):
		self.gui.maven.prefs['plot.channel_colors'] = [colors.fret_green,colors.fret_red,"cyan","purple"]
		self.gui.maven.prefs['plot.relative_colors'] = ["none","none","none","none"]
		self.gui.maven.prefs['plot.ylabel_text1'] = r'Intensity (A.U.)'
		self.gui.maven.prefs['plot.ylabel_text2'] = r''
		self.gui.maven.prefs['plot.fig_height'] /= 2.
		self.update_plots(self.gui.index)

	def normalized_prefs(self):
		self.gui.maven.prefs['plot.channel_colors'] = [colors.fret_green,colors.fret_red,"cyan","purple"]
		self.gui.maven.prefs['plot.relative_colors'] = [colors.fret_green,colors.fret_red,"cyan","purple"]
		self.gui.maven.prefs['plot.ylabel_text1'] = r'Intensity (A.U.)'
		self.gui.maven.prefs['plot.ylabel_text2'] = r'Normalised Intensity'
		self.update_plots(self.gui.index)
	
	def sizeHint(self):
		qs = QSize(int(self.gui.maven.prefs['plot.fig_width']*self.dpi), int(self.gui.maven.prefs['plot.fig_height']*self.dpi))
		return qs

	def build_menu(self):
		self.menu_trajplot = QMenu('Plot',self.gui)

		action_redraw = QAction('Refresh', self.gui)
		action_redraw.triggered.connect(lambda event: self.redrawplot())
		action_save = QAction('Save Figure', self.gui)
		action_save.triggered.connect(lambda event: self.saveplot())

		for ac in [action_redraw,action_save]:
			self.menu_trajplot.addAction(ac)
		# from ....interface.stylesheet import ss_qmenu
		# self.menu_trajplot.setStyleSheet(ss_qmenu)

	## Plot initial data to set aesthetics
	def initialize_plots(self):
		''' Setup the plots

		Notes
		-----
		Also creates blank lines for set data
		'''

		int_ind,rel_ind,model_ind = self.get_axis_inds()


		grid_on = self.ax[int_ind,0].yaxis._major_tick_kw['gridOn']

		## clear everything
		[[aaa.cla() for aaa in aa] for aa in self.ax]

		## resize
		self.figure.patch.set_facecolor("%s"%(self.gui.maven.prefs['plot.bg_color']))
		self.setStyleSheet("background-color:%s;"%(self.gui.maven.prefs['plot.bg_color']))
		[[aaa.patch.set_color(self.gui.maven.prefs['plot.bg_color']) for aaa in aa] for aa in self.ax]
		self.figure.set_dpi(self.dpi*self.dpr)
		self.figure.set_figwidth(self.gui.maven.prefs['plot.fig_width'])
		self.figure.set_figheight(self.gui.maven.prefs['plot.fig_height'])

		## Make it so that certain plots zoom together
		self.ax[int_ind,0].sharey(self.ax[int_ind,1])
		self.ax[rel_ind,0].sharey(self.ax[rel_ind,1])
		self.ax[int_ind,0].sharex(self.ax[rel_ind,0])
		self.ax[int_ind,1].sharex(self.ax[rel_ind,1])

		## Redraw everything
		self.set_ticks()
		self.set_axis_limits()
		self.set_axis_labels()
		self.update_axis_geometry()

		for i in range(self.gui.maven.data.ncolors):
			for ls in [':','-',':']:
				self.ax[int_ind,0].plot(np.random.rand(self.gui.maven.data.ntime), ls=ls)
				self.ax[rel_ind,0].plot(np.random.rand(self.gui.maven.data.ntime), ls=ls)

		for i in range(self.gui.maven.data.ncolors): ## Idealized Relatives.... e.g., for vbFRET idealized
			self.ax[int_ind,0].plot(np.zeros(self.gui.maven.data.ntime)+np.nan)
			self.ax[rel_ind,0].plot(np.zeros(self.gui.maven.data.ntime)+np.nan)

		for i in range(self.gui.maven.data.ncolors):
			self.ax[int_ind,1].plot(np.random.rand(100))
			self.ax[rel_ind,1].plot(np.random.rand(100))

		self.set_linestyles()

		if grid_on:
			self.ax[int_ind,0].grid(True)
			self.ax[rel_ind,0].grid(True)
		
		# self.draw()
		# self.redrawplot()

	def saveplot(self):
		''' convenience function to save figure '''
		self.toolbar.save_figure()

	def redrawplot(self,*args,**kwargs):
		''' convenience function to redraw figure '''
		if not self.gui.maven.data is None:
			self.update_plots(self.gui.index)
			self.resize_figure()
	#
	# def redrawplot2(self,*args,**kwargs):
	# 	''' convenience function to redraw figure '''
	# 	self.update_plots(self.gui.index)
	# 	self.resize_figure()
	def softredraw(self):
		self.update_plots(self.gui.index)

	def resize_figure(self):
		''' Update figure size

		Updates figure canvas size to specified in preferences, redraws everything, then updates blitted regions
		'''
		## OKAY -- call this when the mainwindow that houses the traj_plot_container is resized, because this widget never resizes otherwise
		self.figure.set_dpi(self.dpi*self.dpr)
		self.figure.set_figwidth(self.gui.maven.prefs['plot.fig_width'])
		self.figure.set_figheight(self.gui.maven.prefs['plot.fig_height'])

		self.updateGeometry()
		self.update_axis_geometry()

		self.draw()
		self.update_blits()
		QApplication.instance().processEvents()
		logger.info('resized plot - {}  {}'.format(self.figure.get_size_inches(),self.figure.get_dpi()))
		if self.figure.get_dpi() != self.dpi*self.dpr:
			self.resize_figure()


	def best_ticks(self,ymin,ymax,nticks):
		''' Determine the best tick values to use

		Keep ticks reasonable for base 10 numbers

		Parameters
		----------
		ymin : float
			lowest plotted value
		ymax : float
			highest plotted value
		nticks : int
			number of ticks to use

		Returns
		-------
		out : np.ndarray
			`nticks` tick values to use between `ymin` and `ymax`
		'''
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

	def pretty_plot(self):
		''' Updates plot to specified style

		Updates pixel ratio, axis spines, ticks, fonts according to settings in preferences
		'''

		p = self.gui.maven.prefs

		## Set the ticks/labels so that they look nice
		for aa in self.ax:
			for aaa in aa:
				for asp in ['top','bottom','left','right']:
					aaa.spines[asp].set_linewidth(p['plot.axes_linewidth'])
					if asp in ['top','right']:
						aaa.spines[asp].set_visible(p['plot.axes_topright'])

				tickdirection = p['plot.tick_direction']
				if not tickdirection in ['in','out']: tickdirection = 'in'
				aaa.tick_params(labelsize=p['plot.tick_fontsize'], axis='both', direction=tickdirection , width=p['plot.tick_linewidth'], length=p['plot.tick_length_minor'])
				aaa.tick_params(axis='both',which='major',length=p['plot.tick_length_major'])
				for label in aaa.get_xticklabels():
					label.set_family(p['plot.font'])
				for label in aaa.get_yticklabels():
					label.set_family(p['plot.font'])

	def update_tick_decimals(self,ax,axis='x',n_decimals=2,rotation=0,ha='center'):
		''' Set number of decimals used on tick labels

		Sometimes tick labels have too many or too few decimal points. This function standardizes the number of decimals for each tick label.

		Parameters
		----------
		ax : matplotlib.axes._subplots.AxesSubplot
			The matplotlib axis to work on
		axis : str
			'x' or 'y' axis of `self.ax` to update decimals of
		n_decimals : int
			number of numbers after decimal point to keep
		rotation : float
			degrees to rotate the tick labels by
		ha : str
			horizontal alignment of tick labels
			* 'right'
			* 'center'
			* 'left'

		'''
		fd = {'rotation':rotation, 'ha':ha}
		if axis == 'x':
			xt = ax.get_xticks()
			ax.set_xticklabels(["{0:.{1}f}".format(x,n_decimals) for x in xt],fontdict=fd)
			ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos: "{0:.{1}f}".format(x,n_decimals)))
		if axis == 'y':
			yt = ax.get_yticks()
			ax.set_yticklabels(["{0:.{1}f}".format(y,n_decimals) for y in yt],fontdict=fd)
			ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: "{0:.{1}f}".format(y,n_decimals)))

	def update_axis_geometry(self):
		''' updates spacing of axes

		basically matplotlib.pyplot.subplots_adjust
		'''
		p = self.gui.maven.prefs
		self.figure.subplots_adjust(left=p['plot.subplots_left'],right=p['plot.subplots_right'],top=p['plot.subplots_top'],bottom=p['plot.subplots_bottom'],hspace=p['plot.subplots_hspace'],wspace=p['plot.subplots_wspace'])		
		if self.plot_mode == 'ND Raw':
			[aa.set_visible(False) for aa in self.ax[0]]
			for i in range(2):
				pos0 = np.array(self.ax[0,i].get_position().get_points())
				pos1 = np.array(self.ax[1,i].get_position().get_points())
				x0 = pos1[0,0]
				y0 = pos1[0,1]
				width = pos1[1,0] - pos1[0,0]
				height = pos0[1,1] - pos1[0,1]
				bbox = [x0,y0,width,height]
				self.ax[1,i].set_position(bbox)


	def update_blits(self):
		''' Updates blitted regions of plots

		Hides all lines, blits everything remaining into `self.blit_bgs`, puts all lines back. Works on all axes.

		'''
		[[[l.set_visible(False) for l in aaa.lines] for aaa in aa] for aa in self.ax]
		self.figure.canvas.draw()
		self.figure.canvas.flush_events()
		self.blit_bgs = [[self.figure.canvas.copy_from_bbox(aaa.bbox) for aaa in aa] for aa in self.ax]
		[[[l.set_visible(True) for l in aaa.lines] for aaa in aa] for aa in self.ax]
		self.figure.canvas.draw()
		self.figure.canvas.flush_events()

	def restore_blits(self):
		''' Restore blitted regions of the plot

		Redraws blitted regions of plot stored in `self.blit_bgs` after running `self.update_blits()`. If no blit_bgs exist, it will run `self.update_blits()` first to avoid crashing. Call this after updating the data of lines in a plot for fast plotting.

		'''
		try:
			self.blit_bgs
		except:
			self.update_blits()

		[[self.figure.canvas.restore_region(bbb) for bbb in bb] for bb in self.blit_bgs]
		[[[aaa.draw_artist(l) for l in aaa.lines] for aaa in aa] for aa in self.ax]
		[[self.figure.canvas.blit(aaa.bbox) for aaa in aa] for aa in self.ax]
		self.figure.canvas.flush_events()
		self.figure.canvas.update()

	def get_axis_inds(self):
		int_ind = 1 if self.plot_mode == 'ND Raw' else 0
		rel_ind = int(not int_ind)
		model_ind = 1 #if self.plot_mode == 'ND' else 1
		return int_ind,rel_ind,model_ind

	def set_axis_limits(self):
		''' set xlim and ylim of self.ax'''
		int_ind,rel_ind,model_ind = self.get_axis_inds()

		if self.gui.maven.prefs['plot.time_min'] < self.gui.maven.data.ntime*self.gui.maven.prefs['plot.time_dt']:
			self.ax[rel_ind,0].set_xlim(self.gui.maven.prefs['plot.time_min'],self.gui.maven.data.ntime*self.gui.maven.prefs['plot.time_dt'])
		else:
			xt = self.ax[rel_ind,0].get_xticks()
			self.ax[rel_ind,0].set_xlim(xt[0],xt[-1])
			# self.ax[rel_ind,0].set_xlim(self.gui.maven.prefs['plot.time_min'],self.gui.maven.prefs['plot.time_min']+1)
		self.ax[int_ind,0].set_ylim(self.gui.maven.prefs['plot.intensity_min'],self.gui.maven.prefs['plot.intensity_max'])
		self.ax[int_ind,1].set_xlim(-0.01,1.1)#(0.01, 1.25)
		self.ax[rel_ind,0].set_ylim(self.gui.maven.prefs['plot.fret_min'],self.gui.maven.prefs['plot.fret_max'])
		self.ax[rel_ind,1].set_xlim(self.ax[int_ind,1].get_xlim())

		p = self.gui.maven.prefs
		ha = 'right' if p['plot.time_rotate'] != 0 else 'center'
		self.update_tick_decimals(self.ax[rel_ind,0],'x',p['plot.time_decimals'],p['plot.time_rotate'],ha)
		self.update_tick_decimals(self.ax[rel_ind,0],'y',p['plot.fret_decimals'],0,'right')
		self.update_tick_decimals(self.ax[int_ind,0],'y',p['plot.intensity_decimals'],0,'right')

	def set_ticks(self):
		''' fix the ticks of self.ax '''
		int_ind,rel_ind,model_ind = self.get_axis_inds()
		self.pretty_plot()

		p = self.gui.maven.prefs
		self.ax[int_ind,0].set_yticks(self.best_ticks(p['plot.intensity_min'],p['plot.intensity_max'],p['plot.intensity_nticks']))
		self.ax[rel_ind,0].set_yticks(self.best_ticks(p['plot.fret_min'],p['plot.fret_max'],p['plot.fret_nticks']))
		ntime = self.gui.maven.data.ntime
		if ntime == 0:
			ntime = 1000
		self.ax[rel_ind,0].set_xticks(self.best_ticks(p['plot.time_min'],ntime*p['plot.time_dt'],p['plot.time_nticks']))

		plt.setp(self.ax[0,0].get_xticklabels(), visible=False)
		for aa in [self.ax[int_ind,1],self.ax[rel_ind,1]]:
			plt.setp(aa.get_yticklabels(),visible=False)
			plt.setp(aa.get_xticklabels(),visible=False)
			aa.tick_params(axis='y', which='both', left=False, right=False)
			aa.tick_params(axis='x', which='both', top=False, bottom=False)
		
		self.ax[0,0].tick_params(axis='x', which='both',length=0)

	## Add axis labels to plots
	def set_axis_labels(self):
		''' add the correct axis labels to self.ax'''

		int_ind,rel_ind,model_ind = self.get_axis_inds()

		p = self.gui.maven.prefs
		fs = p['plot.label_fontsize']
		font = {
			'family': p['plot.font'],
			'size': fs,
			'va':'top'
		}

		self.ax[int_ind,0].set_ylabel(p['plot.ylabel_text1'],fontdict=font)
		self.ax[rel_ind,0].set_ylabel(p['plot.ylabel_text2'],fontdict=font)
		self.ax[rel_ind,0].set_xlabel(p['plot.xlabel_text1'],fontdict=font)
		self.ax[rel_ind,1].set_xlabel(p['plot.xlabel_text2'],fontdict=font)

		self.ax[int_ind,0].yaxis.set_label_coords(p['plot.ylabel_offset'], 0.5)
		self.ax[rel_ind,0].yaxis.set_label_coords(p['plot.ylabel_offset'], 0.5)
		self.ax[rel_ind,0].xaxis.set_label_coords(0.5, p['plot.xlabel_offset'])
		self.ax[rel_ind,1].xaxis.set_label_coords(0.5, p['plot.xlabel_offset'])

	def set_linestyle(self,l,color,alpha,linewidth):
		'''helper for self.set_linestyles'''
		l.set_color(color)
		l.set_alpha(alpha)
		l.set_linewidth(linewidth)

	def set_linestyles(self):
		'''Make the lines the right style'''
		int_ind,rel_ind,model_ind = self.get_axis_inds()

		p = self.gui.maven.prefs
		lw = p['plot.line_linewidth']
		hw = p['draw_hist_linewidth']

		la = p['plot.line_alpha']
		lapb = p['plot.line_alpha_pb']
		alphas = [lapb,la,lapb]

		## Intensities
		for i in range(self.gui.maven.data.ncolors):
			color = p['plot.channel_colors'][i]
			for j,alpha in zip(list(range(3)),alphas):
				self.set_linestyle(self.ax[int_ind,0].lines[3*i+j], color, alpha, lw)
			self.set_linestyle(self.ax[int_ind,1].lines[i], color, p['plot.line_alpha'], hw)

		## Rel. Intensities
		for i in range(self.gui.maven.data.ncolors):
			color = self.gui.maven.prefs['plot.relative_colors'][i]
			for j,alpha in zip(list(range(3)),alphas):
				self.set_linestyle(self.ax[rel_ind,0].lines[3*i+j], color, alpha, lw)
			self.set_linestyle(self.ax[rel_ind,1].lines[i], color, p['plot.line_alpha'], hw)

		color = p['plot.idealized_color']
		alpha = p['plot.idealized_alpha']
		lw = p['plot.idealized_linewidth']
		if len(self.ax[rel_ind,0].lines) > 3*self.gui.maven.data.ncolors:
			for i in range(self.gui.maven.data.ncolors):
				self.set_linestyle(self.ax[rel_ind,0].lines[-(1+i)], color, alpha, lw)
		if len(self.ax[int_ind,0].lines) > 3*self.gui.maven.data.ncolors:
			for i in range(self.gui.maven.data.ncolors):
				self.set_linestyle(self.ax[int_ind,0].lines[-(1+i)], color, alpha, lw)

	def update_data(self,index):
		''' Replot only the lines on the plot for one trace

		Plots one trace from self.gui.maven.data.corrected. This strategy relies on the rest of the plot having been blitted using `self.update_blits`. After finished plotting, those elements are restored with `self.restore_blits`.

		Parameters
		----------
		index : int
			The index of the trace in self.gui.maven.data.corrected

		'''
		if self.gui.maven.data.nmol == 0:
			self.clear_data()
		elif self.gui.maven.data is None:
			self.clear_data()
		elif (index < self.gui.maven.data.nmol and index >= 0):
			self.flag_drawing = True
			if self.gui.maven.data.ncolor != len(self.ax[0,0].lines)//3:
				self.initialize_plots()

			t,intensities,rel,pretime,pbtime = self.calc_trajectory(index)
			self.draw_traj(t,intensities,rel,pretime,pbtime)

			if not self.gui.maven.modeler.model is None:
				idealized = self.calc_model_traj(index)
				if idealized is None:
					self.draw_no_model()
				else:
					self.draw_model(t,idealized,pretime,pbtime)

			intensity_hists,fret_hists = self.calc_histograms(intensities,rel,pretime,pbtime)
			for i in range(len(intensity_hists)):
				self.draw_hist(i,*intensity_hists[i])
			for i in range(len(fret_hists)):
				self.draw_fret_hist(i,*fret_hists[i])

			if self.plot_mode == 'ND Raw':
				[aa.set_visible(False) for aa in self.ax[0]]

			self.restore_blits()
			self.flag_drawing = False

	def clear_data(self,index=0):
		self.flag_drawing = True
		for l in self.ax[0,0].lines:
			l.set_ydata(l.get_ydata()*0.)
		for l in self.ax[1,0].lines:
			l.set_ydata(l.get_ydata()*0.)
		for aa in [self.ax[0,1],self.ax[1,1]]:
			for l in aa.lines:
				l.set_xdata(l.get_xdata()*0.)
		self.restore_blits()
		self.flag_drawing = False

	## Plot current trajectory
	def update_plots(self,index):
		''' Fully redraw entire figure for one trace

		Plots one trace from self.gui.maven.data.corrected. This will be slow because it plots everything on the plot, and then blits everything not the lines for later `self.update_data` calls

		Parameters
		----------
		index : int
			The index of the trace in self.gui.maven.data.corrected

		'''
		## stop if we're already drawing
		if self.flag_drawing:
			return
		# if self.gui.maven.data is None:
		# 	return
		# if self.gui.maven.data.nmol == 0:
		# 	return
		if self.gui.maven.data.ncolor != len(self.ax[0,0].lines)//3:
			self.initialize_plots()
		## fix fonts if not good font entered
		try: floc = matplotlib.font_manager.findfont(self.gui.maven.prefs['plot.font']) ## MPL 3.3
		except: floc = matplotlib.font_manager.find_font(self.gui.maven.prefs['plot.font']) ## MPL<3.3
		fontname = matplotlib.font_manager.get_font(floc).family_name
		if fontname != self.gui.maven.prefs['plot.font']:
			self.gui.maven.prefs['plot.font'] = fontname

		## update the trace
		self.update_data(index)

		## we're drawing
		self.flag_drawing = True

		## upate the figure shape
		self.figure.patch.set_facecolor("%s"%(self.gui.maven.prefs['plot.bg_color']))
		self.setStyleSheet("background-color:%s;"%(self.gui.maven.prefs['plot.bg_color']))
		[[aaa.patch.set_color(self.gui.maven.prefs['plot.bg_color']) for aaa in aa] for aa in self.ax]
		self.figure.set_figwidth(self.gui.maven.prefs['plot.fig_width'])
		self.figure.set_figheight(self.gui.maven.prefs['plot.fig_height'])
		# self.gui.updateGeometry()
		self.updateGeometry()

		# ## probably should remove this. turns off histograms
		# if type(self.gui.maven.prefs['draw_hist_show']) is bool:
		# 	for i in range(2):
		# 		self.ax[i][1].set_visible(self.gui.maven.prefs['draw_hist_show'])

		## draws the non-trace stuff
		self.set_ticks()
		self.update_axis_geometry()
		self.set_axis_limits()
		self.set_axis_labels()
		self.set_linestyles()

		## draw and update blits
		self.draw()
		# self.update_blits() ## apparently that's not needed here

		## we're done drawing
		self.flag_drawing = False

	## Handler for mouse clicks in main plots
	def mouse_handler(self,event):
		''' Handles mouse clicks

		Called when user clicks in widget (mpl_connect)

		Parameters
		----------
		event : QEvent
			click event

		Notes
		-----
		If not toolbar buttons are active, then
			* Right click - sets post point (gui.maven.data.post_list)
			* Left click - sets pre point (gui.maven.data.pre_list)
			* Middle click - calculate the photobleaching point `tmaven.photobleaching.photobleaching.get_point_pbtime`

		'''
		if self.gui.flag_locked:
			return
		
		if (event.inaxes == self.ax[0,0]) or (event.inaxes == self.ax[1,0]):
			try:
				if event.button == 3 and self.toolbar.mode == "": ## Right click - set photobleaching point
					self.gui.maven.data.post_list[self.gui.index] = int(np.round(event.xdata/self.gui.maven.prefs['plot.time_dt']))

				if event.button == 1 and self.toolbar.mode == "": ## Left click - set pre-truncation point
					self.gui.maven.data.pre_list[self.gui.index] = int(np.round(event.xdata/self.gui.maven.prefs['plot.time_dt']))

				## Middle click - reset pre and post points to calculated values
				if event.button == 2 and self.toolbar.mode == "":
					if self.gui.maven.data.ncolors == 2:
						from ...photobleaching.photobleaching import get_point_pbtime
						self.gui.maven.data.pre_list[self.gui.index] = 0
						if self.gui.maven.prefs['photobleaching_flag'] is True:
							qq = self.gui.maven.data.corrected[self.gui.index].sum(-1)
						else:
							qq = self.gui.maven.data.corrected[self.gui.index,:,-1]
						self.gui.maven.data.post_list[self.gui.index] = get_point_pbtime(qq,1.,1.,1.,1000.) + 1
				self.update_plots(self.gui.index)
				try:
					self.gui.molecules_viewer.viewer.model.layoutChanged.emit()
				except:
					pass
			except:
				pass
		else:
			self.gui.setFocus()

#########################################################################################
#########################################################################################


	def draw_hist(self,i,hx,hy):
		'''Draw intensity histograms'''
		int_ind,rel_ind,model_ind = self.get_axis_inds()
		self.ax[int_ind,1].lines[i].set_data(hy,hx)

	def draw_fret_hist(self,i,hx,hy):
		'''Draw fret histograms'''
		int_ind,rel_ind,model_ind = self.get_axis_inds()
		self.ax[rel_ind,1].lines[i].set_data(hy,hx)

	def draw_no_model(self):
		'''Hide the model lines'''
		int_ind,rel_ind,model_ind = self.get_axis_inds()
		# for i in range(self.gui.maven.data.ncolors):
		# 	self.ax[rel_ind,0].lines[-i-1].set_visible(False)

	def draw_model(self,t,idealized,pretime,pbtime):
		'''Draw idealized fret lines and show them'''
		int_ind,rel_ind,model_ind = self.get_axis_inds()
		# if remove_idealized:
		# 	d = self.ax[1,0].lines[-3].get_data()
		# 	self.ax[1,0].lines[-3].set_data(t[pretime:pbtime],d[1]-state_means[vitpath])
		for i in range(self.gui.maven.data.ncolors):
			self.ax[model_ind,0].lines[-i-1].set_data(t[pretime:pbtime],idealized[pretime:pbtime])
			self.ax[model_ind,0].lines[-i-1].set_visible(True)

	def draw_traj(self,t,intensities,rel,pretime,pbtime):
		''' draw intensity lines and rel. intensity lines '''
		int_ind,rel_ind,model_ind = self.get_axis_inds()

		for i in range(self.gui.maven.data.ncolors):
			self.ax[int_ind,0].lines[3*i+0].set_data(t[:pretime],intensities[:pretime,i])
			self.ax[int_ind,0].lines[3*i+1].set_data(t[pretime:pbtime],intensities[pretime:pbtime,i])
			self.ax[int_ind,0].lines[3*i+2].set_data(t[pbtime:],intensities[pbtime:,i])

		for i in range(self.gui.maven.data.ncolors):
			self.ax[rel_ind,0].lines[3*i+0].set_data(t[:pretime],rel[:pretime,i])
			self.ax[rel_ind,0].lines[3*i+1].set_data(t[pretime:pbtime],rel[pretime:pbtime,i])
			if not self.gui.maven.prefs['plot.fret_pbzero']:
				self.ax[rel_ind,0].lines[3*i+2].set_data(t[pbtime:],rel[pbtime:,i])
			else:
				self.ax[rel_ind,0].lines[3*i+2].set_data(t[pbtime:],np.zeros_like(rel[pbtime:,i]))

	def calc_trajectory(self,index):
		''' get data for current trajectory '''
		if self.plot_mode == "Normalized":
			intensities = self.gui.maven.data.raw[index].copy()
		else:
			intensities = self.gui.maven.data.corrected[index].copy()
		t = np.arange(intensities.shape[0])*self.gui.maven.prefs['plot.time_dt'] + self.gui.maven.prefs['plot.time_offset']

		pbtime = int(self.gui.maven.data.post_list[index])
		pretime = int(self.gui.maven.data.pre_list[index])

		if self.plot_mode == "Normalized":
			rel = self.gui.maven.data.corrected[index].copy()
		else:
			rel = self.gui.maven.calc_relative(index).copy()
		return t,intensities,rel,pretime,pbtime

	def calc_histograms(self,intensities,rel,pretime,pbtime):
		''' get data for histogram of current trajectory '''
		intensity_hists = []
		fret_hists = []
		hymaxes = []
		yminmax = np.array((self.gui.maven.prefs['plot.intensity_min'],self.gui.maven.prefs['plot.intensity_max']))
		for i in range(self.gui.maven.data.ncolors):
			if pretime < pbtime:
				yy = intensities[pretime:pbtime,i]
				hy,hx = np.histogram(yy,range=yminmax,bins=int(np.sqrt(pbtime-pretime)))
			else:
				hy = np.zeros(100)
				hx = np.linspace(yminmax[0],yminmax[1],101)
			hy = np.append(np.append(0.,hy),0.)
			hx = np.append(np.append(hx[0],.5*(hx[1:]+hx[:-1])),hx[-1])
			hymaxes.append(hy.max())
			intensity_hists.append([hx,hy])

		for i in range(self.gui.maven.data.ncolors):
			if pretime < pbtime:
				hy,hx = np.histogram(rel[pretime:pbtime,i],range=(self.gui.maven.prefs['plot.fret_min'],self.gui.maven.prefs['plot.fret_max']),bins=int(np.sqrt(pbtime-pretime)))
			else:
				hy = np.zeros(100)
				hx = np.linspace(self.gui.maven.prefs['plot.intensity_min'],self.gui.maven.prefs['plot.intensity_max'],101)
			hy = np.append(np.append(0.,hy),0.)
			hx = np.append(np.append(hx[0],.5*(hx[1:]+hx[:-1])),hx[-1])
			hymaxes.append(hy.max())
			fret_hists.append([hx,hy])

		hymax = np.max(hymaxes)
		for i in range(len(intensity_hists)):
			intensity_hists[i][1] /= hymax
		for i in range(len(fret_hists)):
			fret_hists[i][1] /= hymax

		return intensity_hists,fret_hists

	def calc_model_traj(self,index):
		''' get data for idealized current trajectory '''
		try: ## if a model doesn't have a corresponding idealized, then this should crash and fxn will return None,None
			idealized = self.gui.maven.modeler.model.idealized[index]
		except:
			idealized = None
		return idealized
