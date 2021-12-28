import numpy as np

def tryexcept(function):
	def wrapper(*args,**kw_args):
		try:
			return function(*args,**kw_args)
		except Exception as e:
			try:
				self.gui.log.emit(e)
			except:
				print('Error:',function)
				print(e)
		return None
	wrapper.__doc__ = function.__doc__ ## IMPORTANT FOR SPHINX!
	return wrapper

class analysisplots_container(object):
	''' Makes standard plots for smFRET data analysis

	Parameters
	----------
	maven : plotter_gui
		instance of the `fret_plot` GUI

	Attributes
	----------
	menu_plots : QMenu
		menu for ordering trace QActions
			* 1D Histogram
			* 2D Histogram
			* Transition Density Plot
			* Autocorrelation Function
			* VB States
			* Sources

	Notes
	-----
	Each plot launches as a `./popout_plot.py` `popout_plot_container`

	'''
	def __init__(self,gui):
		super().__init__()
		self.gui = gui

	def init_menu_plots(self):
		from PyQt5.QtWidgets import  QMenu
		menu_plots = QMenu('Plots',self.gui)
		from ..interface.stylesheet import ss_qmenu
		menu_plots.setStyleSheet(ss_qmenu)

		menu_plots.addAction('1D Histogram',self.launch_hist1d)
		menu_plots.addAction('2D Histogram',self.launch_hist2d)
		menu_plots.addAction('Transition Density Plot',self.launch_tdp)
		menu_plots.addAction('Autocorrelation Function Plot',self.launch_acorr)
		menu_plots.addAction('VB States',self.launch_vbstates)
		menu_plots.addAction('Sources',self.launch_sources)

		return menu_plots

	def launch_hist1d(self,event=None):
		''' launches 1D histogram plot '''
		from . import hist_1d
		try:
			if not self.hist1d.isVisible():
				self.hist1d.setVisible(True)
			self.hist1d.raise_()
		except:
			from .popout_plot import popout_plot_container
			self.hist1d = popout_plot_container(1,1,self.gui)
			self.hist1d.setWindowTitle('1D Histogram')
			self.hist1d.prefs.add_dictionary(hist_1d.default_prefs)
			self.hist1d.replot_fxn = hist_1d.plot
			hist_1d.setup(self.hist1d)
			hist_1d.plot(self.hist1d)
			self.hist1d.plot.resize_figure()
			self.hist1d.show()

	def launch_hist2d(self,event=None):
		''' launches 2D histogram plot '''
		from . import hist_2d
		try:
			if not self.hist2d.isVisible():
				self.hist2d.setVisible(True)
			self.hist2d.raise_()
		except:
			from .popout_plot import popout_plot_container
			self.hist2d = popout_plot_container(1,1,self.gui)
			self.hist2d.setWindowTitle('2D Histogram')
			self.hist2d.prefs.add_dictionary(hist_2d.default_prefs)
			self.hist2d.replot_fxn = hist_2d.plot
			hist_2d.plot(self.hist2d)
			self.hist2d.plot.resize_figure()
			self.hist2d.show()


	def launch_tdp(self,event=None):
		''' launches transition density plot '''
		from . import tdp
		try:
			if not self.tdp.isVisible():
				self.tdp.setVisible(True)
			self.tdp.raise_()
		except:
			from .popout_plot import popout_plot_container
			self.tdp = popout_plot_container(1,1,self.gui)
			self.tdp.setWindowTitle('Transition Density Plot')
			self.tdp.prefs.add_dictionary(tdp.default_prefs)
			self.tdp.replot_fxn = tdp.plot
			tdp.plot(self.tdp)
			self.tdp.plot.resize_figure()
			self.tdp.show()

	def launch_acorr(self,event=None):
		''' launches autocorrelation plot '''
		from . import autocorr
		try:
			if not self.autocorr.isVisible():
				self.autocorr.setVisible(True)
			self.autocorr.raise_()
		except:
			from .popout_plot import popout_plot_container
			self.autocorr = popout_plot_container(1,1,self.gui)
			self.autocorr.setWindowTitle('Autocorrelation')
			self.autocorr.prefs.add_dictionary(autocorr.default_prefs)
			self.autocorr.replot_fxn = autocorr.plot
			autocorr.setup(self.autocorr)

			autocorr.plot(self.autocorr)
			self.autocorr.plot.resize_figure()
			self.autocorr.show()

	def launch_vbstates(self,event=None):
		''' launches number of vbFRET states plot '''
		from . import vb_states
		try:
			if not self.vbstates.isVisible():
				self.vbstates.setVisible(True)
			self.vbstates.raise_()
		except:
			from .popout_plot import popout_plot_container
			self.vbstates = popout_plot_container(1,1,self.gui)
			self.vbstates.setWindowTitle('VB NStates')
			self.vbstates.prefs.add_dictionary(vb_states.default_prefs)
			self.vbstates.replot_fxn = vb_states.plot
			vb_states.plot(self.vbstates)
			self.vbstates.plot.resize_figure()
			self.vbstates.show()

	def launch_sources(self,event=None):
		''' launches plots by source plot '''
		from . import sources
		try:
			if not self.sources.isVisible():
				self.sources.setVisible(True)
			self.sources.raise_()
		except:
			from .popout_plot import popout_plot_container
			self.sources = popout_plot_container(1,1,self.gui)
			self.sources.setWindowTitle('Sources')
			self.sources.prefs.add_dictionary(sources.default_prefs)
			self.sources.replot_fxn = sources.plot
			sources.setup(self.sources)
			sources.plot(self.sources)
			self.sources.plot.resize_figure()
			self.sources.show()

	def get_plot_fret(self):
		''' Get the fret data for plotting

		Has to be in a toggled class. Removes photobleaching pre and post times

		Returns
		-------
		fpb : np.ndarray
		 	fret (nmol toggled, ntime, ncolors)
		'''
		fpb = self.gui.maven.calc_fret()
		for i in range(self.gui.maven.data.nmol): ## photobleach molecules
			fpb[i,:self.gui.maven.data.pre_list[i]] = np.nan
			fpb[i,self.gui.maven.data.post_list[i]:] = np.nan
		mask = self.gui.maven.selection.get_toggled_mask() ## only chosen classes
		fpb = fpb[mask]
		return fpb

	def get_idealized_data(self,signal=False):
		''' Get toggled idealized data for plotting

		Get's idealized from traces in classes toggled on.

		Parameters
		-----------
		signal : bool
		 	* True - return idealized in signal space
			* False - return idealized in state space

		Returns
		-------
		v : np.ndarray
			* idealized data in form (n_hmm_ran,ntime) with `np.nan` where no model
			* `None` if no `maven.modeler.model` or failure

		'''
		## get_viterbi_data
		if not self.gui.maven.modeler.model is None:
			v = np.zeros((self.gui.maven.data.nmol,self.gui.maven.data.ntime)) + np.nan
			for i in range(self.gui.maven.data.nmol):
				try:
					if self.gui.maven.modeler.model.ran.count(i) > 0:
						ii = self.gui.maven.modeler.model.ran.index(i)
						if self.gui.maven.modeler.model.type == 'vb Consensus HMM':
							vi = self.gui.maven.modeler.model.result.viterbi[ii]
							if signal:
								vi = self.gui.maven.modeler.model.result.m[vi]
							v[i,self.gui.maven.data.pre_list[i]:self.gui.maven.data.post_list[i]] = vi
						elif self.gui.maven.modeler.model.type in ['vb','ml','threshold']:
							r = self.gui.maven.modeler.model.results[ii]
							vi = r.viterbi
							if signal:
								vi = r.mu[vi]
							v[i,self.gui.maven.data.pre_list[i]:self.gui.maven.data.post_list[i]] = vi
				except:
					pass
			checked = self.gui.maven.selection.get_toggled_mask()
			v = v[checked]
			return v
		else:
			return None

	def close_all_plots(self):
		try: self.hist1d.really_close()
		except: pass
		try: self.hist2d.really_close()
		except: pass
		try: self.tdp.really_close()
		except: pass
		try: self.autocorr.really_close()
		except: pass
		try: self.vbstates.really_close()
		except: pass
		try: self.sources.really_close()
		except: pass
