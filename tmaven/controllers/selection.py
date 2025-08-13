import numpy as np
import logging
logger = logging.getLogger(__name__)

class controller_selection(object):
	''' Handles selection

	.selected - np.ndarray of indices of selection
	'''

	def __init__(self,maven):
		self.maven = maven

	def get_class_counts(self):
		''' Display number of molecules in each class
		'''
		ns = np.unique(self.maven.data.classes)
		ons = np.array([np.sum(self.maven.data.flag_ons*(self.maven.data.classes == i)) for i in ns])
		counts = np.array([np.sum((self.maven.data.classes == i)) for i in ns])
		s = 'Class:\n'
		s += '\n'.join(['{}: {}/{}'.format(ns[i],counts[i],self.maven.data.nmol) for i in range(ns.size)])
		return s

	def get_toggled_mask(self):
		return np.flatnonzero(self.maven.data.flag_ons)

	def toggle_selection(self,selected):
		return np.setdiff1d(np.arange(self.maven.data.nmol),selected)

	def select_all(self):
		return np.arange(self.maven.data.nmol)
	def select_none(self):
		return np.array([])
	def select_class(self,i):
		return np.unique(np.flatnonzero(self.maven.data.classes == i))

	def on_selection(self,selection):
		self.maven.data.flag_ons[selection] = True
	def off_selection(self,selection):
		self.maven.data.flag_ons[selection] = False

	def on_all(self):
		self.maven.data.flag_ons[:] = True
	def off_all(self):
		self.maven.data.flag_ons[:] = False

	def set_class_from_selection(self,selection,i):
		self.maven.data.classes[self.selected] = i

	def calc_fret_cross_corr(self,d=None): ## of gradient
		''' Calculate cross correlation of gradient of 1D series

		Cross correlation of position 0 and position 1 in data dimension (e.g. color 1 and color 2). Uses calculates the gradient of each series, and then cross correlation of those gradients to only see if "jumps" are occuring at the same time

		Parameters
		----------
		d : np.ndarray (ntime,2)
			If `None`, cross-correlation is calculated for *every* trace in maven.data.corrected. Otherwise, only calculate for the data in `d`.

		Returns
		-------
		cc : np.ndarray (nmol) or float
			cross-correlation. Length depends upon `d`

		'''
		if d is None:
			x = self.maven.data.corrected[:,:,0] #- self.corrected[:,0].mean(1)[:,None]
			y = self.maven.data.corrected[:,:,1] #- self.corrected[:,1].mean(1)[:,None]
			x = np.gradient(x,axis=1) # over time
			y = np.gradient(y,axis=1)

			a = np.fft.fft(x,axis=1)
			b = np.conjugate(np.fft.fft(y,axis=1))
			cc = np.fft.ifft((a*b),axis=1)
			cc = cc[:,0].real
		else:
			if d.shape[0] == 0:
				return 0.
			x = d[:,0]
			y = d[:,1]
			x = np.gradient(x) # over time
			y = np.gradient(y)

			a = np.fft.fft(x)
			b = np.conjugate(np.fft.fft(y))
			cc = np.fft.ifft((a*b))
			cc = cc[0].real
		return cc


	def order_fret_cross_corr(self):
		''' Order traces in `maven.data` from highest cross-correlation of color dimensions to lowest
		'''
		if self.maven.data.ncolors == 2:
			cc_list = np.array([self.calc_fret_cross_corr(self.maven.data.corrected[i, self.maven.data.pre_list[i]:self.maven.data.post_list[i]]) for i in range(self.maven.data.nmol)])
			neworder = np.argsort(cc_list)
			if not neworder is None:
				self.maven.data.order(neworder)
				self.maven.emit_data_update()
				logger.info('Data ordered by cross-correlation')
			else:
				logger.error('cross-correlation calculation failed')
		else:
			logger.error('Order by cross-correlation requires two color data')

	def order_original(self):
		neworder = np.argsort(self.maven.data.data_index)
		self.maven.data.order(neworder)
		self.maven.emit_data_update()
		logger.info('Data reverted to original sort order')

	def order_classes(self):
		neworder = np.argsort(self.maven.data.classes)
		self.maven.data.order(neworder)
		self.maven.emit_data_update()
		logger.info('Data ordered by classes')
