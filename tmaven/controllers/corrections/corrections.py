import numpy as np
import logging
logger = logging.getLogger(__name__)

default_prefs = {
	'correction.filterwidth':2.0,
	'correction.bleedthrough':0.05,
	'correction.gamma':1.,
	'correction.backgroundframes':100,
}

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

class controller_corrections(object):
	''' Handles modifying raw data into corrected data

	Parameters
	----------

	Notes
	-----
	smd.raw is considered immutable data, while data.corrected is what is modified by the functions in this class and used for plotting/analyses elsewhere

	'''
	def __init__(self,maven):
		super().__init__()
		self.maven = maven
		self.maven.prefs.add_dictionary(default_prefs)

	def reset(self):
		''' Reset data.corrected to data.raw

		Copies data.raw to data.corrected. This resets the corrections
		'''

		self.maven.data.corrected = self.maven.smd.raw.copy()
		logger.info('Corrected reset to Raw')
		self.correction_update()

	@tryexcept
	def filter_gaussian(self):
		''' Gaussian filter data

		Uses `scipy.ndimage.gaussian_filter1d` on data.corrected (each color separately) using a sigma/width of prefs['correction.filterwidth']
		'''

		width = self.maven.prefs['correction.filterwidth']
		y = self.maven.data.corrected.copy()
		self.flag_running = True

		from scipy.ndimage import gaussian_filter1d
		for i in range(self.maven.data.nmol):
			if self.flag_running:
				# self.signaler.emit_nextmol(i,self.gui.data.nmol)
				for c in range(self.maven.data.ncolors):
					y[i,:,c] = gaussian_filter1d(self.maven.data.corrected[i,:,c],width)
		logger.info('Gaussian filter applied, sigma=%f'%(width))

		self.flag_running = False
		self.maven.data.corrected = y
		self.correction_update()

	@tryexcept
	def filter_wiener(self):
		''' Wiener filter data.corrected

		Applies `scipy.signal.wiener` to data.corrected. Uses a size of int(floor(prefs['correction.filterwidth']/2)*2+1), with a minimum value of 3
		'''

		width = self.maven.prefs['correction.filterwidth']
		y = self.maven.data.corrected.copy()
		self.flag_running = True

		from scipy.signal import wiener
		width = int(np.floor(width/2)*2+1)
		width = np.max((width,3))
		for i in range(self.maven.data.nmol):
			if self.flag_running:
				# self.signaler.emit_nextmol(i,self.gui.data.nmol)
				for c in range(self.maven.data.ncolors):
					y[i,:,c] = wiener(self.maven.data.corrected[i,:,c],mysize=width)
		logger.info('Wiener filter applied, width=%d'%(width))

		self.flag_running = False
		self.maven.data.corrected = y
		self.correction_update()

	@tryexcept
	def filter_median(self):
		'''median filter data.corrected

		Applies `scipy.ndimage.median_filter` to data.corrected using a width of prefs['correction.filterwidth'].
		'''

		width = self.maven.prefs['correction.filterwidth']
		y = self.maven.data.corrected.copy()
		self.flag_running = True

		from scipy.ndimage import median_filter
		width = int(np.max((1,width)))
		for i in range(self.maven.data.nmol):
			if self.flag_running:
				# self.gui.emit_nextmol(i,self.guid.datanmol)
				for c in range(self.maven.data.ncolors):
					y[i,:,c] = median_filter(self.maven.data.corrected[i,:,c],width)
		logger.info('Median filter applied, width=%d'%(width))

		self.flag_running = False
		self.maven.data.corrected = y
		self.correction_update()

	@tryexcept
	def filter_bessel(self):
		''' 8th-order Bessel filter data.corrected

		Applies an 8-pole Bessel filter from `scipy.signal` with frequency of 1/.prefs['correction.filterwidth'] to data.corrected
		'''
		width = self.maven.prefs['correction.filterwidth']
		y = self.maven.data.corrected.copy()
		self.flag_running = True

		from scipy.signal import bessel,filtfilt
		if width < 2: width = 2.
		b, a = bessel(8, 1./width)
		for i in range(self.maven.data.nmol):
			if self.flag_running:
				# self.signaler.emit_nextmol(i,self.gui.data.nmol)
				for c in range(self.maven.data.ncolors):
					y[i,:,c] = filtfilt(b, a, self.maven.data.corrected[i,:,c])
		logger.info('8th-order Bessel filter applied, freq=%f'%(1./width))

		self.flag_running = False
		self.maven.data.corrected = y
		self.correction_update()

	@tryexcept
	def filter_chungkennedy(self):
		''' Bayesian-ish Chung-Kennedy filter data.corrected

		Applies the Chung-Kennedy filter to data.corrected using p = prefs['correction.filterwidth']. More information on the filter in `fret_plot/corrections/ck_filter.py`

		'''
		width = self.maven.prefs['correction.filterwidth']
		y = self.maven.data.corrected.copy()
		self.flag_running = True

		from .ck_filter import ck_filter
		for i in range(self.maven.data.nmol):
			if self.flag_running:
				# self.signaler.emit_nextmol(i,self.gui.data.nmol)
				y[i] = ck_filter(self.maven.data.corrected[i],p=width)
		logger.info('Chung-Kennedy filter applied, p=%f'%(width))

		self.flag_running = False
		self.maven.data.corrected = y
		self.correction_update()

	@tryexcept
	def remove_beginning(self,nd):
		if nd > 0 and nd < self.maven.data.ntime:
			self.maven.data.raw = self.maven.data.raw[:,nd:]
			self.maven.data.corrected = self.maven.data.corrected[:,nd:]
			self.maven.data.pre_list -= nd
			self.maven.data.post_list -= nd
			self.maven.data.pre_list[self.maven.data.pre_list < 0] = 0
			self.maven.data.post_list[self.maven.data.post_list < 1] = 1
			logger.info('Removed %d datapoints from start of all traces'%(nd))
			self.correction_update()
			return True
		else:
			logger.error('Remove datapoints from beginning - bad number')
			return False


	def bleedthrough(self):
		''' Remove percentage color 1 from color 2

		Only works for ncolors = 2. Removes prefs['correction.bleedthrough']*data.corrected[:,:,0] from data.corrected[:,:,1]
		'''
		if self.maven.data.ncolors == 2:
			bleedthroughs = np.array(((0.,self.maven.prefs['correction.bleedthrough']),(0.,0.)))

			# bleedthrough = self.gui.prefs['correction.bleedthrough']
			if not bleedthroughs.shape == (self.maven.data.ncolors,self.maven.data.ncolors):
				logger.error('bleedthrough correction coefficients matrix is wrong shape')
				return

			y = self.maven.data.corrected.copy()
			for i in range(self.maven.data.ncolors):
				for j in range(self.maven.data.ncolors):
					if i != j:
						y[:,:,j] = self.maven.data.corrected[:,:,j] - bleedthroughs[i,j]*self.maven.data.corrected[:,:,i]
			self.maven.data.corrected = y
			logger.info('Bleedthrough correction %s'%(str(bleedthroughs)))
			self.correction_update()
		else:
			logger.error('bleedthrough not implemented for %d colors'%(maven.data.ncolors))
			## TODO: must figure out to get use to input multiple values of bleedthrough coefficients

	def gamma(self):
		''' Gamma correct 2 color data

		Only works for two color data. Multiplies first color of data.corrected by prefs['correction.gamma']
		'''
		if self.maven.data.ncolors == 2:
			gammas = np.array((self.maven.prefs['correction.gamma'],1.))

			if not gammas.size == self.maven.data.ncolors:
				logger.error('gamma correction coefficients array is wrong size')
				return
			for i in range(self.maven.data.ncolors):
				self.maven.data.corrected[:,:,i] *= gammas[i]
			logger.info('Gamma correction %s'%(str(gammas)))
			self.correction_update()
		else:
			logger.error('gammas not implemented for %d colors'%(self.maven.data.ncolors))

	def background_correct(self):
		'''
		Experimental for now. Tries to fix high background in red (for now)
		'''
		if self.maven.data.ncolors == 2:
			cutoff = int(self.maven.prefs['correction.backgroundframes'])

			if cutoff >= self.maven.data.ntime:
				logger.error('Background frames cutoff needs to be less than total frames')
				return
			else:
				for i in range(self.maven.data.nmol):
					diff = self.maven.data.corrected[i,:-cutoff,1].mean() - self.maven.data.corrected[i,:-cutoff,0].mean()
					self.maven.data.corrected[i,:,1] -= diff
				logger.info('Background correction: %s frames'%(str(cutoff)))
				self.correction_update()
		else:
			logger.error('Background correction not implemented for %d colors'%(self.maven.data.ncolors))

	def correction_update(self):
		self.maven.emit_data_update()
