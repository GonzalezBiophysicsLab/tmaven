import numpy as np
import logging
logger = logging.getLogger(__name__)

from .corrections.corrections import controller_corrections

default_prefs = {
	'normalize.single_axis': False,
	'normalize.axis': -1
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

class controller_normalizations(controller_corrections):
	''' Handles normalising raw data. Changes corrected data.
		Inherited from controller corrections since it handles corrected data

	Parameters
	----------

	Notes
	-----
	smd.raw is considered immutable data, while data.corrected is what is modified by the functions in this class and used for plotting/analyses elsewhere
	'''
	def __init__(self,maven):
		super().__init__(maven)
		self.maven.prefs.add_dictionary(default_prefs)
	
	def normalize_minmax_ind(self):
		self.reset()
		norm = np.zeros_like(self.maven.data.corrected)
		pre = self.maven.data.pre_list
		post = self.maven.data.post_list
		
		for j in range(self.maven.data.ncolors):
			for i in range(self.maven.data.nmol):
				maxim = self.maven.data.corrected[i,pre[i]:post[i],j].max()
				minim = self.maven.data.corrected[i,pre[i]:post[i],j].min()
					
				norm[i, :, j] = (self.maven.data.corrected[i, :, j] - minim)/(maxim - minim + 1e-300)
		
		self.maven.data.corrected = norm
		self.correction_update()

	def normalize_minmax_ckfilt_ind(self):
		self.reset()
		self.normalize_minmax_ind()
		self.filter_chungkennedy()
		self.normalize_minmax_ind()
		
	def normalize_minmax_glob(self):
		self.reset()
		norm = np.zeros_like(self.maven.data.corrected)
		pre = self.maven.data.pre_list
		post = self.maven.data.post_list
		maxim_arr = []
		minim_arr = []
		for j in range(self.maven.data.ncolors):
			for i in range(self.maven.data.nmol):
				maxim_arr.append(self.maven.data.corrected[i,pre[i]:post[i],j].max())
				minim_arr.append(self.maven.data.corrected[i,pre[i]:post[i],j].min())

			maxim = np.max(maxim_arr)
			minim = np.min(minim_arr)

			norm[:, :, j] = (self.maven.data.corrected[:, :, j] - minim)/(maxim - minim + 1e-300)
		
		self.maven.data.corrected = norm
		self.correction_update()
	