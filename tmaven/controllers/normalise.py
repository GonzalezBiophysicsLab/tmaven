import numpy as np
import logging
logger = logging.getLogger(__name__)

from .corrections.corrections import controller_corrections

default_prefs = {
	'normalize.axis': 0,
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
		norm = np.zeros_like(self.maven.data.corrected)
		for i in range(self.maven.data.nmol):
			for j in range(self.maven.data.ncolors):
					maxim = self.maven.data.corrected[i,:,j].max()
					minim = self.maven.data.corrected[i,:,j].min()
					
					norm[i, :, j] = (self.maven.data.corrected[i, :, j] - minim)/(maxim - minim + 1e-300)
		
		self.maven.data.corrected = norm
		self.correction_update()