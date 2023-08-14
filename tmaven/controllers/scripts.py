import os
import logging
logger = logging.getLogger(__name__)

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

class controller_scripts(object):
	''' run scripts
	scripts have access to variable `maven` and can do anything programatically to it

	'''
	def __init__(self,maven):
		super().__init__()
		self.maven = maven
		logger.info('Script Runner Initialized')

	@tryexcept
	def run(self,fname,gui=None):
		''' run a plugin file
		'''

		try:
			plugin = open(fname,'r').read()
		except Exception as e:
			logging.error('Script: could not open script file %s\n%s'%(fname,str(e)))
			return False

		try:
			logger.info('Script: running %s'%(fname))
			code = compile(plugin,fname,'exec')
			context = {'maven':self.maven,'gui':gui} ## this will hold all of the elements in the plugin file... eg functions, global variables.
			exec(code,context)
		except Exception as e:
			logging.error(str(e))
			return False
		return True
