import logging
logger = logging.getLogger('tmaven')
logger.setLevel(logging.DEBUG)
# log_fmt = ("%(name)s.%(funcName)s(%(lineno)d): %(message)s")
log_fmt = ("[(%(lineno)4s:%(filename)s %(funcName)s)  %(message)s]")
formatter = logging.Formatter(log_fmt)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("h5py").setLevel(logging.WARNING)

import os
import sys
import numpy as np

class maven_class(object):
	def __init__(self,log_stdout=False):
		super().__init__()
		self.setup_logging(log_stdout)
		self.initialize_objects()

	def setup_logging(self,log_stdout):
		from io import StringIO

		self.log_stream = StringIO()
		str_handler = logging.StreamHandler(self.log_stream)
		str_handler.setFormatter(formatter)
		str_handler.setLevel(logging.DEBUG)
		logger.addHandler(str_handler)

		if log_stdout:
			stdout_handler = logging.StreamHandler()
			stdout_handler.setFormatter(formatter)
			stdout_handler.setLevel(logging.DEBUG)
			logger.addHandler(stdout_handler)

		# logging.getLogger("tmaven.controllers.prefs").setLevel(logging.WARNING)

		from . import __version__
		logger.info("Starting tmaven {}".format(__version__))

		import platform
		logger.info(platform.uname())
		logger.info("Platform: {}".format(platform.platform()))
		logger.info("\nPython path: \n   {}".format('\n   '.join(sys.path)))

		import pkg_resources
		these_pkgs = ['matplotlib','scipy','numpy','h5py','numba','PyQt5']
		packages = '\n   '.join(['{} {}'.format(d.project_name, d.version) for d in pkg_resources.working_set if d.project_name in these_pkgs])
		logger.info('\nLibrary Versions:\n   '+packages)

	def get_log(self):
		return self.log_stream.getvalue()

	def initialize_objects(self):
		logger.info("Setting up maven")

		from .controllers import prefs_object, default_prefs
		self.prefs = prefs_object()
		self.prefs.add_dictionary(default_prefs)
		# self.prefs.emit_changed = lambda : print('>>>prefs.emit_changed')
		# self.prefs.emit_failed = lambda : print('>>>prefs.emit_failed')

		from .controllers import controller_io, controller_data
		self.io = controller_io(self)
		self.smd = self.io.blank_smd()
		self.data = controller_data(self)

		from .controllers import controller_corrections, controller_cull,  controller_modeler, controller_scripts, controller_selection
		self.corrections = controller_corrections(self)
		self.cull = controller_cull(self)
		self.modeler = controller_modeler(self)
		self.scripts = controller_scripts(self)
		self.selection = controller_selection(self)

		from .controllers import controller_trace_filter, controller_photobleaching
		self.trace_filter = controller_trace_filter(self)
		self.photobleaching = controller_photobleaching(self)

		from .controllers import controller_analysisplots
		self.plots = controller_analysisplots(self)

	def calc_relative(self,index=None):
		if not index is None:
			return self.data.corrected[index]/(self.data.corrected[index].sum(-1) + 1e-300)[:,None]
		return self.data.corrected / (self.data.corrected.sum(2) + 1e-300)[:,:,None]

	def emit_data_update(self):
		# print('>>>maven.data_update')
		pass
