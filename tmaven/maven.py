### borrows heavily from Mu-editor. Lots of credit to them

import os
import sys
import functools
import numpy as np
import logging
logger = logging.getLogger(__name__)

class maven_class(object):
	def __init__(self):
		super().__init__()
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

	def calc_fret(self,index=None):
		if not index is None:
			return self.data.corrected[index]/(self.data.corrected[index].sum(-1) + 1e-300)[:,None]
		return self.data.corrected / (self.data.corrected.sum(2) + 1e-300)[:,:,None]

	def emit_data_update(self):
		# print('>>>maven.data_update')
		pass
