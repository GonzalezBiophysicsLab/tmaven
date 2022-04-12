class controller_analysisplots(object):
	def __init__(self,maven):
		self.maven = maven

		from .fret_hist1d import controller_fret_hist1d
		from .fret_hist2d import controller_fret_hist2d
		from .fret_tdp import controller_fret_tdp
		from .model_vbstates import controller_model_vbstates
		from .survival_dwell import controller_survival_dwell

		self.fret_hist1d = controller_fret_hist1d(self.maven)
		self.fret_hist2d = controller_fret_hist2d(self.maven)
		self.fret_tdp = controller_fret_tdp(self.maven)
		self.model_vbstates = controller_model_vbstates(self.maven)
		self.survival_dwell = controller_survival_dwell(self.maven)
