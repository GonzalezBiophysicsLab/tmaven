class controller_analysisplots(object):
	def __init__(self,maven):
		self.maven = maven

		from .data_hist1d import controller_data_hist1d
		from .data_hist2d import controller_data_hist2d
		from .data_tdp import controller_data_tdp
		from .model_vbstates import controller_model_vbstates
		from .survival_dwell import controller_survival_dwell
		from .tm_hist import controller_tm_hist

		self.data_hist1d = controller_data_hist1d(self.maven)
		self.data_hist2d = controller_data_hist2d(self.maven)
		self.data_tdp = controller_data_tdp(self.maven)
		self.model_vbstates = controller_model_vbstates(self.maven)
		self.survival_dwell = controller_survival_dwell(self.maven)
		self.tm_hist = controller_tm_hist(self.maven)
