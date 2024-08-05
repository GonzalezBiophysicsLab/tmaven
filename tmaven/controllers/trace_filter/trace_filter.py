import numpy as np
import logging
logger = logging.getLogger(__name__)


class controller_trace_filter(object):
	''' Filter traces by SBR

	Parameters
	----------
	maven

	Attributes
	----------
	models_list : list
		list of traces in each model category
			* 0: Dead (bleaches before 'min frames' value), any SBR
			* 1: Bleaching, Low SBR
			* 2: Bleaching, High SBR
			* 3: No bleaching, Low SBR
			* 4: No bleaching, High SBR
	combo_data : QComboBox
		determines what data to use
			* Donor (color 0)
			* Acceptor (color 1)
			* Donor + Acceptor (color 0 + color 1)
	le_low_sbr : QLineEdit (float)
		Prior for what 'Low' SBR means (e.g. ~2)
	le_high_sbr : QLineEdit (float)
		Prior for what 'High' SBR means (e.g. ~10)
	le_min_frames : QLineEdit (int)
		Determines what 'Dead' means (i.e. if bleaching point is before this value
	le_skip_frames : QLineEdit (int)
		Ignored this many frames from the start of the calculation (e.g. skip the first 5)
	class_actions : list[QComboBox]
		The action to take for traces in each class
	# fig : matplotlib.figure.Figure
	# ax : matplotlib.axes._subplots.AxesSubplot
	# toolbar : matplotlib.backend_bases.NavigationToolbar2

	Notes
	-----
	"Calculate" means perform the model selection calculate
	"Process" means perform the action specified in `self.class_actions`

	only really working now for two colors

	'''
	def __init__(self,maven):
		self.maven = maven
		self.model_lists = [None]*5

		self.reset_defaults()

		self.mode_labels = ['Donor','Acceptor','Donor+Acceptor']
		self.model_labels = ['Dead','Low SBR, Bleach', 'High SBR, Bleach','Low SBR, No bleach', 'High SBR, No bleach']


	def reset_defaults(self):
		'''Set default values'''
		self.low_sbr = 2.0
		self.high_sbr = 5.0
		self.min_frames = 10
		self.skip_frames = 0
		self.mode = 2

	def calculate_model(self):
		''' Perform SBR model selection calculation

		Calculation details in `tmaven/selection/model_selection.py` in `model_select_many`
		Takes the max model selection probability as assignment. Plots histograms of assigned data on `self.ax`.

		Notes
		-----
		Will JIT compile things the first time it runs, so might be a delay.
		'''
		logger.info('Compiling model_selection.model_select_many')
		from .model_selection import model_select_many
		logger.info('Running trace_filter.calculate_model. %f %f %d %d'%(self.low_sbr,self.high_sbr,self.min_frames,self.skip_frames))

		mode_text = self.mode_labels[self.mode]

		if self.mode == 2:
			self.corrected = self.maven.data.corrected[:,:,0].copy() + self.maven.data.corrected[:,:,1].copy()
		else:
			self.corrected = self.maven.data.corrected[:,:,self.mode].copy()

		if self.skip_frames >= self.corrected.shape[0]:
			logger.error('Too many skipped frames')
			return

		logger.info('Running Filter Calculation: %d, %d, %i, %i, %s'%(self.low_sbr,self.high_sbr,self.min_frames,self.skip_frames,mode_text))

		self.corrected = self.corrected.astype('double')
		self.probs = model_select_many(self.corrected[self.skip_frames:],self.low_sbr,self.high_sbr,self.min_frames)
		logger.info('Finished Filter Calculation')

		pmax = self.probs.argmax(1)
		self.model_lists = [None]*5
		self.model_lists[0] = np.nonzero(np.bitwise_or(pmax == 0, pmax==3))[0] ## dead, any SBR
		self.model_lists[1] = np.nonzero(pmax == 1)[0] ## Bleaching, Low SBR
		self.model_lists[2] = np.nonzero(pmax == 4)[0] ## Bleaching, High SBR
		self.model_lists[3] = np.nonzero(pmax == 2)[0] ## No bleaching, Low SBR
		self.model_lists[4] = np.nonzero(pmax == 5)[0] ## No bleaching, High SBR

		nums = [len(ml) for ml in self.model_lists]
		self.label_proportions=str(nums)
		msg = 'Trace Filter:\n'
		for i in range(len(nums)):
			msg+='\t%s - %d\n'%(self.model_labels[i],nums[i])
		logger.info(msg)

	def plot(self):
		if np.any([ml is None for ml in self.model_lists]):
			logger.error('Calculate trace filter model first')
			return
		import matplotlib.pyplot as plt
		fig,ax = plt.subplots(1)
		self._plot(fig,ax)
		fig.tight_layout()
		plt.show()

	def _plot(self,fig,ax):
		# import matplotlib.pyplot as plt
		logger.info('Plotting Trace Filter')

		try:
			self.corrected is None
		except:
			self.corrected = self.maven.data.corrected

		xmin = self.corrected.min()
		xmax = self.corrected.max()
		delta = xmax-xmin
		xmin -= delta*.05
		xmax += delta*.05

		colors = ['gray','blue','red','green','orange']
		for i in range(len(colors)):
			ml = self.model_lists[i]
			label = self.model_labels[i]
			if ml is None:
				continue
			ax.hist(self.corrected[ml].flatten(), range=(xmin,xmax), bins=300, histtype='step', lw=1.2, log=True, alpha=.8, label=label, color=colors[i])
		ax.hist(self.corrected.flatten(), range=(xmin,xmax), bins=300, histtype='step', lw=1.2, log=True, alpha=.8, label='All',color='k',ls='--')
		ax.legend(loc=1,fontsize=6)
		ax.set_ylabel('Number of Datapoints')
		ax.set_xlabel('Intensity')
		fig.subplots_adjust(left=.18,right=.95,bottom=.2,top=.95)



	def process_classify(self,model_class,maven_class):
		''' After classifying traces, put them in a maven class
		Input
		------
		model_class - int
			0 - 'Dead',
			1 - 'Low SBR, Bleach'
			2 - 'High SBR, Bleach'
			3 - 'Low SBR, No bleach'
			4 - 'High SBR, No bleach'
		maven_class - int
			0-9

		Updates `maven.data`
		'''
		if np.any([ml is None for ml in self.model_lists]):
			logger.error('Calculate trace filter model first')
			return

		if model_class < len(self.model_lists):
			new_classes = np.copy(self.maven.data.classes)
			keep = np.ones(new_classes.size,dtype='bool')

			ml = self.model_lists[model_class]
			new_classes[ml] = maven_class
			self.maven.data.classes = new_classes.copy()
			logger.info('Put %s in class %s'%(self.model_labels[model_class],maven_class))

	def process_remove(self,model_class):
		''' After classifying traces, remove a class
		Input
		------
		model_class - int
			0 - 'Dead',
			1 - 'Low SBR, Bleach'
			2 - 'High SBR, Bleach'
			3 - 'Low SBR, No bleach'
			4 - 'High SBR, No bleach'

		Updates `maven.data`
		'''
		if np.any([ml is None for ml in self.model_lists]):
			logger.error('Calculate trace filter model first')
			return

		if model_class < len(self.model_lists):
			new_classes = np.copy(self.maven.data.classes)

			ml = self.model_lists[model_class]
			keep = np.ones(new_classes.size,dtype='bool')
			keep[ml] = False

			old = self.maven.data.classes.copy()
			self.maven.data.classes = new_classes.copy()
			if self.maven.cull.cull_remove_traces(keep):
				msg = "Filtered traces: kept %d out of %d = %f"%(keep.sum(),keep.size,keep.sum()/float(keep.size))
				logger.info(msg)
				self.maven.emit_data_update()
			else:
				self.maven.data.classes = old.copy()
				logger.error('trace filter - failed to remove model class %d'%(model_class))

			self.model_lists = [None]*5
			self.label_proportions=""
