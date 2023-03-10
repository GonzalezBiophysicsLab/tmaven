import numpy as np
import logging
logger = logging.getLogger(__name__)

class controller_cull(object):
	''' Handles removing traces from data

	Parameters
	----------

	'''
	def __init__(self,maven):
		self.maven = maven

	def cull_short(self,min_length,current_index=None):
		'''
		Remove trajectories with number of kept-frames < threshold

		Parameters
		----------
		min_length : int
			Number of frames a trace must have to keep.

		Returns
		-------
		success : bool
		'''

		keep = self.maven.data.post_list-self.maven.data.pre_list > min_length
		msg = "Cull short traces < %d: kept %d out of %d = %f"%(min_length,keep.sum(),keep.size,keep.sum()/float(keep.size))
		return self.cull_remove_traces(keep,msg,current_index)

	def cull_class(self,c,current_index=None):
		keep = self.maven.data.classes != c
		msg = "Cull traces: kept %d out of %d = %f %%, class %d"%(keep.sum(),keep.size,keep.sum()/float(keep.size),c)
		return self.cull_remove_traces(keep,msg,current_index)

	def cull_min(self,threshold,current_index=None):
		'''
		Remove trajectories with one or more datapoints of value < min_threshold

		Parameters
		----------
		min_threshold : double
			Value that if any datapoints have a value less than, the trace is removed.

		Returns
		-------
		success : bool
		'''
		keep = np.min(self.maven.data.corrected,axis=(1,2)) > threshold
		msg = "Cull traces: kept %d out of %d = %f %%, with a value less than %f"%(keep.sum(),keep.size,keep.sum()/float(keep.size),threshold)
		return self.cull_remove_traces(keep,msg,current_index)


	def cull_max(self,threshold,current_index=None):
		'''
		Remove trajectories with one or more datapoints of value > max_threshold

		Parameters
		----------
		max_threshold : double
			Value that if any datapoints have a value greater than, the trace is removed.

		Returns
		-------
		success : bool
		'''
		keep = np.max(self.maven.data.corrected,axis=(1,2)) < threshold
		msg = "Cull traces: kept %d out of %d = %f %%, with a value greater than %f"%(keep.sum(),keep.size,keep.sum()/float(keep.size),threshold)
		return self.cull_remove_traces(keep,msg,current_index)

	def cull_remove_traces(self,mask,msg='',current_index=None):
		''' Keeps only those traces in `container_data` specified by mask

		Parameters
		----------
		mask : np.ndarray (bool;int)
			The traces to keep in `container_data`
		msg : str
			The message to emit regarding how the mask was generated

		Returns
		-------
		success : bool

		Notes
		-----
		Mask can be bool or integer so be careful...
		'''
		if mask.sum() == 0:
			logger.error("cannot remove all traces")
			if current_index is None:
				return False
			return False, current_index
		else:
			## check current_index changes
			if not current_index is None:
				if mask.dtype == 'bool':
					if mask[current_index] == True:
						current_index = mask[:current_index].sum()
					else:
						current_index = 0
				else:
					if current_index in mask:
						current_index = np.argmax(mask==current_index)
					else:
						current_index = 0
			self.maven.data.order(mask)
			self.maven.emit_data_update()
			logger.info(msg)
			if current_index is None:
				return True
			return True,current_index
