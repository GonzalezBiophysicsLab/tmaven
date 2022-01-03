import numpy as np
import multiprocessing as mp
import logging
logger = logging.getLogger(__name__)

default_prefs = {
	'ncpu':mp.cpu_count(),
	'useless':np.array([[1,2,4.],[2,.3,4]],dtype='float16')
}

class pref_item(object):
	def __init__(self,key,val,dtype=None):
		self.key = key
		self.val = val
		self.dtype = dtype

class prefs_object(dict):
	'''
	Numpy aware dictionary sub-class that remembers the initial datatype of a format and enforces the initial type.
	Override .emit_changed and .emit_failed to do something upon changing/failing to change values
	'''
	def __init__(self,*args,**kwargs):
		self.update(*args,**kwargs)

	def __setitem__(self, key, val, quiet=False):
		'''Try to create new or modify an old pref_item like a dictionary'''

		if key in self.keys():
			success = False
			dk = dict.__getitem__(self,key)

			## Numpy array -- match types
			if dk.dtype==np.ndarray and type(val)==np.ndarray:
				if val.dtype != dk.val.dtype:
					val = val.astype(dk.val.dtype)
				success = True

			## List - match type of first element... lazy
			elif dk.dtype == list and type(val) == list:
				if type(val[0]) == type(dk.val[0]): ## just audit the first one
					success = True

			## everything else
			elif type(val) == dk.dtype:
				success = True

			s = 'changed' if success else 'failed'
			logger.info('preference %s %s:%s:%s --> %s:%s:%s'%(s,key,dk.dtype,dk.val,key,type(val),val))
			if success:
				pi = pref_item(key,val,type(val))
				dict.__setitem__(self,key,pi)
				if not quiet: self.emit_changed()
			else:
				if not quiet: self.emit_failed()

		else:
			pi = pref_item(key,val,type(val))
			dict.__setitem__(self,key,pi)
			if not quiet: self.emit_changed()
			# logger.info('preference new %s:%s:%s'%(key,pi.dtype,pi.val))

	def __getitem__(self, key):
		''' return the value, not a preference_item '''
		val = dict.__getitem__(self,key).val
		return val

	def __str__(self):
		''' output format to look like an evalutable ditionary string '''
		out = []
		for key in self.keys():
			dk = dict.__getitem__(self,key)
			if dk.dtype == np.ndarray:
				import re
				val = str(np.array_repr(dk.val))
				val = re.sub(r"dtype=(\w+)\)$",r"dtype='\1')",val)
				val = re.sub(r"^array",r"np.array",val)
				val = re.sub("\n",r"",val)
				val = re.sub("    ",r"",val)
			elif dk.dtype is type('a'):
				val = "\"%s\""%(str(dk.val))
			else:
				val = dk.val
			out.append('%s:%s'%(key,val))
		# return '{'+',\n'.join(out)+'}'
		out = '{' + ',\n'.join(out) + '}'
		return out

	def dtype(self,key):
		return dict.__getitem__(self,key).dtype

	def emit_changed(self):
		pass
	def emit_failed(self):
		pass

	def add_dictionary(self,dictionary):
		''' Adds an entire dictionary of preferences '''
		for k in dictionary.keys():
			try:
				self.__setitem__(k,dictionary[k],quiet=True)
			except Exception as e:
				pass
				# logger.info('failed to set %s as %s\n(%s)'%(k,dictionary[k],str(e)))
		self.emit_changed()

	def load(self,fname=None,quiet=True):
		''' Load a text file of preferences

		file is assumed to be ":" delimited for each entry, with each entry on a new line

		Parameters
		----------
		fname : str
			filename of file to load
		quiet : bool
			if True, do not emit self.emit_changed for each added entry. Only once done
		'''

		if fname is None:
			return

		from ast import literal_eval ## for safety instead of eval
		### might this approach have namespace issues? et right now I have numpy as np but what if a maniac did just numpy?
		try:
			d = {}
			with open(fname,'r') as f:
				for line in f:
					l = line.rstrip('\n')
					l = l.rstrip('\r')
					l = l.rstrip(',')
					l = l.split(':')
					if len(l) == 2:
						try:
							d[l[0]] = eval(l[1]) # literal_eval fails on numpy arrays?
						except:
							pass ## seems safer.... less likely to trip people up
							# d[l[0]] = l[1] ## use a string
			if type(d) is dict:
				self.add_dictionary(d)
				logger.info('Loaded preferences from %s'%(fname))
				return
		except:
			pass
		logging.error('Failed to load preferences from %s'%(fname))

	def save(self,fname=None):
		'''Save the current preferences into a text file

		Parameters
		----------
		fname : str
			the textfile name to save in. If `None`, prompts user with QFileDialog

		Notes
		-----
		Saves each entry in a new line, and the key and value are separated by ":"

		'''
		if fname is None:
			return
		try:
			ps = self.__str__()
			with open(fname,'w') as f:
				f.write(ps[1:-1])
			logger.info('Saved preferences to %s'%(fname))
		except:
			logging.error('Failed to save preferences to %s'%(fname))




if __name__ == '__main__':
	p = prefs_object()
	a = {
		'asdf':1,
		'qwer':'dfd',
		'wer':np.array([[1,2,4.],[2,.3,4]],dtype='float16')
	}
	p.add_dictionary(default_prefs)
	p.add_dictionary(a)
	p['qwer'] = 1 ## should fail

	print(p)
	p.save('test_prefs.txt')
	p = prefs_object()
	print(p)
	p.load('test_prefs.txt')
	print(p)
