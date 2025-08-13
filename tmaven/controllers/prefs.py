"""
A simple, numpy-aware, TOML-backed preferences system with per-key type and min/max enforcement.
"""
import re
try:
	import tomllib as toml ## 3.11 and later
except:
	import toml ## pre-3.11 ... external
import logging
import numpy as np
logger = logging.getLogger(__name__)

default_prefs = '''
ncpu = 1
'''

def load_toml_string(toml_str):
	"""
	Parse a TOML string where each entry either is a bare dotted key → treated as default-only, or is a [section.key] table with default/min/max.
	Returns a dict mapping each full key name to a dict of its attrs.

	Example Input:
		[correction]
		filterwidth = "asdf"
		bleedthrough = 0.05

		[correction.gamma]
		default = 1.
		min = 0.
		max = 1.

		[correction.backgroundframes]
		default = 100
		min = 1
	"""

	# Fix leading-dot floats like `.5` → `0.5`
	toml_str_fix = re.sub(r'(?<![\d])\.(\d)', r'0.\1', toml_str)
	# normalize floats with trailing dot (e.g., "1.") to valid "1.0"
	toml_str_fix = re.sub(r'(?P<n>\d+)\.(?=\s|$)', r'\g<n>.0', toml_str_fix)
	# Fix weirdly capitalized `true` values --> True
	toml_str_fix = re.sub(r'=\s*["\']?(true)["\']?', r'= true', toml_str_fix, flags=re.IGNORECASE)
	# Fix weirdly capitalized `false` values --> false
	toml_str_fix = re.sub(r'=\s*["\']?(false)["\']?', r'= false', toml_str_fix, flags=re.IGNORECASE)
	# for i, line in enumerate(toml_str_fix.splitlines(), 1):
	# 	print(f"{i:2}: {line}")

	data = toml.loads(toml_str_fix)
	tomlprefs = {}
	def walk(prefix, node):
		for k, v in node.items():
			# build the literal dotted name
			name = f"{prefix}.{k}" if prefix else k

			if isinstance(v, dict):
				if 'default' in v: # table form → pull default, optional min/max
					tomlprefs[name] = {'default': v['default']}
					tomlprefs[name]['min'] = v.get('min') if 'min' in v else None
					tomlprefs[name]['max'] = v.get('max') if 'max' in v else None
				else: # nested tables → dive deeper
					walk(name, v)
			else: 
				# bare dotted assignment → default-only
				tomlprefs[name] = {'default': v, 'min': None, 'max': None}
				# continue
	walk('', data)
	return tomlprefs

class pref_item(object):
	"""Holds a single preference value along with its type and optional bounds."""
	def __init__(self, key, default, min=None, max=None):
		self.key = key
		self.default = default
		self.val = self.default
		self.dtype = type(self.default)
		self.min = min
		self.max = max

	def reset(self):
		self.val = self.default

	def within_limits(self, val):
		"""Return True if val is within [min_val, max_val], if bounds are set."""
		if not self.min is None:
			if val < self.min:
				return False
		if not self.max is None:
			if val > self.max:
				return False
		return True

class prefs_object(dict):
	'''
	Numpy aware dictionary sub-class that remembers the initial datatype of a format and enforces the initial type.
	Override .emit_changed and .emit_failed to do something upon changing/failing to change values
	'''
	def __init__(self,*args,**kwargs):
		self.update(*args,**kwargs)

	def create_item(self,key,default,min=None,max=None):
		dict.__setitem__(self,key,pref_item(key,default,min,max))
	
	def add(self,newstuff,quiet=True):
		if type(newstuff) is str:
			self.add_toml(newstuff,quiet)
		elif type(newstuff) is dict:
			self.add_dictionary(newstuff,quiet)
		else:
			raise Exception('prefs.py does not know how to parse this'
				   )
	def add_toml(self,toml_string,quiet=True):
		### CREATE NEW ENTRIES
		defs = load_toml_string(toml_string)
		for key in defs:
			if not key in self:
				self.create_item(key,defs[key]['default'],defs[key]['min'],defs[key]['max'])
			else:
				self.__setitem__(key, defs[key]['default'], quiet=True)
		if not quiet:
			self.emit_changed()

	def add_dictionary(self,newdict,quiet=True):
		### CREATE NEW ENTRIES
		for key in newdict:
			if not key in self:
				self.create_item(key,newdict[key],None,None)
			else:
				self.__setitem__(key, newdict[key], quiet=True)
		if not quiet:
			self.emit_changed()
	
	# def load_toml(self,toml_string,quiet=False):
	# 	### UPDATE ENTRIES
	# 	defs = load_toml_string(toml_string)
	# 	for key in defs:
	# 		if key in self:
	# 			self.__setitem__(key, defs[key]['default'], quiet=True)
	# 	if not quiet:
	# 		self.emit_changed()
	
	# def load_dictionary(self,newdict,quiet=False):
	# 	### UPDATE ENTRIES
	# 	for key in newdict:
	# 		if key in self:
	# 			self.__setitem__(key, newdict[key], quiet=True)
	# 	if not quiet:
	# 		self.emit_changed()

	def __setitem__(self, key, new_val, quiet=False):
		'''Try to modify an old pref_item like a dictionary'''
		
		success = False
		if key in self.keys(): ## 1. if there is a preference called key
			prefi = dict.__getitem__(self,key)
			if prefi.dtype is type(new_val): ## 2. check that the replacement is the same type
				
				## handle arrays
				success_arrays = True
				if prefi.dtype is np.ndarray: ## NumPy
					if not new_val.dtype is prefi.val.dtype: ## enforce same datatype 
						new_val = new_val.astype(prefi.val.dtype)
					if not prefi.val.shape == new_val.shape: ## enforce same shape
						success_arrays = False
				elif prefi.dtype is list: ## list
					if len(prefi.val) == len(new_val): ## enforce same shape
						for i in range(len(prefi.val)): ## enforce same datatypes
							if not type(prefi.val[i]) == type(new_val[i]):
								success_arrays = False
					else:
						success_arrays = False

				if success_arrays:
					if prefi.within_limits(new_val): ## 3. enforce min/max
						prefi.val = new_val
						dict.__setitem__(self,key,prefi)
						success = True
			
			elif prefi.dtype in [float,int] and type(new_val) in [float,int]:
				new_val = prefi.dtype(new_val)
				if prefi.within_limits(new_val): ## 3. enforce min/max
					prefi.val = new_val
					dict.__setitem__(self,key,prefi)
					success = True

		if not quiet:
			if success:
				self.emit_changed()
			else:
				self.emit_failed()
			
	def __getitem__(self, key):
		''' return the value, not a preference_item '''
		val = dict.__getitem__(self,key).val
		return val

	def format(self,style='full'):
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
			if style == 'full':
				out.append('[%s]\ndefault = %s\nmin = %s\nmax = %s'%(key,val,str(dk.min),str(dk.max)))
			else: ## thin
				out.append('%s = %s'%(key,val))
		out = '\n'.join(out) ## toml
		return out

	def __str__(self):
		return self.format(style='thin')

	def dtype(self,key):
		return dict.__getitem__(self,key).dtype
	def limits(self,key):
		return [dict.__getitem__(self,key).min, dict.__getitem__(self,key).max]

	def emit_changed(self):
		pass
	def emit_failed(self):
		pass

if __name__ == '__main__':
	p = prefs_object()

	s = '''
	[correction]
	filterwidth = "asdf"
	bleedthrough.asdf = 0.05

	[correction.gamma]
	default = 1.
	min = 0.
	max = 1.

	[correction.backgroundframes]
	default = 100
	min = 1
	'''
	p.add_toml(s,quiet=False)
	print(p.format())
	
	print(p['correction.filterwidth'])
	print(p['correction.gamma'])
	p['correction.gamma'] = -1
	print(p['correction.gamma'])
	p.add(s,quiet=False)
	print(p['correction.gamma'])

	print(p)
	print(p.dtype('correction.filterwidth'))
	print(p.limits('correction.gamma'))

	
