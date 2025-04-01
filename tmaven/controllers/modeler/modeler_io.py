import numpy as np
import types
import h5py as h

def export_dict_to_group(h_group, dicty, attributes=[]):
	for k in dicty.keys():
		if k in attributes:
			pass
		elif not dicty[k] is None:
			if isinstance(dicty[k], dict):
				hh_group = h_group.create_group(k)
				export_dict_to_group(hh_group, dicty[k])
			elif np.isscalar(dicty[k]):
				h_group.create_dataset(k,data=dicty[k])
			elif not isinstance(dicty[k],types.FunctionType):
				h_group.create_dataset(k,data=dicty[k],compression='gzip')


def load_group_to_dict(h_group):
	dicty = {}
	for key, item in h_group.items():
		if isinstance(item, h._hl.dataset.Dataset):
			dicty[key] = item[()]
		elif isinstance(item, h._hl.group.Group):
			dicty[key] = load_group_to_dict(item)
	return dicty
