smd = maven.io.load_smd_hdf5('./example_data.hdf5','L1-L9')

import h5py
with h5py.File('example_data.hdf5','r') as f:
	pre = f['L1-L9/time_series/pre_list'][:]
	post = f['L1-L9/time_series/post_list'][:]

smd.raw = smd.raw.astype('double')
maven.io.add_data(smd)
maven.data.pre_list = pre
maven.data.post_list = post
# print(maven.data.corrected.shape)
# print(maven.smd.smd_dict)
maven.emit_data_update()
