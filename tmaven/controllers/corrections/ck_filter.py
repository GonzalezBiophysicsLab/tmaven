import numpy as np
import matplotlib.pyplot as plt


def ck_filter(data, pert_window=10, filters=[2,4,6,8,10], p=2.):
	''' Chung-Kennedy filter

	ck_filter implements the Chung-Kennedy filter (DOI: 10.1016/0165-0270(91)90118-J)
	on a time series of N-dimensional data which is formatted as Time*Data. ~KKR

	Parameters
	----------
	data: np.ndarray
		either a 1-D time series or N-D time series (as T*D).
	pert_window' : int
		the window for the calculation of the variances of filtered vs real data. Recommended value: 10.
	filters : list
		a list which contains the windows for the bank of filters. Recommended list: [2,4,6,8,10].
	p : int
		is an integer which modulates the sensitivity of the CK filter. Recommended value: 2.


	'''

	start = int(np.max(filters) + pert_window)

	if len(data.shape) == 1:
		data = np.array([data]).T #standardising 1D data with multi-D data arrays
		# print("1D? We handle 'em just fine'")

	filtered_data = np.zeros_like(data[start:-start, :])
	#print(data.shape)

	forward_filters = []
	backward_filters = []

	for f in filters:
		forward_filters.append(causal(data, f))
		backward_filters.append(anti_causal(data, f))

	forward_filters = np.array(forward_filters)
	backward_filters = np.array(backward_filters)

	#forward_filters and backward_filters are F*T*D arrays, where F = filters, T = time, D = data

	dif_forward = (forward_filters - data[None, :, :])**2
	pert_for = dif_forward.cumsum(1)
	pert_for[:, pert_window:, :] -= pert_for[:, :-pert_window, :]
	pert_for /= np.amin(pert_for, axis = 0)
	forward_weights = (pert_for.sum(-1))**(-p)
	#print(forward_weights.min())

	dif_backward = (backward_filters - data[None, :, :])**2
	pert_back = np.flip(np.flip(dif_backward, 1).cumsum(1), 1)
	pert_back[:, :-pert_window, :] -= pert_back[:, pert_window:, :]
	pert_back /= np.amin(pert_back, axis = 0)
	backward_weights = (pert_back.sum(-1))**(-p)

	#forward_weights and backward_weights should be F*T at the end of this

	filtered_data = (forward_weights[:, :, None]*forward_filters + backward_weights[:, :, None]*backward_filters).sum(0)
	filtered_data /= (forward_weights.sum(0) + backward_weights.sum(0))[:, None]

	return filtered_data

def causal(data, window):
	filtered = data.cumsum(0)
	filtered[window:, :] -= filtered[:-window, :]
	filtered /= window

	return filtered

def anti_causal(data, window):
	filtered = np.flip(np.flip(data, 0).cumsum(0), 0)
	filtered[:-window, :] -= filtered[window:, :]
	filtered /= window

	return filtered


# if __name__ == '__main__':
# 	d1 = np.random.normal(size=(100,2)) + np.array((5.,8.))[None,:]
# 	d2 = np.random.normal(size=(100,2))  + np.array((3.,5.))[None,:]
# 	d = np.append(d1,d2,axis=0)
#
# 	y = ck_filter(d)
# 	import matplotlib.pyplot as plt
# 	plt.plot(d[:,0],alpha=.5)
# 	plt.plot(d[:,1],alpha=.5)
# 	plt.plot(y[:,0],alpha=.5)
# 	plt.plot(y[:,1],alpha=.5)
# 	plt.show()
