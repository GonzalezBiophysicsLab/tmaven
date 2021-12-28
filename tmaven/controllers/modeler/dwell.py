import numpy as np
import logging
logger = logging.getLogger(__name__)

def return_dwells(x, mu):
	state = (x == mu).astype('int')
	start = []
	stop = []

	for i in range(len(x) - 1):
		current = state[i]
		next = state[i + 1]

		if current == 0 and next == 1:
			start.append(i + 1)

		if current == 1 and next == 0:
			stop.append(i + 1)

	if state[0] == 1:
		stop = np.array(stop[1:]).astype('int')
	else:
		stop = np.array(stop)

	if state[-1] == 1:
		start = np.array(start[:-1]).astype('int')
	else:
		start = np.array(start)
	return stop - start

def analyze_dwells(maven):
	dwells = []
	try: ## if no appropriate model exists, it will die at some point
		ran = maven.modeler.model.ran
		if ran == []:
			ran = list(range(maven.modeler.model.idealized.shape[0]))
		for mu in maven.modeler.model.mu:
			dwell_mu = []
			for i in ran:
				pre = maven.data.pre_list[i]
				post = maven.data.post_list[i]
				idealpath = maven.modeler.model.idealized[i, pre:post]
				dwell_mu.append(return_dwells(idealpath, mu))
			dwells.append(dwell_mu)
		maven.modeler.model.dwells = dwells
		logger.info('Dwell time calculated')
	except Exception as e:
		logger.error('Failed dwell time\n{}'.format(e))
		return
