import numpy as np
import numba as nb
import logging
logger = logging.getLogger(__name__)

def generate_dwells(trace, dwell_list, means):
    trace = trace[~np.isnan(trace)]
    #print(trace)
    if len(trace) > 0: #proetcting if all is NaN
        dwell_split = np.split(trace, np.argwhere(np.diff(trace)!=0).flatten()+1)

        if len(dwell_split) > 2: #protecting against no or single transition in a trace
            dwell_split = dwell_split[1:-1] #skipping first and last dwells
            for d in dwell_split:
                ind = int(np.argwhere(d[0] == means))
                dwell_list[str(ind)].append(len(d))

    return dwell_list

def calculate_dwells(result):
    traces = result.idealized
    means = result.mean
    dwell_list = {str(i):[] for i in range(result.nstates)}

    for t in range(len(traces)):
        dwell_list = generate_dwells(traces[t],dwell_list, means)

    result.dwells = dwell_list

@nb.njit
def survival(dist):
    n = np.int(np.max(dist))

    raw_surv = np.zeros(n)

    for i in np.arange(n):
        temp = np.zeros_like(dist)
        temp[np.where(dist > i)] = 1
        raw_surv[i] = np.sum(temp)

    norm_surv = raw_surv/raw_surv[0]

    return np.arange(n), norm_surv

def analyze_dwells(gui):
    #try: ## if no appropriate model exists, it will die at some point
    model = gui.maven.modeler.model
    calculate_dwells(model)
        #maven.modeler.model.dwells = dwells
    logger.info('Dwell time calculated')
    #except Exception as e:
        #logger.error('Failed dwell time\n{}'.format(e))
    return
