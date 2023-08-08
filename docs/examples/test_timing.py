#### DIRECTIONS
## Launch `python -m tmaven --log_stdout --startup=testin/dev_boot.py`
## Use Scripts > Run Script
## Select test_timing.py
## Output is at the end in the log

import numpy as np
import matplotlib.pyplot as plt
import time

niters = 10
nstates = 2
msmax = 6
threshold = 0.5

fxns_mixture = {
	'threshold':lambda : maven.modeler.run_fret_threshold(threshold),
	'kmeans':lambda : maven.modeler.run_fret_kmeans(nstates),
	'mlgmm':lambda : maven.modeler.run_fret_mlgmm(nstates),
	'vbgmm':lambda : maven.modeler.run_fret_vbgmm(nstates),
}

fxns_hmm = {
	'mlhmm':lambda : maven.modeler.run_fret_mlhmm(nstates),
	'vbhmm':lambda : maven.modeler.run_fret_vbhmm(nstates),
}
fxns_global = {
	'vbconhmm':lambda : maven.modeler.run_fret_vbconhmm(nstates),
	'ebhmm':lambda : maven.modeler.run_fret_ebhmm(nstates),
}
fxns_composite = {
	'kmeans_mlhmm':lambda : maven.modeler.run_fret_kmeans_mlhmm(nstates),
	'kmeans_vbhmm':lambda : maven.modeler.run_fret_kmeans_vbhmm(nstates),
	'vbgmm_vbhmm':lambda : maven.modeler.run_fret_vbgmm_vbhmm(nstates),
	'threshold_vbhmm':lambda : maven.modeler.run_fret_threshold_vbhmm(nstates,threshold),
	'threshold_vbconhmm':lambda : maven.modeler.run_fret_threshold_vbconhmm(nstates,threshold),	
}

fxns_modelselection = {
	'vbgmm_modelselection':lambda : maven.modeler.run_fret_vbgmm_modelselection(1,msmax),
	'vbhmm_modelselection':lambda : maven.modeler.run_fret_vbhmm_modelselection(1,msmax),
	'vbconhmm_modelselection':lambda : maven.modeler.run_fret_vbconhmm_modelselection(1,msmax),
	# 'ebhmm_modelselection':lambda : maven.modeler.run_fret_ebhmm_modelselection(1,msmax), ## this seems to hang for some reason
	'vbgmm_vbhmm_modelselection':lambda : maven.modeler.run_fret_vbgmm_vbhmm_modelselection(1,msmax),
}


modeltypes = ['Mixture','HMM','Composite','Global','w/ Model Selection(1-6)']
modeldicts = [fxns_mixture,fxns_hmm,fxns_global,fxns_composite,fxns_modelselection]
# modeltypes = ['w/ Model Selection(1-6)']
# modeldicts = [fxns_modelselection]


out = {}
for modeltype,modeldict in zip(modeltypes,modeldicts):
	output = {}
	for key in modeldict.keys():
		ts = []
		for _ in range(niters):
			t0 = time.time()
			modeldict[key]()
			t1 = time.time()
			ts.append(t1-t0)
		medt = np.median(t1-t0)
		output[key] = medt
	out[modeltype] = output
	
for keytype in out.keys():
	print("\n| %s | Time (s) |"%(keytype))
	print("|----|----|")
	for key in out[keytype].keys():
		print("| %s | %.3f |"%(key,out[keytype][key]))
		
		
		
		
		
		
		