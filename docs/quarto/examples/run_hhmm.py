from tmaven.app import setup_maven

#### Analyze data with hHMM
## Make a new MAVEN instance
maven  = setup_maven(['--log_stdout'])

## Load data
maven.io.load_smdtmaven_hdf5('example_smd.hdf5','L1-tRNA')

## Turn on only the first 10 trajectorys
maven.data.flag_ons*=False
maven.data.flag_ons[:10]+=True

# Specify the model parameters
maven.prefs['modeler.hhmm.restarts'] = 0
maven.prefs['modeler.hhmm.maxiters'] = 1
maven.prefs['modeler.hhmm.tolerance'] = 1


## Run 2xw hHMM
maven.modeler.run_fret_hhmm()


# Export
oname = 'result_hhmm.hdf5'
maven.modeler.export_result_to_hdf5(oname)


