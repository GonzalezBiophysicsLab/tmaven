from tmaven import run_app
# run_app(['--safe_mode','--log_stdout','--startup=notes/test_script.py'])
# run_app(['--startup=notes/test_script.py','--log_stdout'])
# run_app(['--log_stdout'])

run_app([])


# from tmaven.app import setup_maven
# maven  = setup_maven(['--log_stdout'])
# maven.io.load_smdtmaven_hdf5('./notes/example_smd.hdf5','L1-tRNA')
# # maven.data.flag_ons*=False
# # maven.data.flag_ons[:10]+=True
# # maven.modeler.run_fret_vbhmm_modelselection(1,6)
# # maven.modeler.run_fret_threshold(.5)
# # maven.plots.fret_hist1d.prefs['idealized'] = False
#
#
# import matplotlib.pyplot as plt
# fig,ax =plt.subplots(1)
# # maven.plots.fret_hist1d.plot(fig,ax)
# # maven.plots.fret_hist2d.plot(fig,ax)
# # maven.plots.fret_tdp.plot(fig,ax)
# maven.plots.model_vbstates.plot(fig,ax)
#
#
# # plt.plot(maven.data.corrected[0,:,0],'g')
# # plt.plot(maven.data.corrected[0,:,1],'r')
# plt.show()
