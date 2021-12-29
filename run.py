from tmaven import run_app
# run_app(['--safe_mode','--log_stdout','--startup=test_script.py'])
run_app(['--startup=test_script.py','--log_stdout'])
# run_app(['--log_stdout'])

# from tmaven.app import setup_maven
# maven  = setup_maven([])
# # print(maven.prefs)
# # print(maven.data.raw.shape)
# smd = maven.io.load_smd_hdf5('./example_data.hdf5','L1-L9')
# maven.io.add_data(smd)

#
# import matplotlib.pyplot as plt
# plt.plot(maven.data.corrected[0,:,0],'g')
# plt.plot(maven.data.corrected[0,:,1],'r')
# plt.show()
