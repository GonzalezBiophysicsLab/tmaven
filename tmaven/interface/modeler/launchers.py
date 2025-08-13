import logging
logger = logging.getLogger(__name__)

from .ui_modeler import close_dialog
from . import dialogs

def launch_threshold(gui):
	method_name = 'Threshold'
	fxn = gui.maven.modeler.run_threshold
	logger.info(f'Launching {method_name} dialog')
	dialogs.dialog_threshold(gui, lambda: general_fxn(gui,method_name,fxn))
	gui.model_dialog.start()

def launch_threshold_jump(gui):
	method_name = 'Jump Threshold'
	fxn = gui.maven.modeler.run_threshold_jump
	logger.info(f'Launching {method_name} dialog')
	dialogs.dialog_threshold_jump(gui, lambda: general_fxn(gui,method_name,fxn))
	gui.model_dialog.start()

def launch_kmeans(gui):
	method_name = 'K-means'
	fxn = gui.maven.modeler.run_kmeans
	logger.info(f'Launching {method_name} dialog')
	dialogs.dialog_kmeans(gui, lambda: general_fxn(gui,method_name,fxn))
	gui.model_dialog.start()

def launch_mlhmm(gui):
	method_name = 'ML HMM'
	fxn = gui.maven.modeler.run_mlhmm
	logger.info(f'Launching {method_name} dialog')
	dialogs.dialog_mlmm(gui, lambda: general_fxn(gui,method_name,fxn),method_name, hmm_not_gmm=True)
	gui.model_dialog.start()

def launch_kmeans_mlhmm(gui):
	method_name = 'ML HMM + K-means'
	fxn = gui.maven.modeler.run_kmeans_mlhmm
	logger.info(f'Launching {method_name} dialog')
	dialogs.dialog_mlmm(gui, lambda: general_fxn(gui,method_name,fxn),method_name, hmm_not_gmm=True)
	gui.model_dialog.start()

def launch_mlhmm_one(gui):
	keep = gui.maven.modeler.get_traces()
	if keep.sum() != 1:
		from PyQt5.QtWidgets import QMessageBox
		QMessageBox.warning(gui,"Error: Run all, Apply all","You have more than one molecule turned on")
		return
	method_name = 'ML HMM (One)'
	fxn = gui.maven.modeler.run_mlhmm_one
	logger.info(f'Launching {method_name} dialog')
	dialogs.dialog_mlmm(gui, lambda: general_fxn(gui,method_name,fxn),method_name, hmm_not_gmm=True)
	gui.model_dialog.start()

def launch_vbgmm(gui):
	method_name = 'VB GMM'
	fxn = gui.maven.modeler.run_vbgmm
	logger.info(f'Launching {method_name} dialog')
	dialogs.dialog_vbgmm(gui, lambda: general_fxn(gui,method_name,fxn),method_name,model_selection=False)
	gui.model_dialog.start()

def launch_vbgmm_modelselection(gui):
	method_name = 'VB GMM + Model Selection'
	fxn = gui.maven.modeler.run_vbgmm_modelselection
	logger.info(f'Launching {method_name} dialog')
	dialogs.dialog_vbgmm(gui, lambda: general_fxn(gui,method_name,fxn),method_name,model_selection=True)
	gui.model_dialog.start()

def launch_mlgmm(gui):
	method_name = 'ML GMM'
	fxn = gui.maven.modeler.run_mlgmm
	logger.info(f'Launching {method_name} dialog')
	dialogs.dialog_mlmm(gui, lambda: general_fxn(gui,method_name,fxn),method_name,hmm_not_gmm=False)
	gui.model_dialog.start()

def launch_vbconhmm_modelselection(gui):
	method_name = 'VB Consensus HMM + Model Selection'
	fxn = gui.maven.modeler.run_vbconhmm_modelselection
	logger.info(f'Launching {method_name} dialog')
	dialogs.dialog_vbconhmm(gui, lambda: general_fxn(gui,method_name,fxn),model_selection=True)
	gui.model_dialog.start()

def launch_vbconhmm(gui):
	method_name = 'VB Consensus HMM'
	fxn = gui.maven.modeler.run_vbconhmm
	logger.info(f'Launching {method_name} dialog')
	dialogs.dialog_vbconhmm(gui, lambda: general_fxn(gui,method_name,fxn),model_selection=False)
	gui.model_dialog.start()

def launch_ebhmm(gui):
	method_name = 'EB HMM'
	fxn = gui.maven.modeler.run_ebhmm
	logger.info(f'Launching {method_name} dialog')
	dialogs.dialog_ebhmm(gui, lambda: general_fxn(gui,method_name,fxn),model_selection=False)
	gui.model_dialog.start()

def launch_ebhmm_modelselection(gui):
	method_name = 'EB HMM + Model Selection'
	fxn = gui.maven.modeler.run_ebhmm_modelselection
	logger.info(f'Launching {method_name} dialog')
	dialogs.dialog_ebhmm(gui, lambda: general_fxn(gui,method_name,fxn),model_selection=True)
	gui.model_dialog.start()

def launch_vbhmm(gui):
	method_name = 'VB HMM'
	fxn = gui.maven.modeler.run_vbhmm
	logger.info(f'Launching {method_name} dialog')
	dialogs.dialog_vbhmm(gui, lambda: general_fxn(gui,method_name,fxn),method_name,model_selection=False)
	gui.model_dialog.start()

def launch_vbhmm_modelselection(gui):
	method_name = 'VB HMM + Model Selection'
	fxn = gui.maven.modeler.run_vbhmm
	logger.info(f'Launching {method_name} dialog')
	dialogs.dialog_vbhmm(gui, lambda: general_fxn(gui,method_name,fxn),method_name,model_selection=True)
	gui.model_dialog.start()
	
def launch_kmeans_vbhmm(gui):
	method_name = 'VB HMM -> K-means'
	fxn = gui.maven.modeler.run_kmeans_vbhmm
	logger.info(f'Launching {method_name} dialog')
	dialogs.dialog_vbhmm(gui, lambda: general_fxn(gui,method_name,fxn),method_name,model_selection=False)
	gui.model_dialog.start()

def launch_vbgmm_vbhmm(gui):
	method_name = 'VB HMM -> VB GMM'
	fxn = gui.maven.modeler.run_vbgmm_vbhmm
	logger.info(f'Launching {method_name} dialog')
	dialogs.dialog_vbhmm(gui, lambda: general_fxn(gui,method_name,fxn),method_name,model_selection=False)
	gui.model_dialog.start()

def launch_vbgmm_vbhmm_modelselection(gui):
	method_name = 'VB HMM -> VB GMM + Model Selection'
	fxn = gui.maven.modeler.run_vbgmm_vbhmm_modelselection
	logger.info(f'Launching {method_name} dialog')
	dialogs.dialog_vbhmm(gui, lambda: general_fxn(gui,method_name,fxn),method_name,model_selection=True)
	gui.model_dialog.start()

def launch_threshold_vbhmm(gui):
	method_name = 'VB HMM -> Threshold'
	fxn = gui.maven.modeler.run_threshold_vbhmm
	logger.info(f'Launching {method_name} dialog')
	dialogs.dialog_vbhmm(gui, lambda: general_fxn(gui,method_name,fxn),method_name,model_selection=False)
	gui.model_dialog.start()

def launch_threshold_vbconhmm(gui):
	method_name = 'VB Consensus HMM -> Threshold'
	fxn = gui.maven.modeler.run_threshold_vbconhmm
	logger.info(f'Launching {method_name} dialog')
	dialogs.dialog_vbconhmm(gui, lambda: general_fxn(gui,method_name,fxn),model_selection=False,threshold=True)
	gui.model_dialog.start()

def launch_vbhmm_one(gui):
	keep = gui.maven.modeler.get_traces()
	if keep.sum() != 1:
		from PyQt5.QtWidgets import QMessageBox
		QMessageBox.warning(gui,"Error: Run all, Apply all","You have more than one molecule turned on")
		return
	method_name = 'VB HMM (One)'
	fxn = gui.maven.modeler.run_vbhmm_one
	logger.info(f'Launching {method_name} dialog')
	dialogs.dialog_vbhmm(gui, lambda: general_fxn(gui,method_name,fxn),method_name,model_selection=False,threshold=True)
	gui.model_dialog.start()

def launch_biasd_setup(gui):
	method_name = 'BIASD Setup'
	fxn = gui.maven.modeler.run_biasd_setupfile
	logger.info(f'Launching {method_name} dialog')
	dialogs.dialog_biasdsetup(gui, lambda: general_fxn(gui,method_name,fxn))
	gui.model_dialog.start()

def general_fxn(gui,method_name,run_fxn):
	logger.info(f'Executing {method_name}')
	close_dialog(gui)
	run_fxn()
	logger.info(f'Finished {method_name}')
