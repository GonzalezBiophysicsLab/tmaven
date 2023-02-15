import logging
logger = logging.getLogger(__name__)

from .ui_modeler import close_dialog
from . import dialogs

def launch_fret_threshold(gui):
	logger.info('Launching FRET threshold dialog')
	dialogs.dialog_threshold(gui, lambda: fxn_fret_threshold(gui))
	gui.model_dialog.threshold.setText("0.5")
	gui.model_dialog.start()

def launch_fret_kmeans(gui):
	logger.info('Launching FRET Kmeans dialog')
	dialogs.dialog_kmeans(gui, lambda: fxn_fret_kmeans(gui))
	gui.model_dialog.start()

def launch_fret_mlhmm(gui):
	logger.info('Launching FRET ML HMM dialog')
	dialogs.dialog_mlmm(gui,lambda: fxn_fret_mlhmm(gui),'ML HMM')
	gui.model_dialog.start()

def launch_fret_kmeans_mlhmm(gui):
	logger.info('Launching FRET ML HMM+KMeans dialog')
	dialogs.dialog_mlmm(gui,lambda: fxn_fret_kmeans_mlhmm(gui),'ML HMM+Kmeans')
	gui.model_dialog.start()

def launch_fret_mlhmm_one(gui):
	keep = gui.maven.modeler.get_traces()
	print(keep)
	if keep.sum() != 1:
		from PyQt5.QtWidgets import QMessageBox
		QMessageBox.warning(gui,"Error: Run all, Apply all","You have more than one molecule turned on")
		return
	logger.info('Launching FRET ML HMM (One) dialog')
	dialogs.dialog_mlmm(gui,lambda: fxn_fret_mlhmm_one(gui),'ML HMM (One)')
	gui.model_dialog.start()

def launch_fret_vbgmm(gui):
	logger.info('Launching FRET VB GMM')
	dialogs.dialog_vbgmm(gui,lambda: fxn_fret_vbgmm(gui),'VB GMM')
	gui.model_dialog.start()

def launch_fret_vbgmm_modelselection(gui):
	logger.info('Launching FRET VB GMM')
	dialogs.dialog_vbgmm(gui,lambda: fxn_fret_vbgmm_modelselection(gui),'VB GMM',model_selection=True)
	gui.model_dialog.start()

def launch_fret_mlgmm(gui):
	logger.info('Launching FRET ML GMM')
	dialogs.dialog_mlmm(gui,lambda: fxn_fret_mlgmm(gui),'ML GMM')
	gui.model_dialog.start()

def launch_fret_vbconhmm_modelselection(gui):
	logger.info('Launching FRET VB Consensus HMM + Model Selection')
	dialogs.dialog_vbconhmm(gui,lambda: fxn_fret_vbconhmm_modelselection(gui),model_selection=True)
	gui.model_dialog.start()

def launch_fret_vbconhmm(gui):
	logger.info('Launching FRET VB Consensus HMM')
	dialogs.dialog_vbconhmm(gui,lambda: fxn_fret_vbconhmm(gui),model_selection=False)
	gui.model_dialog.start()

def launch_fret_ebhmm(gui):
	logger.info('Launching FRET EB HMM')
	dialogs.dialog_ebhmm(gui,lambda: fxn_fret_ebhmm(gui),model_selection=False)
	gui.model_dialog.start()

def launch_fret_ebhmm_modelselection(gui):
	logger.info('Launching FRET EB HMM + Model Selection')
	dialogs.dialog_ebhmm(gui,lambda: fxn_fret_ebhmm_modelselection(gui),model_selection=True)
	gui.model_dialog.start()

def launch_fret_vbhmm(gui):
	logger.info('Launching FRET VB HMM')
	dialogs.dialog_vbhmm(gui,lambda: fxn_fret_vbhmm(gui),'VB HMM',model_selection=False)
	gui.model_dialog.start()

def launch_fret_vbhmm_modelselection(gui):
	logger.info('Launching FRET VB HMM + Model Selection')
	dialogs.dialog_vbhmm(gui, lambda: fxn_fret_vbhmm_modelselection(gui),'VB HMM+Model selection',model_selection=True)
	gui.model_dialog.start()

def launch_fret_kmeans_vbhmm(gui):
	logger.info('Launching FRET VB HMM -> KMeans')
	dialogs.dialog_vbhmm(gui,lambda: fxn_fret_kmeans_vbhmm(gui),'VB HMM->KMeans',model_selection=False)
	gui.model_dialog.start()

def launch_fret_vbgmm_vbhmm(gui):
	logger.info('Launching FRET VB HMM -> VB GMM')
	dialogs.dialog_vbgmm_vbhmm(gui,lambda: fxn_fret_vbgmm_vbhmm(gui),'VB HMM->VB GMM',model_selection=False)
	gui.model_dialog.start()

def launch_fret_vbgmm_vbhmm_modelselection(gui):
	logger.info('Launching FRET VB HMM -> VB GMM + Model Selection')
	dialogs.dialog_vbgmm_vbhmm(gui,lambda: fxn_fret_vbgmm_vbhmm_modelsection(gui),'VB HMM->VB GMM+Model selection',model_selection=True)
	gui.model_dialog.start()

def launch_fret_threshold_vbhmm(gui):
	logger.info('Launching FRET VB HMM -> Threshold')
	dialogs.dialog_vbhmm(gui,lambda: fxn_fret_threshold_vbhmm(gui),'VB HMM->Threshold',threshold=True)
	gui.model_dialog.threshold.setText("0.5")
	gui.model_dialog.start()

def launch_fret_threshold_vbconhmm(gui):
	logger.info('Launching FRET VB HMM -> Threshold')
	dialogs.dialog_vbconhmm(gui,lambda: fxn_fret_threshold_vbconhmm(gui),threshold=True)
	gui.model_dialog.threshold.setText("0.5")
	gui.model_dialog.start()

def launch_fret_vbhmm_one(gui):
	logger.info('Launching FRET VB HMM (One)')
	keep = gui.maven.modeler.get_traces()
	if keep.sum() != 1:
		from PyQt5.QtWidgets import QMessageBox
		QMessageBox.warning(gui,"Error: Run all, Apply all","You have more than one molecule turned on")
		return
	dialogs.dialog_vbhmm(gui,lambda: fxn_fret_vbhmm_one(gui),'VB HMM (One)')
	gui.model_dialog.start()

def fxn_fret_threshold(gui):
	logger.info('Executing FRET threshold')
	threshold = float(gui.model_dialog.threshold.text())
	close_dialog(gui)
	gui.maven.modeler.run_fret_threshold(threshold)
	logger.info('Finished FRET threshold')

def fxn_fret_kmeans(gui):
	logger.info('Executing FRET Kmeans')
	nstates = gui.model_dialog.nstates.value()
	close_dialog(gui)
	gui.maven.modeler.run_fret_kmeans(nstates)
	logger.info('Finished FRET Kmeans')

def fxn_fret_mlhmm(gui):
	logger.info('Executing FRET ML HMM')
	nstates = gui.model_dialog.nstates.value()
	close_dialog(gui)
	gui.maven.modeler.run_fret_mlhmm(nstates)
	logger.info('Finished FRET ML HMM')

def fxn_fret_kmeans_mlhmm(gui):
	logger.info('Executing FRET ML HMM+KMeans')
	nstates = gui.model_dialog.nstates.value()
	close_dialog(gui)
	gui.maven.modeler.run_fret_kmeans_mlhmm(nstates)
	logger.info('Finished FRET ML HMM+KMeans')

def fxn_fret_mlhmm_one(gui):
	logger.info('Executing FRET ML HMM (One)')
	nstates = gui.model_dialog.nstates.value()
	close_dialog(gui)
	gui.maven.modeler.run_fret_mlhmm_one(nstates)
	logger.info('Finished FRET ML HMM (One)')

def fxn_fret_vbgmm(gui):
	logger.info('Executing FRET VB GMM')
	nstates = gui.model_dialog.nstates.value()
	close_dialog(gui)
	gui.maven.modeler.run_fret_vbgmm(nstates)
	logger.info('Finished FRET VB GMM')

def fxn_fret_vbgmm_modelselection(gui):
	logger.info('Executing FRET VB GMM + Model Selection')
	nstates_min = gui.model_dialog.nstates_min.value()
	nstates_max = gui.model_dialog.nstates_max.value()
	close_dialog(gui)
	gui.maven.modeler.run_fret_vbgmm_modelselection(nstates_min,nstates_max)
	logger.info('Finished FRET VB GMM + Model Selection')

def fxn_fret_mlgmm(gui):
	logger.info('Executing FRET ML GMM')
	nstates = gui.model_dialog.nstates.value()
	close_dialog(gui)
	gui.maven.modeler.run_fret_mlgmm(nstates)
	logger.info('Finished FRET ML GMM')

def fxn_fret_vbconhmm(gui):
	logger.info('Executing FRET VB Consensus HMM')
	nstates = gui.model_dialog.nstates.value()
	close_dialog(gui)
	gui.maven.modeler.run_fret_vbconhmm(nstates)
	logger.info('Finished FRET VB Consensus HMM')

def fxn_fret_vbconhmm_modelselection(gui):
	logger.info('Executing FRET VB Consensus HMM + Model Selection')
	nstates_min = gui.model_dialog.nstates_min.value()
	nstates_max = gui.model_dialog.nstates_max.value()
	if nstates_min > nstates_max:
		logger.info('Error: nstates min > max')
		return
	close_dialog(gui)
	gui.maven.modeler.run_fret_vbconhmm_modelselection(nstates_min,nstates_max)
	logger.info('Finished FRET VB Consensus HMM + Model Selection')

def fxn_fret_threshold_vbconhmm(gui):
	logger.info('Executing FRET VB Consensus HMM -> threshold')
	threshold = float(gui.model_dialog.threshold.text())
	nstates = gui.model_dialog.nstates.value()
	close_dialog(gui)
	gui.maven.modeler.run_fret_threshold_vbconhmm(nstates,threshold)
	logger.info('Finished FRET VB Consensus HMM->threshold')

def fxn_fret_vbhmm(gui):
	logger.info('Executing FRET VB HMM')
	nstates = gui.model_dialog.nstates.value()
	close_dialog(gui)
	gui.maven.modeler.run_fret_vbhmm(nstates)
	logger.info('Finished FRET VB HMM')

def fxn_fret_vbhmm_modelselection(gui):
	logger.info('Executing FRET VB HMM + Model Selection')
	nstates_min = gui.model_dialog.nstates_min.value()
	nstates_max = gui.model_dialog.nstates_max.value()
	if nstates_min > nstates_max:
		logger.info('Error: nstates min > max')
		return
	close_dialog(gui)
	gui.maven.modeler.run_fret_vbhmm_modelselection(nstates_min,nstates_max)
	logger.info('Finished FRET VB HMM + Model Selection')

def fxn_fret_kmeans_vbhmm(gui):
	logger.info('Executing FRET VB HMM->Kmeans')
	nstates = gui.model_dialog.nstates.value()
	close_dialog(gui)
	gui.maven.modeler.run_fret_kmeans_vbhmm(nstates)
	logger.info('Finished FRET VB HMM->Kmeans')

def fxn_fret_vbgmm_vbhmm_modelsection(gui):
	logger.info('Executing FRET VB HMM->VB GMM + Model Selection')
	nstates_min = gui.model_dialog.nstates_min.value()
	nstates_max = gui.model_dialog.nstates_max.value()
	close_dialog(gui)
	gui.maven.modeler.run_fret_vbgmm_vbhmm_modelselection(nstates_min,nstates_max)
	logger.info('Finished FRET VB HMM->VB GMM + Model Selection')

def fxn_fret_vbgmm_vbhmm(gui):
	logger.info('Executing FRET VB HMM->VB GMM')
	nstates = gui.model_dialog.nstates.value()
	close_dialog(gui)
	gui.maven.modeler.run_fret_vbgmm_vbhmm(nstates)
	logger.info('Finished FRET VB HMM->VB GMM')

def fxn_fret_threshold_vbhmm(gui):
	logger.info('Executing FRET VB HMM->threshold')
	threshold = float(gui.model_dialog.threshold.text())
	nstates = gui.model_dialog.nstates.value()
	close_dialog(gui)
	gui.maven.modeler.run_fret_threshold_vbhmm(nstates,threshold)
	logger.info('Finished FRET VB HMM->threshold')

def fxn_fret_vbhmm_one(gui):
	logger.info('Executing FRET VB HMM (One)')
	nstates = gui.model_dialog.nstates.value()
	close_dialog(gui)
	gui.maven.modeler.run_fret_vbhmm_one(nstates)
	logger.info('Finished FRET VB HMM (One)')

def fxn_fret_ebhmm(gui):
	logger.info('Executing FRET EB HMM')
	nstates = gui.model_dialog.nstates.value()
	close_dialog(gui)
	gui.maven.modeler.run_fret_ebhmm(nstates)
	logger.info('Finished FRET EB HMM')

def fxn_fret_ebhmm_modelselection(gui):
	logger.info('Executing FRET EB HMM + Model Selection')
	nstates_min = gui.model_dialog.nstates_min.value()
	nstates_max = gui.model_dialog.nstates_max.value()
	if nstates_min > nstates_max:
		logger.info('Error: nstates min > max')
		return
	close_dialog(gui)
	gui.maven.modeler.run_fret_ebhmm_modelselection(nstates_min,nstates_max)
	logger.info('Finished FRET EB HMM + Model Selection')
