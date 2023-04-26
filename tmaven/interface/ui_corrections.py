import logging
logger = logging.getLogger(__name__)

def remove_beginning(gui):
	'''Remove datapoints from the beginning of data.raw

	This *permanently* removes datapoints from the time dimension from the beginning of data.raw (and consequentially data.corrected). `corrections.reset` will *not* undo this.

	'''
	from PyQt5.QtWidgets import QInputDialog
	nd,success = QInputDialog.getInt(gui,"Remove beginning","Number of frames to remove from beginnging",value=1,min=1,max=gui.maven.data.ntime-2)
	if success:
		success = gui.maven.corrections.remove_beginning(nd)
	if not success:
		logger.error('Correction failed removing frames from beginning')

def reset(gui):
	gui.maven.corrections.reset()

def build_menu(gui):
	## Qt
	from PyQt5.QtWidgets import QMenu
	menu = QMenu('Corrections',gui)
	menu_filter = QMenu('Signal Filters',gui)

	### Corrections
	menu.addMenu(menu_filter)
	corrections_reset = menu.addAction('Reset',lambda : reset(gui))
	corrections_remove = menu.addAction('Remove From Beginning',lambda : remove_beginning(gui))
	filter_gaussian = menu_filter.addAction('Gaussian',gui.maven.corrections.filter_gaussian)
	filter_wiener = menu_filter.addAction('Wiener',gui.maven.corrections.filter_wiener)
	filter_median = menu_filter.addAction('Median',gui.maven.corrections.filter_median)
	# filter_lds = self.menu_filter.addAction('Linear Dynamical System',self.filter_lds)
	filter_bessel = menu_filter.addAction('8-pole Bessel',gui.maven.corrections.filter_bessel)
	filter_chungkennedy = menu_filter.addAction('Chung-Kennedy',gui.maven.corrections.filter_chungkennedy)

	menu.addAction('Bleedthrough',gui.maven.corrections.bleedthrough)
	menu.addAction('Gamma',gui.maven.corrections.gamma)
	menu.addAction('Background',gui.maven.corrections.background_correct)


	# from .stylesheet import ss_qmenu
	# [m.setStyleSheet(ss_qmenu) for m in [menu,menu_filter]]
	return menu,menu_filter
