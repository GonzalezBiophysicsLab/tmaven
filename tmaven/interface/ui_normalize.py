import logging
logger = logging.getLogger(__name__)


def reset(gui):
	gui.maven.normalize.reset()

def build_menu(gui):
	## Qt
	from PyQt5.QtWidgets import QMenu
	menu = QMenu('Normalize',gui)
	menu_ind = QMenu('Individual',gui)
	menu_glob = QMenu('Global',gui)

	### Corrections
	#menu.addMenu(menu_filter)
	menu.addAction('Reset',lambda : reset(gui))

	menu_ind.addAction('Min-Max', gui.maven.normalize.normalize_minmax_ind)
	menu_ind.addAction('CK Filter Min-Max', gui.maven.normalize.normalize_minmax_ckfilt_ind)

	menu.addMenu(menu_ind)

	menu_glob.addAction('Min-Max', gui.maven.normalize.normalize_minmax_glob)

	menu.addMenu(menu_glob)

	# from .stylesheet import ss_qmenu
	# [m.setStyleSheet(ss_qmenu) for m in [menu,menu_filter]]
	return menu
