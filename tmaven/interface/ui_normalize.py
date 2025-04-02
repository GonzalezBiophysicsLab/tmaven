import logging
logger = logging.getLogger(__name__)


def reset(gui):
	gui.maven.normalize.reset()

def build_menu(gui):
	## Qt
	from PyQt5.QtWidgets import QMenu
	menu = QMenu('Normalize',gui)
	#menu_filter = QMenu('Signal Filters',gui)

	### Corrections
	#menu.addMenu(menu_filter)
	menu.addAction('Reset',lambda : reset(gui))
	menu.addAction('Individual Min-Max', gui.maven.normalize.normalize_minmax_ind)


	# from .stylesheet import ss_qmenu
	# [m.setStyleSheet(ss_qmenu) for m in [menu,menu_filter]]
	return menu
