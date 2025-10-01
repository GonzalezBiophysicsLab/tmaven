def build_menu(gui):
	from PyQt5.QtWidgets import QMenu

	menu_photobleaching = QMenu('Photobleaching',gui)
	menu_photobleaching.addAction('Photobleach Detection',gui.maven.photobleaching.photobleach_sum)
	menu_photobleaching.addAction('Reset Photobleaching',gui.maven.photobleaching.remove_photobleaching)
	# from .stylesheet import ss_qmenu
	# menu_photobleaching.setStyleSheet(ss_qmenu)

	return menu_photobleaching
