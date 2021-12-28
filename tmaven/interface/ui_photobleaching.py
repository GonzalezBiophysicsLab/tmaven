def build_menu(gui):
	from PyQt5.QtWidgets import QMenu

	menu_photobleaching = QMenu('Photobleaching')
	menu_photobleaching.addAction('Photobleach Detection',gui.maven.photobleaching.photobleach_sum)
	from .stylesheet import ss_qmenu
	menu_photobleaching.setStyleSheet(ss_qmenu)

	return menu_photobleaching
