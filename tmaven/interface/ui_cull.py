from PyQt5.QtWidgets import QInputDialog, QMenu, QAction

def build_menu(gui):
	menu = QMenu('Cull')
	menu.addAction('Cull class', lambda: cull_class(gui))
	menu.addAction('Cull minimums', lambda: cull_min(gui))
	menu.addAction('Cull maximums', lambda: cull_max(gui))
	menu.addAction('Cull short', lambda: cull_short(gui))
	from ..interface.stylesheet import ss_qmenu
	menu.setStyleSheet(ss_qmenu)
	return menu

def cull_min(gui):
	threshold,success = QInputDialog.getDouble(gui,"Cull Minimum","Removes traces with at least one datapoint less than what value?", value=-10000.,decimals=6)
	if success:
		gui.maven.cull.cull_min(threshold)

def cull_max(gui):
	threshold,success = QInputDialog.getDouble(gui,"Cull Maximum","Removes traces with at least one datapoint greater than what value?", value=65535.,decimals=6)
	if success:
		gui.maven.cull.cull_max(threshold)

def cull_class(gui):
	c,success = QInputDialog.getInt(gui,"Remove Class","Which class to remove?", value=0,min=0)
	if success:
		gui.maven.cull.cull_class(c)

def cull_short(gui):
	nt,success = QInputDialog.getInt(gui,"Cull Shorter","Removes traces less than what length?", value=10,min=1,max=gui.maven.data.ntime-2)
	if success:
		gui.maven.cull.cull_short(nt)
