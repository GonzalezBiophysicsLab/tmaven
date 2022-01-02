import logging
logger = logging.getLogger(__name__)
import numpy as np

def toggle_selection(gui):
	if gui.molecules_viewer.viewer is None:
		return
	v = gui.molecules_viewer.viewer
	c = gui.molecules_viewer.viewer.model

	sm = v.selectionModel()
	v.blockSignals(True)
	sm.blockSignals(True)

	rows_on = [r.row() for r in sm.selectedRows()]
	rc = c.rowCount(0)
	rows_off = [i for i in range(rc) if not i in rows_on]
	inds = [c.index(r, 0) for r in rows_off]

	from PyQt5.QtCore import QItemSelectionModel
	mode = QItemSelectionModel.Select | QItemSelectionModel.Rows
	v.clearSelection()
	[sm.select(i, mode) for i in inds]

	sm.blockSignals(False)
	v.blockSignals(False)
	c.layoutChanged.emit()

def show_counts(gui):
	from PyQt5.QtWidgets import QMessageBox
	s = gui.maven.selection.get_class_counts()
	QMessageBox.information(gui,"Class Counts",s)

def select_all(gui):
	if gui.molecules_viewer.viewer is None:
		return
	gui.molecules_viewer.viewer.selectAll()
def select_none(gui):
	if gui.molecules_viewer.viewer is None:
		return
	gui.molecules_viewer.viewer.clearSelection()

def select_class(gui):
	if gui.molecules_viewer.viewer is None:
		return
	from PyQt5.QtWidgets import QInputDialog
	from PyQt5.QtCore import QItemSelectionModel
	nt,success = QInputDialog.getInt(gui,"Select class","Select all molecules in what class", value=0,min=0)
	if success:
		keep = gui.maven.selection.select_class(nt)

		v = gui.molecules_viewer.viewer
		sm = v.selectionModel()

		v.blockSignals(True)
		sm.blockSignals(True)

		mode = QItemSelectionModel.Select | QItemSelectionModel.Rows
		v.clearSelection()
		[sm.select(v.model.index(i,0), mode) for i in keep]

		v.blockSignals(False)
		sm.blockSignals(False)
		v.model.layoutChanged.emit()

def set_class_from_selection(gui):
	if gui.molecules_viewer.viewer is None:
		return
	from PyQt5.QtWidgets import QInputDialog
	nt,success = QInputDialog.getInt(gui,"Set class","Set selected molecules into which class", value=0,min=0)
	if success:
		sel = gui.molecules_viewer.viewer.get_selection()
		print(sel)
		if sel is None:
			return
		keep = np.isin(np.arange(gui.maven.data.classes.size),sel)
		gui.maven.data.classes[keep] = nt
		gui.maven.emit_data_update()


def _class_onoff(gui,s,b):
	if gui.molecules_viewer.viewer is None:
		return
	from PyQt5.QtWidgets import QInputDialog
	nt,success = QInputDialog.getInt(gui,"Turn {} class".format(s),"Turn {} all molecules in what class".format(s), value=0,min=0)
	if success:
		keep = gui.maven.selection.select_class(nt)
		gui.maven.data.flag_ons[keep] = b
		gui.maven.emit_data_update()
		logger.info('Turned {} class {}'.format(s,nt))
	else:
		logger.error('Failed turning {} classes {}'.format(s,nt))

def _all_onoff(gui,s,b):
	if gui.molecules_viewer.viewer is None:
		return
	from PyQt5.QtWidgets import QMessageBox
	reply = QMessageBox.question(gui,"Turn {}?".format(s),"Are you sure you want to turn {} all molcules?".format(s),QMessageBox.Yes | QMessageBox.No)
	if reply == QMessageBox.Yes:
		gui.maven.data.flag_ons = np.ones(gui.maven.data.nmol,dtype='bool')*b
		gui.maven.emit_data_update()
		logger.info('Turned {} all molecules'.format(s))

def _selection_onoff(gui,s,b):
	if gui.molecules_viewer.viewer is None:
		return
	sel = gui.molecules_viewer.viewer.get_selection()
	if sel is None: return
	from PyQt5.QtWidgets import QMessageBox
	reply = QMessageBox.question(gui,"Turn {}?".format(s),"Are you sure you want to turn {} {} molcules?".format(s,sel.size),QMessageBox.Yes | QMessageBox.No)
	if reply == QMessageBox.Yes:
		gui.maven.data.flag_ons[sel] = np.ones(sel.size,dtype='bool')*b
		gui.maven.emit_data_update()
		logger.info('Turned {} {} molecules'.format(s,sel.size))

def class_on(gui):
	_class_onoff(gui,'on',True)
def class_off(gui):
	_class_onoff(gui,'off',False)
def all_on(gui):
	_all_onoff(gui,'on',True)
def all_off(gui):
	_all_onoff(gui,'off',False)
def selection_on(gui):
	_selection_onoff(gui,'on',True)
def selection_off(gui):
	_selection_onoff(gui,'off',False)

def build_menu(gui):
	from PyQt5.QtWidgets import QMenu
	menu_selection = QMenu('Selection',gui)
	menu_on = QMenu('Turn On',gui)
	menu_off = QMenu('Turn Off',gui)

	menu_on.addAction('All',lambda : all_on(gui))
	menu_on.addAction('Selection',lambda : selection_on(gui))
	menu_on.addAction('Class',lambda : class_on(gui))
	menu_off.addAction('All',lambda : all_off(gui))
	menu_off.addAction('Selection',lambda : selection_off(gui))
	menu_off.addAction('Class',lambda : class_off(gui))
	menu_selection.addAction('Select All',lambda : select_all(gui))
	menu_selection.addAction('Select None',lambda : select_none(gui))
	menu_selection.addAction('Select Class',lambda : select_class(gui))
	menu_selection.addAction('Toggle Selection',lambda : toggle_selection(gui))
	menu_selection.addSeparator()
	menu_selection.addAction('Display Class Counts',lambda : show_counts(gui))
	menu_selection.addSeparator()
	menu_selection.addAction('Set Class',lambda : set_class_from_selection(gui))
	menu_selection.addAction('Order by CC',gui.maven.selection.order_fret_cross_corr)
	menu_selection.addMenu(menu_on)
	menu_selection.addMenu(menu_off)

	# from .stylesheet import ss_qmenu
	# for m in [menu_selection,menu_on,menu_off]:
		# m.setStyleSheet(ss_qmenu)

	return menu_selection,menu_on,menu_off
