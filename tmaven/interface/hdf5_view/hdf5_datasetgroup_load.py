from PyQt5.QtWidgets import QMenuBar,QAction,QTextEdit,QMessageBox,QFileDialog
from PyQt5.QtCore import pyqtSignal, Qt

from .hdf5_primative import hdf5_view_primative

class hdf5_datasetgroup_load(hdf5_view_primative):
	''' Select an entry in HDF5 file

	This provides an interface to select a group and/or a dataset from an HDF5 file. Useful for, for instance, loading SMD/HDF5 files where you need to choose which group is the base dataset to load in.

	Parameters
	----------
	filename : str
		Name of HDF5 file to load
	allowed : str
		* 'both' - can load both dataset and group
		* 'group' - can only load groups
		* 'dataset' - can only load dataset

	Attributes
	----------
	treeview : _hdf5_treeview
		the model
	menubar : QMenuBar
		* Select - use the currently selected entry
		* Close - close the dialog
	new_dataset.emit : pyqtSignal(str,str)
		emits a signal when a dataset is selected
	new_group.emit : pyqtSignal(str,str)
		emits a signal when a group is selected

	Notes
	-----
	have to connect to `self.new_dataset` or `self.new_group` in order to catch a successful selection. Closing window or `make_selection` do not return anything.

	'''
	new_dataset = pyqtSignal(str,str)
	new_group = pyqtSignal(str,str)

	def __init__(self,filename=None,allowed='both',*args,**kw_args):
		super(hdf5_datasetgroup_load,self).__init__(filename,*args,**kw_args)

		self.allowed = allowed
		self.treeview.keys_enabled = False
		self.menubar = QMenuBar(self)

		action_load = QAction('Select',self)
		action_load.triggered.connect(lambda event: self.make_selection())
		self.menubar.addAction(action_load)

		action_close = QAction('Close',self)
		action_close.triggered.connect(self.cancel)
		self.menubar.addAction(action_close)

		self.menubar.setNativeMenuBar(False)
		self.layout().setMenuBar(self.menubar)

	def make_selection(self):
		''' Choose current selection

		If allowed and a group, emits `self.new_group(file name, HDF5 entry name)`.
		If allowed and a dataset, emits `self.new_dataset(file name, HDF5 entry name)`

		'''
		try:
			key = self.treeview.get_selection()[0][0]
			if not key.startswith('#'):
				if self.allowed == 'both' or self.allowed == 'group':
					if self.treeview.check_group(key):
						self.new_group.emit(self.filename,key)
						self.close()
				if self.allowed == 'both' or self.allowed == 'dataset':
					if self.treeview.check_dataset(key):
						self.new_dataset.emit(self.filename,key)
						self.close()
		except:
			pass

	def cancel(self):
		self.close()

	def keyPressEvent(self,event):
		if event.key() in [Qt.Key_Enter,Qt.Key_Return]:
			self.make_selection()
		else:
			super(hdf5_datasetgroup_load,self).keyPressEvent(event)

	def closeEvent(self,event):
		try:
			self.disconnect()
			del self.treeview.current_info
		except:
			pass
		event.accept()
