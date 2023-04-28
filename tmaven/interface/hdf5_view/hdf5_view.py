from PyQt5.QtWidgets import QMenuBar,QAction,QTextEdit,QMessageBox,QFileDialog
from PyQt5.QtCore import QObject
import numpy as np

from .hdf5_primative import hdf5_view_primative

class hdf5_view(hdf5_view_primative):
	''' Interface for exploring HDF5 files using hdf5_view_primative

	Paramters
	---------
	filename : None or str
		Can start with file loaded

	Attributes
	----------
	menubar : QMenu
		* Load HDF5 - self.select_file
		* Export Seleted - self.export_array
		* Close - self.closeEvent


	'''
	def __init__(self,filename=None,*args,**kw_args):
		super(hdf5_view,self).__init__(filename,*args,**kw_args)

		self.menubar = QMenuBar(self)
		action_load = QAction('Load HDF5',self)
		action_load.triggered.connect(self.select_file)
		self.menubar.addAction(action_load)
		action_export = QAction('Export selected',self)
		action_export.triggered.connect(self.export_array)
		self.menubar.addAction(action_export)
		action_close = QAction('Close',self)
		action_close.triggered.connect(self.close)
		self.menubar.addAction(action_close)
		self.menubar.setNativeMenuBar(False)
		self.layout().setMenuBar(self.menubar)

		self.textedit = QTextEdit()
		self.textedit.setText('')
		self.splitter.addWidget(self.textedit)
		self.treeview.new_info.connect(lambda : self.update_textedit(False))
		self.treeview.new_info_force.connect(lambda : self.update_textedit(True))

	def sizeHint(self):
		from PyQt5.QtCore import QSize
		return QSize(800,400)

	def update_textedit(self,force=False):
		'''Update displayed data to current entry'''
		try:
			if type(self.treeview.current_info) is np.ndarray:
				np.set_printoptions(threshold=1000)
				if force:
					np.set_printoptions(threshold=np.inf)
					if self.treeview.current_info.size > 1e4:
						if not (QMessageBox.Yes == QMessageBox(QMessageBox.Information, "Lots of data", "This is a very large array. Do you want to display it all?", QMessageBox.Yes|QMessageBox.No).exec()):
							np.set_printoptions(threshold=1000)
			self.textedit.setText(str(self.treeview.current_info))
		except:
			self.textedit.setText('')

	def export_array(self):
		''' Export data from selected `treeview` entry

		Uses numpy if np.ndarray or writes a text file otherwise
		'''
		success = False
		stxt = False
		snpy = False
		try:
			if type(self.treeview.current_info) is np.ndarray:
				success = True
				snpy = True
				stx
			elif type(self.treeview.current_info) is str:
				success = True
				stxt = True
		except:
			pass

		if success:
			if snpy:
				fname = QFileDialog.getSaveFileName(self,'Export data (.npy)','.npy','*.npy')[0]
				if fname != "":
					try:
						np.save(fname,self.treeview.current_info)
					except:
						print('failed saving %s'%(fname))
			elif stxt:
				fname = QFileDialog.getSaveFileName(self,'Export data (.txt)','.txt','*.txt')[0]
				if fname != "":
					try:
						with open(fname,'w') as f:
							f.write(self.treeview.current_info)
					except:
						print('failed saving %s'%(fname))

	def select_file(self):
		''' Select HDF5 file to open '''
		success = False
		fname = QFileDialog.getOpenFileName(self,'Choose HDF5 file to load','./')[0]
		if not fname == "":
			success = True
		if success:
			try:
				self.load_hdf5(fname)
				self.textedit.setText('')
			except:
				print('Could not load %s'%(fname))

class hdf5_view_container(QObject):
	def __init__(self,gui):
		super(hdf5_view_container,self).__init__(parent=gui)
		self.gui = gui

		self.action = QAction('HDF5 Viewer',self.gui)
		self.action.triggered.connect(self.launch)

	def launch(self,event=None):
		try:
			del self.hdf5_view
		except:
			pass

		self.hdf5_view = hdf5_view()
		from ..stylesheet import ui_stylesheet
		self.hdf5_view.setStyleSheet(ui_stylesheet)
		self.hdf5_view.show()
