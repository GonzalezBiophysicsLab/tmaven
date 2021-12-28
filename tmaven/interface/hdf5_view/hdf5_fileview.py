from PyQt5.QtWidgets import QFileDialog,QPushButton,QInputDialog,QMessageBox
from .hdf5_primative import hdf5_view_primative

class hdf5_fileview(QFileDialog):
	''' Modified QFileDialog to preview contents of selected HDF5 file
	'''
	def __init__(self,*args,**kw_args):
		super(QFileDialog,self).__init__(*args,**kw_args)

		self.setOption(QFileDialog.DontUseNativeDialog)

		self.viewer = hdf5_view_primative()
		self.button_add_group = QPushButton('+Group')
		self.button_add_group.setStyleSheet('''QPushButton {border-radius: 0px;border: 1px solid rgb(100, 100, 100);padding: 2px 2px 2px 2px; background: white;}''')
		self.button_add_group.clicked.connect(self.add_group)
		self.button_add_group.setEnabled(False)
		self.fname = None

		layout = self.layout()
		splitter = layout.itemAtPosition(1,0).widget()
		splitter.addWidget(self.viewer.treeview)
		self.viewer.treeview.hide()

		l = self.layout().itemAtPosition(0,2).layout()
		l.addWidget(self.button_add_group)

		self.currentChanged.connect(self.try_new_file)
		self.update()

	def add_group(self):
		''' add new HDF5 group to current file '''
		if not self.fname is None:
			gname = QInputDialog.getText(self,'New Group Name','Add New HDF5 group name',text='')[0]
			if not gname == '':
				try:
					with h.File(self.fname,'r+') as f:
						f.create_group(gname)
						f.close()
					self.viewer.load_hdf5(self.fname)
				except:
					QMessageBox.information(self,"Failed",'Failed to create HDF5 group %s in file %s'%(gname,fname))

	def try_new_file(self,fname):
		if fname.endswith('.hdf5') or fname.endswith('.hdf'):
			self.fname = fname
			self.viewer.treeview.show()
			self.viewer.load_hdf5(fname)
			self.button_add_group.setEnabled(True)

		else:
			self.fname = None
			self.viewer.treeview.hide()
			self.button_add_group.setEnabled(False)
			# self.buttons.hide()
