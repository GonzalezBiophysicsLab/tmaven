import numpy as np
import h5py as h
from PyQt5.QtCore import pyqtSignal

from PyQt5.QtCore import QAbstractItemModel, QModelIndex, Qt, QSize
from PyQt5.QtGui import QKeyEvent, QKeySequence
from PyQt5.QtWidgets import QApplication, QTreeView, QHBoxLayout, QWidget, QTextEdit,QFileDialog, QMenuBar,  QAction, QMessageBox,  QSizePolicy, QAbstractItemView

#### modified from the QT example by CKT

#############################################################################
##
## Copyright (C) 2013 Riverbank Computing Limited.
## Copyright (C) 2010 Nokia Corporation and/or its subsidiary(-ies).
## All rights reserved.
##
## This file is part of the examples of PyQt.
##
## $QT_BEGIN_LICENSE:BSD$
## You may use this file under the terms of the BSD license as follows:
##
## "Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are
## met:
##   * Redistributions of source code must retain the above copyright
##	 notice, this list of conditions and the following disclaimer.
##   * Redistributions in binary form must reproduce the above copyright
##	 notice, this list of conditions and the following disclaimer in
##	 the documentation and/or other materials provided with the
##	 distribution.
##   * Neither the name of Nokia Corporation and its Subsidiary(-ies) nor
##	 the names of its contributors may be used to endorse or promote
##	 products derived from this software without specific prior written
##	 permission.
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
## "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
## LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
## A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
## OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
## SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
## LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
## DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
## THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
## (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
## OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
## $QT_END_LICENSE$
##
#############################################################################

class _TreeItem(object):
	def __init__(self, data, parent=None):
		self.parentItem = parent
		self.itemData = data
		self.childItems = []

	def appendChild(self, item):
		self.childItems.append(item)

	def child(self, row):
		return self.childItems[row]

	def childCount(self):
		return len(self.childItems)

	def columnCount(self):
		return len(self.itemData)

	def data(self, column):
		try:
			return self.itemData[column]
		except IndexError:
			return None

	def parent(self):
		return self.parentItem

	def row(self):
		if self.parentItem:
			return self.parentItem.childItems.index(self)
		return 0

class _TreeModel(QAbstractItemModel):
	def __init__(self, filename, parent=None):
		super(_TreeModel, self).__init__(parent)
		self.filename= filename
		self.rootItem = _TreeItem(("group", "info"))
		f = h.File(self.filename,'r')
		self.setupModelData(f, self.rootItem)
		f.close()

	def columnCount(self, parent):
		if parent.isValid():
			return parent.internalPointer().columnCount()
		else:
			return self.rootItem.columnCount()

	def data(self, index, role):
		if not index.isValid():
			return None

		if role != Qt.DisplayRole:
			return None

		item = index.internalPointer()

		return item.data(index.column())

	def flags(self, index):
		if not index.isValid():
			return Qt.NoItemFlags

		if index.column() == 0:
			return Qt.ItemIsEnabled | Qt.ItemIsSelectable
		else:
			return Qt.NoItemFlags

	def headerData(self, section, orientation, role):
		if orientation == Qt.Horizontal and role == Qt.DisplayRole:
			return self.rootItem.data(section)

		return None

	def index(self, row, column, parent):
		if not self.hasIndex(row, column, parent):
			return QModelIndex()

		if not parent.isValid():
			parentItem = self.rootItem
		else:
			parentItem = parent.internalPointer()

		childItem = parentItem.child(row)
		if childItem:
			return self.createIndex(row, column, childItem)
		else:
			return QModelIndex()

	def parent(self, index):
		if not index.isValid():
			return QModelIndex()

		childItem = index.internalPointer()
		parentItem = childItem.parent()

		if parentItem == self.rootItem:
			return QModelIndex()

		return self.createIndex(parentItem.row(), 0, parentItem)

	def rowCount(self, parent):
		if parent.column() > 0:
			return 0

		if not parent.isValid():
			parentItem = self.rootItem
		else:
			parentItem = parent.internalPointer()

		return parentItem.childCount()

	def recursive_add(self,data,parent):
		for k in data:
			dk = data[k]
			if type(dk) is h._hl.group.Group:
				newchild = _TreeItem([dk.name,"","",""],parent)
			elif type(dk) is h._hl.dataset.Dataset:
				newchild = _TreeItem([dk.name,str(dk.shape)+" "+str(dk.dtype)],parent)
			for attr in dk.attrs:
				attrchild = _TreeItem(["# "+attr,dk.attrs[attr],"","",""],newchild)
				newchild.appendChild(attrchild)
			if type(dk) is h._hl.group.Group:
				self.recursive_add(dk,newchild)
			parent.appendChild(newchild)

	def setupModelData(self, data, parent):
		for attr in data.attrs:
			attrchild = _TreeItem(["# "+attr,data.attrs[attr],"","",""],parent)
			parent.appendChild(attrchild)
		self.recursive_add(data,parent)

class _hdf5_treeview(QTreeView):
	new_info = pyqtSignal()
	new_info_force = pyqtSignal()

	def __init__(self,*args,**kw_args):
		super(_hdf5_treeview,self).__init__(*args,**kw_args)
		self.header().setSectionResizeMode(3)
		self.header().setStretchLastSection(False)
		np.set_printoptions(threshold=np.inf)
		self.keys_enabled = True

	def setup(self,filename):
		self.filename = filename
		self.model = _TreeModel(self.filename)
		self.setModel(self.model)

	def get_selection(self):
		si = self.selectedIndexes()
		d = []
		for ss in si:
			sss = ss.sibling(ss.row(),1).data()
			d.append([self.model.data(ss,0),sss])

		return d

	def get_dataset(self,key):
		success = self.check_dataset(key)
		if success:
			with h.File(self.filename,'r') as f:
				out = f[key][:]
			return success,out
		return success,None



	# def get_entry(self,key):
	# 	if self.check_dataset(key):
	# 		return self.get_dataset(key)
	# 	elif self.check_group(key):
	# 		return True,key
	# 	else:
	# 		return False,None

	def check_dataset(self,key):
		success = False
		with h.File(self.filename,'r') as f:
			if key in f:
				if type(f[key]) is h._hl.dataset.Dataset:
					success = True
		return success

	def check_group(self,key):
		success = False
		with h.File(self.filename,'r') as f:
			if key in f:
				if type(f[key]) is h._hl.group.Group:
					success = True
		return success

	def select(self,signal):
		try:
			d = self.get_selection()
			if d[0][0][0] == '#':
				self.current_info = d[0][1]
				signal.emit()
			else:
				success,self.current_info = self.get_dataset(d[0][0])
				if success:
					signal.emit()

		except:
			pass

	def keyPressEvent(self,key):
		super(_hdf5_treeview,self).keyPressEvent(key)
		if self.keys_enabled and key.key() in [Qt.Key_Return,Qt.Key_Enter,Qt.Key_Space]:
			self.select(self.new_info_force)
		else:
			self.select(self.new_info)


class hdf5_view_primative(QWidget):
	''' QWidget base for viewing HDF5 files as QModels

	Attributes
	----------
	treeview : _hdf5_treeview
		* treeview.model = _TreeModel
		* treeview.setup(filename)
			* load hdf5 file into model

	'''
	def __init__(self,filename=None,*args,**kw_args):
		super(hdf5_view_primative,self).__init__(*args,**kw_args)

		self.treeview = _hdf5_treeview()
		# self.treeview.setStyleSheet(ss_treeview)
		# self.treeview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

		from PyQt5.QtWidgets import QSplitter
		self.splitter = QSplitter()
		self.splitter.addWidget(self.treeview)
		# self.splitter.setStyleSheet(ss_splitter)
		self.hbox = QHBoxLayout()
		self.hbox.addWidget(self.splitter)
		self.setLayout(self.hbox)
		# self.setStyleSheet(ss_hdf5)

		if not filename is None:
			self.load_hdf5(filename)

	def load_hdf5(self,filename):
		''' Load file

		Parameters
		----------
		filename : str
			filename of hdf5 file to load
		'''
		self.filename = filename
		self.treeview.setup(self.filename)


#
# pad = 0
# ss_box = '''
# margin-left:%d;margin-right:%d;margin-top:%d;margin-bottom:%d;
# '''%((pad,)*4)
# ss_button = '''QPushButton {%s}'''%(ss_box)

#
#
# if __name__ == '__main__':
# 	# ## Create fake data
# 	# filename = 'test2.hdf5'
# 	# f = h.File(filename,'w')
# 	# f.create_group('g1')
# 	# f['g1'].create_group('h1')
# 	# f['g1'].attrs['description'] = 'asdf'
# 	# f['g1'].create_group('h2')
# 	# f['g1/h2'].create_group('i1')
# 	# f['g1/h2'].attrs['description'] = 'asdf'
# 	# f['g1/h1'].attrs['description'] = 'asdf'
# 	# d = f.create_dataset('tester',data=np.random.rand(100,100,2))
# 	# d.attrs['desc'] = 'fun'
# 	# d.attrs['dim desc'] = 'trace,color,time'
# 	# f.create_group('g2')
# 	# f['g2'].create_group('h1')
# 	# f.create_group('g3')
# 	# f.close()
#
# 	## View data
#
# 	import sys
# 	app = QApplication(sys.argv)
#
# 	# view = hdf5_view()
#
# 	# view = hdf5_datasetgroup_load('./rf1.hdf5')
# 	# view.new_dataset.connect(lambda x,y: print('dataset',x,y))
# 	# view.new_group.connect(lambda x,y: print('group',x,y))
#
# 	# view = hdf5_view_primative()
# 	# view.load_hdf5('./rf1.hdf5')
#
# 	view = hdf5_fileview()
#
# 	# l = view.layout().itemAtPosition(0,2).layout()
#
# 	# print(l.itemAt(l.corunt()-1).widget()
# 	# n = l.count()
# 	# for i in range(n):
# 	# 	print(l.itemAt(i).widget())
#
#
#
#
# 	view.show()
# 	sys.exit(app.exec_())
