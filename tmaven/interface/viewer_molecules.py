from PyQt5.QtWidgets import QTableView, QMenu, QSizePolicy, QDockWidget, QAbstractItemView
from PyQt5.QtCore import Qt, QModelIndex, QAbstractTableModel, QVariant, QObject,QItemSelectionModel
from PyQt5.QtGui import QKeySequence
import numpy as np
import logging
logger = logging.getLogger(__name__)

class_keys = {Qt.Key_1:1,Qt.Key_2:2,Qt.Key_3:3,Qt.Key_4:4,Qt.Key_5:5,Qt.Key_6:6,Qt.Key_7:7,Qt.Key_8:8,Qt.Key_9:9,Qt.Key_0:0}

class molecules_viewer(QObject):
	def __init__(self,gui):
		super().__init__(parent=gui)
		self.gui = gui
		self.viewer = molecules_widget(self.gui)
		self.gui.data_update.connect(self.viewer.model.layoutChanged.emit)
		self.dock = QDockWidget("Molecule Table",self.gui)

		# from .stylesheet import ss_qdockwidget
		# self.dock.setStyleSheet(ss_qdockwidget)
		self.dock.setWidget(self.viewer)
		self.dock.setFeatures(QDockWidget.AllDockWidgetFeatures)
		self.dock.setAllowedAreas(Qt.TopDockWidgetArea | Qt.BottomDockWidgetArea | Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
		self.gui.addDockWidget(Qt.LeftDockWidgetArea, self.dock)
		self.dock.hide()
		# self.dock.closeEvent = lambda e: self.toggle()

	def update_theme(self,palette):
		self.dock.setPalette(palette)
		self.viewer.setPalette(palette)
		self.viewer.horizontalHeader().setPalette(palette)
		self.viewer.verticalHeader().setPalette(palette)
		self.viewer.verticalScrollBar().setPalette(palette)
		self.viewer.horizontalScrollBar().setPalette(palette)
		for c in self.viewer.children():
			try: c.setPalette(palette)
			except: pass

	def toggle(self):
		if self.dock.isHidden():
			self.dock.show()
			if not self.gui.preferences_viewer.dock.isHidden():
				self.gui.tabifyDockWidget(self.gui.preferences_viewer.dock,self.dock)
		self.dock.raise_()

	def update(self):
		if not self.viewer is None:
			self.viewer.model.dataChanged.emit(QModelIndex(),QModelIndex())


class molecules_model(QAbstractTableModel):
	'''
	'''

	def __init__(self, gui):
		super().__init__()
		self.gui = gui

	def columnCount(self, index):
		return 6

	def rowCount(self,index):
		return self.gui.maven.data.nmol

	def headerData(self,section,orientation,role):
		if role == Qt.DisplayRole:
			if orientation == Qt.Horizontal:
				return ['Class','On/Off','Pre','Post','Data\nindex','Source\nindex'][section]
			return section
		elif role == Qt.TextAlignmentRole:
			if orientation == Qt.Horizontal:
				if section == 4 or section == 5:
					return Qt.AlignVCenter #| Qt.AlignLeft
				else:
					return Qt.AlignCenter
			if orientation == Qt.Vertical:
				return Qt.AlignRight
		return QVariant()

	def data(self,index,role):
		if not index.isValid():
			return QVariant()

		if role == Qt.TextAlignmentRole:
			if index.column() == 4 or index.column() == 5:
				return Qt.AlignVCenter #| Qt.AlignLeft
			else:
				return Qt.AlignCenter
			return

		column = index.column()
		row = index.row()

		# if column == 0:
		# 	if role == Qt.DisplayRole:
		# 		return str(row)

		if column == 0:
			if role == Qt.DisplayRole or role == Qt.EditRole:
				return str(self.gui.maven.data.classes[row])
			elif role == Qt.InitialSortOrderRole:
				return QVariant(int(self.gui.maven.data.classes[row]))

		if column == 1:
			if role == Qt.DisplayRole:
				return QVariant() # str(self.flag_ons[row])
			elif role == Qt.CheckStateRole:
				if self.gui.maven.data.flag_ons[row]:
					return Qt.Checked
				else:
					return Qt.Unchecked
				# return QVariant(bool(self.flag_ons[row]))
			# elif role == Qt.CheckStateRole:
				# return QVariant(bool(self.flag_ons[row]))
			return QVariant()

		if column == 4:
			if role == Qt.DisplayRole:
				return str(self.gui.maven.data.data_index[row])
		if column == 5:
			if role == Qt.DisplayRole:
				return str(self.gui.maven.smd.source_index[row])
				# return str(self.gui.maven.smd.source_names[self.gui.maven.smd.source_index[row]])

		if column == 2:
			if role == Qt.DisplayRole:
				return str(self.gui.maven.data.pre_list[row])
		if column == 3:
			if role == Qt.DisplayRole:
				return str(self.gui.maven.data.post_list[row])

		return QVariant()

	def setData(self,index,value,role):
		if not index.isValid():
			return False

		column = index.column()
		row = index.row()

		if column == 0:
			if _is_int(value):
				v = int(value)
				if v >= 0:
					self.gui.maven.data.classes[row] = v
					self.dataChanged.emit(index,index)
					return True
		if column == 1:
			if role == Qt.CheckStateRole:
				self.gui.maven.data.flag_ons[index.row()] = bool(value)
				self.dataChanged.emit(index,index)
				return True
		return False

	def flags(self,index):
		####Qt.ItemIsSelectable | Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsUserCheckable
		if not index.isValid():
			return Qt.NoItemFlags

		# if index.column() == 0:
		# 	return Qt.ItemIsEnabled | Qt.ItemIsSelectable
		if index.column() == 0:
			return Qt.ItemIsEnabled | Qt.ItemIsEditable | Qt.ItemIsSelectable
		elif index.column() == 1:
			return Qt.ItemIsUserCheckable | Qt.ItemIsEnabled  | Qt.ItemIsSelectable
		elif index.column() == 4:
			return Qt.ItemIsSelectable | Qt.ItemIsEnabled
		elif index.column() == 5:
			return Qt.ItemIsSelectable | Qt.ItemIsEnabled
		elif index.column() == 2:
			return Qt.ItemIsSelectable | Qt.ItemIsEnabled
		elif index.column() == 3:
			return Qt.ItemIsSelectable | Qt.ItemIsEnabled
		return Qt.NoItemFlags

	# def sort(self,column,role):
	# 	if not column in [0,1,2,3]:
	# 		return
	#
	# 	if column == 0:
	# 		order = np.arange(self.gui.maven.data.nmol)
	# 	elif column == 1:
	# 		order = np.argsort(self.gui.maven.data.classes,kind='stable')
	# 	elif column == 2:
	# 		order = np.argsort(self.gui.maven.data.flag_ons,kind='stable')
	# 	elif column == 3:
	# 		order = np.argsort(self.gui.maven.smd.source_names,kind='stable')
	# 	elif column == 4:
	# 		order = np.argsort(self.gui.maven.data.pre_list,kind='stable')
	# 	elif column == 5:
	# 		order = np.argsort(self.gui.maven.data.post_list,kind='stable')
	#
	# 	# if role == Qt.AscendingOrder:
	# 	# 	order = order[::-1]
	# 	self.custom_sort(order)
	#
	# def custom_sort(self,order):
	# 	if order.size == 0:
	# 		return
	# 	self.indices = self.indices[order]
	# 	self.classes = self.classes[order]
	# 	self.flag_ons = self.flag_ons[order]
	# 	self. = self.[order]
	# 	self.data_index = self.data_index[order]
	# 	self.layoutChanged.emit()
	#
	# def get_molecule(self,row):
		# return [self.indices[row],self.classes[row],self.flag_ons[row],self.[row],self.data_index[row]]
	# def update_row(self,row):
	# 	self.dataChanged.emit(QModelIndex(0,row),QModelIndex(5,row))

def _is_int(x):
	try:
		int(x)
		return True
	except:
		return False


class molecules_widget(QTableView):
	def __init__(self,gui):
		super().__init__()
		self.gui = gui
		self.undo = []
		self.redo = []
		# self.wheelEvent = super().wheelEvent

		# from .stylesheet import ss_qtableview
		# self.setStyleSheet(ss_qtableview)
		#
		self.model = molecules_model(self.gui)
		super().setModel(self.model)

		self.setSortingEnabled(False)
		self.horizontalHeader().setStretchLastSection(True)
		self.setSelectionBehavior(QTableView.SelectRows)
		# self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
		# self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
		self.setSizePolicy(QSizePolicy.MinimumExpanding,QSizePolicy.MinimumExpanding)
		# self.resizeColumnsToContents()
		for i in range(self.model.columnCount(0)):
			self.setColumnWidth(i,40)

	def get_selection(self):
		all = np.array([])
		all_inds = self.selectedIndexes()
		if len(all_inds) > 0:
			all = np.unique(np.array([ind.row() for ind in all_inds]))
			return all
		return None

	def keyPressEvent(self,event):
		if event.key() in class_keys:
			new_class_ind = class_keys[event.key()]
			sel = self.get_selection()
			if not sel is None:
				self.undo.append(self.gui.maven.data.classes.copy())
				for i in sel:
					self.gui.maven.data.classes[i] = new_class_ind
			self.model.dataChanged.emit(QModelIndex(),QModelIndex())
		# elif event.key() == Qt.Key_Tab:
		# 	sel = self.get_selection()
		# 	if not sel is None:
		# 		for i in sel:
		# 			self.gui.maven.data.flag_ons[i] = not self.gui.maven.data.flag_ons[i]
		# 	self.model.dataChanged.emit(QModelIndex(),QModelIndex())
		elif event == QKeySequence.Undo:
			self.undo()
			self.defocus()
		elif event == QKeySequence.Redo:
			self.redo()
			self.defocus()
		elif event.key() == Qt.Key_Escape:
			self.defocus()
		else:
			super().keyPressEvent(event)

	def defocus(self):
		try:
			self.parent().parent().setFocus()
		except:
			pass

	def undo(self):
		if len(self.undo) > 0:
			old = self.undo.pop()
			self.redo.append(self.gui.maven.data.classes.copy())
			self.gui.maven.data.classes = old
			self.model.dataChanged.emit(QModelIndex(),QModelIndex())
			logger.info('undo classes change')

	def redo(self):
		if len(self.redo) > 0:
			old = self.redo.pop()
			self.undo.append(self.gui.maven.data.classes.copy())
			self.gui.maven.data.classes = old
			self.model.dataChanged.emit(QModelIndex(),QModelIndex())
			logger.info('redo classes change')

	def _toggle_viewer(self):
		self.setVisible(not self.isVisible())

		if not self.isVisible():
			self.gui.vbox.insertStretch(0)
		else:
			self.gui.vbox.removeItem(self.gui.vbox.itemAt(0))
