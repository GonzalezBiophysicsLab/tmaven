from PyQt5.QtWidgets import QWidget, QLineEdit, QTableView, QHeaderView, QVBoxLayout,QFileDialog, QDockWidget, QFileDialog
from PyQt5.QtCore import  QObject, Qt, QSortFilterProxyModel, pyqtSignal, QRegExp, QAbstractTableModel, QVariant, QModelIndex

import numpy as np
array = np.array

class preferences_viewer(QObject):
	def __init__(self,gui):
		super().__init__()
		self.gui = gui
		self.viewer = prefs_widget(parent=self.gui)
		self.prefs_model = pref_model(self.gui.maven)
		self.viewer.set_model(self.prefs_model)

		self.dock = QDockWidget("Preferences",self.gui)
		# from .stylesheet import ss_qdockwidget,ss_qtableview
		# self.dock.setStyleSheet(ss_qdockwidget)
		# self.viewer.setStyleSheet(ss_qtableview)
		self.dock.setWidget(self.viewer)
		self.dock.setFeatures(QDockWidget.AllDockWidgetFeatures)
		self.dock.setAllowedAreas(Qt.TopDockWidgetArea | Qt.BottomDockWidgetArea | Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
		self.gui.addDockWidget(Qt.LeftDockWidgetArea, self.dock)
		self.dock.hide()
		# self.dock.closeEvent = lambda e: self.toggle()

	def update_theme(self,palette):
		self.viewer.setPalette(palette)
		self.dock.setPalette(palette)
		self.viewer.le_filter.setPalette(palette)
		self.viewer.proxy_view.setPalette(palette)
		self.viewer.proxy_view.horizontalHeader().setPalette(palette)
		self.viewer.proxy_view.verticalHeader().setPalette(palette)
		self.viewer.proxy_view.verticalScrollBar().setPalette(palette)
		self.viewer.proxy_view.horizontalScrollBar().setPalette(palette)

	def toggle(self):
		if self.dock.isHidden():
			self.dock.show()
			if not self.gui.molecules_viewer.dock.isHidden():
				self.gui.tabifyDockWidget(self.gui.molecules_viewer.dock,self.dock)
		self.dock.raise_()

	def save(self,event=None):
		fname,_ = QFileDialog.getSaveFileName(None,'Save preferences text file','./prefs.txt','.txt')
		self.gui.maven.prefs.save(fname=fname)

	def load(self,event=None):
		fname,_ = QFileDialog.getOpenFileName(None,'Choose preferences file to load','./prefs.txt')
		self.gui.maven.prefs.emit_changed = lambda : None
		self.gui.maven.prefs.load(fname=fname,quiet=True)
		self.gui.maven.prefs.emit_changed = lambda : self.gui.pref_edited.emit()
		self.gui.preferences_viewer.prefs_model.layoutChanged.emit()


class pref_model(QAbstractTableModel):
	'''
	_window should be an object that has .prefs immediately available
	'''
	def __init__(self,_window,*args):
		super().__init__()
		self._window = _window

	def columnCount(self, index):
		return 2

	def rowCount(self,index):
		return len(self._window.prefs.keys())

	def headerData(self,section,orientation,role):
		if role == Qt.DisplayRole:
			if orientation == Qt.Horizontal:
				return ['Name','Value'][section]
			return section
		elif role == Qt.TextAlignmentRole:
			if orientation == Qt.Horizontal:
				return Qt.AlignLeft | Qt.AlignBottom
			elif orientation == Qt.Vertical:
				return Qt.AlignRight
		return QVariant()

	def data(self,index,role):
		if not index.isValid():
			return QVariant()

		if role == Qt.TextAlignmentRole:
			return Qt.AlignLeft | Qt.AlignVCenter

		column = index.column()
		row = index.row()
		key = list(self._window.prefs.keys())[row]

		if column == 0:
			if role == Qt.DisplayRole:
				return str(key)
			elif role == Qt.InitialSortOrderRole:
				return QVariant(str(key))
		elif column == 1:
			if role == Qt.DisplayRole:
				return str(self._window.prefs[key])
			elif role == Qt.EditRole:
				dtype = self._window.prefs.dtype(key)
				if dtype == np.ndarray:
					import re
					val = str(np.array_repr(self._window.prefs[key]))
					val = re.sub(r"dtype=(\w+)\)$",r"dtype='\1')",val)
					val = re.sub(r"^array",r"np.array",val)
					val = re.sub(r"\t",r"",val)
					val = re.sub(" ",r"",val)
					return val
				else:
					return str(self._window.prefs[key])
			elif role == Qt.InitialSortOrderRole:
				return QVariant(str(key))

		return QVariant()

	def setData(self,index,value,role):
		if not index.isValid():
			return False

		column = index.column()
		row = index.row()
		key = list(self._window.prefs.keys())[row]

		if column == 1:
			dtype = dict.__getitem__(self._window.prefs,key).dtype
			try:
				if dtype == type("a"):
					val = str(value)
				elif dtype == int:
					val = int(value)
				elif dtype == float:
					val = float(value)
				else:
					val = eval(str(value))
				self._window.prefs[key] = val
				if self._window.prefs[key] == val:
					self.dataChanged.emit(index,index)
					return True
			except:
				return False
		return False

	def flags(self,index):
		####Qt.ItemIsSelectable | Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsUserCheckable
		if not index.isValid():
			return Qt.NoItemFlags

		if index.column() == 0:
			return Qt.ItemIsEnabled | Qt.ItemIsSelectable
		elif index.column() == 1:
			return Qt.ItemIsEnabled | Qt.ItemIsEditable | Qt.ItemIsSelectable

		return Qt.NoItemFlags


class prefs_widget(QWidget):
	''' QWidget that displays a `prefs_object`

	This widget has an input bar that allows the model in `prefs_model` to be filtered for displaying a subset of items

	Parameters
	----------

	Attributes
	----------
	proxy_model : QSortFilterProxyModel
		the filtered version of `prefs_model`
	proxy_view : QTableView
		the widget that shows the filter model
	le_filter : QLineEdit
		the input bar for filtering


	'''
	def __init__(self,parent=None):
		super().__init__(parent=parent)
		self.setWindowTitle("Preferences")
		self.setup_widgets()

		proxyLayout = QVBoxLayout()
		proxyLayout.addWidget(self.le_filter)
		proxyLayout.addWidget(self.proxy_view)
		self.setLayout(proxyLayout)

	def set_model(self,prefs_model):
		self.proxy_model.setSourceModel(prefs_model)
		# self.proxy_view.resizeColumnToContents(0)
		# self.proxy_view.resizeColumnToContents(1)
		self.le_filter.clear()
		self.proxy_view.clearSelection()


	def setup_widgets(self):
		self.proxy_model = QSortFilterProxyModel(parent=self.parent())
		self.proxy_model.setDynamicSortFilter(True)

		self.proxy_view = QTableView(parent=self.parent())
		# self.delegate = precision_delegate()
		# self.delegate.refocus = self.refocus_le
		# self.proxy_view.setItemDelegate(self.delegate)

		self.proxy_view.setModel(self.proxy_model)
		self.proxy_view.setSortingEnabled(True)
		self.proxy_view.horizontalHeader().setStretchLastSection(True)
		self.proxy_view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
		self.proxy_view.verticalHeader().setVisible(False)
		self.proxy_view.sortByColumn(0, Qt.AscendingOrder)
		# self.proxy_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
		# self.proxy_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

		self.le_filter = QLineEdit(parent=self.parent())
		self.le_filter.textChanged.connect(self.filter_regex_changed)
		self.le_filter.setPlaceholderText("Filter")

	def filter_regex_changed(self):
		syntax = QRegExp.PatternSyntax(QRegExp.RegExp)
		# QRegExp.RegExp
		# QRegExp.Wildcard
		# QRegExp.FixedString
		regExp = QRegExp(self.le_filter.text(), Qt.CaseInsensitive, syntax)
		self.proxy_model.setFilterRegExp(regExp)

	def refocus_le(self):
		self.le_filter.clear()
		self.proxy_view.clearSelection()
		self.le_filter.setFocus()

	def keyPressEvent(self,event):
		'''Handles key presses

		Notes
		-----
		Escape
			* if `self.le_filter` is not clear, it is cleared. It is also given focus

		'''
		if event.key() == Qt.Key_Escape:
			if self.le_filter.hasFocus() and not str(self.le_filter.text()) == "":
				self.le_filter.clear()
				return
			self.le_filter.setFocus()
			self.proxy_view.clearSelection()
			super().keyPressEvent(event)

		elif event.key() == Qt.Key_Return and self.le_filter.hasFocus():
			# clicksign = 'Execute'
			if self.proxy_model.rowCount() > 0:
				# if self.proxy_model.itemData(self.proxy_model.index(0,0))[0] == self.le_filter.text():
					# self.proxy_model.itemData(self.proxy_model.index(0,1))[0].fxn()
					# self.le_filter.clear()
					# self.proxy_view.clearSelection()
					# self.le_filter.setFocus()
				# else:
				self.focusNextChild()
				self.proxy_view.setCurrentIndex(self.proxy_model.index(0,1))
				return
