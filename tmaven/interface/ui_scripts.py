import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QToolBar, QAction, QFileDialog
from PyQt5.QtCore import Qt
import appdirs
import json
import os 

def run(gui):
	from PyQt5.QtWidgets import QFileDialog
	fname = QFileDialog.getOpenFileName(gui,'Choose script (.py) to run ','./')[0]
	if fname == "":
		return
	run_file(gui,fname)

def input_run(gui):
	from PyQt5.QtWidgets import QInputDialog
	text = QInputDialog.getMultiLineText(gui,'Enter code to run', "Code", "")[0]
	logger.info('input script: %s'%(text))
	if text == "" :
		return False
	try:
		logger.info('Script: running \n%s'%(text))
		code = compile(text,'<string>','exec')
		context = {'maven':gui.maven,'gui':gui} ## this will hold all of the elements in the plugin file... eg functions, global variables.
		exec(code,context)
		gui.maven.emit_data_update()
	except Exception as e:
		logger.error(str(e))
		return False
	return True

def run_file(gui,fname):
	gui.maven.scripts.run(fname, gui=gui)
	gui.maven.emit_data_update()

class scripts_toolbar(QToolBar):
	def __init__(self, gui,):
		super().__init__("Scripts Toolbar", parent=gui)
		self.gui = gui

		self.toggle_toolbar_action = QAction("Hide", self)
		self.toggle_toolbar_action.triggered.connect(self._toggle)
		self.addAction(self.toggle_toolbar_action)

		self.clear_action = QAction("Clear", self)
		self.clear_action.triggered.connect(self.clear_actions)
		self.addAction(self.clear_action)

		self.add_action = QAction("+", self)
		self.add_action.triggered.connect(self.add_file_action)
		self.addAction(self.add_action)
		
		self.addSeparator()

		self.action_group = []
		self.load_last()

	def toggle(self):
		# self.toggle_toolbar_action.trigger()
		if self.isHidden():
			self.show()
		else:
			self.hide()
		
	def _toggle(self, checked):
		if checked:
			self.show()
		else:
			self.hide()

	def clear_actions(self,event=None,update=True):
		for action in self.action_group[::-1]:
			self.remove_action(action)
		if update:
			self.update_script_list()

	def add_file_action(self):
		options = QFileDialog.Options()
		filenames, _ = QFileDialog.getOpenFileNames(self, "Select Files", self.gui.lwd, "Python (*.py)", options=options)
		if len(filenames) > 0:
			for filename in filenames:
				self._add_file_action(filename)
			lwd = os.path.dirname(filenames[-1])
			self.gui.lwd_update(lwd)

	def _add_file_action(self,filename,update=True):
		file_action = QAction(filename.split('/')[-1], self)
		file_action.setToolTip(filename)
		file_action.triggered.connect(lambda checked, f=filename: self.run_script(f))
		file_action.script_location = filename
		self.addAction(file_action)
		self.action_group.append(file_action)
		if update:
			self.update_script_list()
	
	def update_script_list(self):
		scripts = [saction.script_location for saction in self.action_group]
		self.gui.config_file_update('recent_files.json','scripts',scripts)

	def load_last(self):
		files = self.gui.config_file_read('recent_files.json')
		if 'scripts' in files:
			self.clear_actions(update=False)
			for script in files['scripts']:
				self._add_file_action(script,update=False)

	def run_script(self, fname):
		run_file(self.gui,fname)

	def contextMenuEvent(self, event):
		pass  # Do nothing to disable the default context menu

	def mousePressEvent(self, event):
		if event.button() == Qt.RightButton:
			action = self.actionAt(event.pos())
			if action and action not in [self.toggle_toolbar_action, self.add_action]:
				self.remove_action(action)
				self.update_script_list()
		super().mousePressEvent(event)

	def remove_action(self, action):
		self.removeAction(action)
		self.action_group.remove(action)