import os
import sys
import logging
logger = logging.getLogger('tmaven')

def setup_maven(args=[]):
	from .maven import maven_class
	maven = maven_class(log_stdout='--log_stdout' in args)
	return maven

def setup_gui(maven,args=[]):
	logger.info("Launching GUI")
	from PyQt5.QtCore import Qt
	from PyQt5.QtWidgets import QApplication

	from .interface.resources import load_icon
	from .interface.main_window import main_window
	from . import __version__

	# Images (such as toolbar icons) aren't scaled nicely on retina/4k displays
	# unless this flag is set
	os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
	if hasattr(Qt, "AA_EnableHighDpiScaling"):
		QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
	QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

	# An issue in PyQt5 v5.13.2 to v5.15.1 makes PyQt5 application
	# hang on Mac OS 11 (Big Sur)
	# Setting this environment variable fixes the problem.
	# See issue Mu-editor#1147 for more information
	os.environ["QT_MAC_WANTS_LAYER"] = "1"

	# app = QApplication([])
	app = QApplication.instance()
	if app is None:
		app = QApplication([])
	
	app.setApplicationName("tMAVEN")
	app.setDesktopFileName("t.MAVEN")

	app.setApplicationVersion(__version__)
	app.setAttribute(Qt.AA_DontShowIconsInMenus)

	# logging.getLogger("tmaven").setLevel(logging.WARNING)
	gui = main_window(maven,app)
	# logging.getLogger("tmaven").setLevel(logging.INFO)
	app.setWindowIcon(load_icon('logo.png'))

	for a in args:
		### This is where other command line options are added
		continue

	return app,gui

def run_app(args=[]):
	"""
	run the application
	"""

	maven  = setup_maven(args)
	app, gui = setup_gui(maven,args)

	# Restore the previous session?
	if not '--safe_mode' in args:
		gui.restore_session()
	else:
		gui.default_session()

	for a in args:
		if a.startswith('--startup='):
			script_name = a[10:]
			maven.scripts.run(script_name)
		else: ### This is where other command line options are added
			pass


	sys.exit(app.exec_())
