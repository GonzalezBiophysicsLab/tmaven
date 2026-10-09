import os
import sys
import logging
import appdirs
logger = logging.getLogger('tmaven')

import warnings
warnings.filterwarnings("ignore",message="FNV hashing is not implemented in Numba",category=UserWarning,module="numba.cpython.hashing")

def setup_maven(args=[]):
	from .maven import maven_class
	maven = maven_class(log_stdout='--log_stdout' in args)
	return maven

def setup_application():
	"""Create Qt before importing the scientific controllers and main window."""
	os.environ["MPLCONFIGDIR"] = appdirs.user_config_dir(appname="tmaven", appauthor="python")
	os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
	os.environ["QT_MAC_WANTS_LAYER"] = "1"

	from PyQt5.QtCore import Qt
	from PyQt5.QtWidgets import QApplication
	from . import __version__

	app = QApplication.instance()
	if app is None:
		if hasattr(Qt, "AA_EnableHighDpiScaling"):
			QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
		QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
		app = QApplication([])
	# Let QApplication own the style for the lifetime of its widgets.
	app.setStyle('Fusion')
	app.setApplicationName("tMAVEN")
	app.setDesktopFileName("t.MAVEN")
	app.setApplicationVersion(__version__)
	app.setAttribute(Qt.AA_DontShowIconsInMenus)
	return app


def setup_gui(maven,args=[],app=None):
	logger.info("Launching GUI")
	if app is None:
		app = setup_application()
	from .interface.resources import load_icon
	from .interface.main_window import main_window
	gui = main_window(maven,app)
	maven.gui = gui
	app.setWindowIcon(load_icon('logo.png'))
	return app,gui


def run_app(args=[]):
	"""
	run the application
	"""

	app = setup_application()
	from .interface.splash import loading_splash
	splash = loading_splash(app)
	splash.show()
	app.processEvents()
	try:
		maven = setup_maven(args)
		app, gui = setup_gui(maven,args,app=app)

		if '--safe_mode' in args:
			gui.session_default()
		else:
			gui.session_restore()

		for argument in args:
			if argument.startswith('--startup='):
				maven.scripts.run(argument[10:])
		splash.finish(gui)
	finally:
		splash.close()

	sys.exit(app.exec_())
