import os
import sys
import logging
import platform

def setup_logging(stdout=False):
	log = logging.getLogger()
	log.setLevel(logging.DEBUG)
	logging.getLogger("matplotlib").setLevel(logging.WARNING)
	logging.getLogger("numba").setLevel(logging.WARNING)

	log_fmt = ("%(asctime)s: %(name)s.%(funcName)s(%(lineno)d): %(message)s")
	formatter = logging.Formatter(log_fmt)

	stdout_handler = logging.StreamHandler()
	stdout_handler.setFormatter(formatter)
	stdout_handler.setLevel(logging.DEBUG)

	if stdout:
		log.addHandler(stdout_handler)

def setup_maven(args=[]):
	for a in args:
		if a.startswith('--log_stdout'):
			setup_logging(stdout=True)

	from . import __version__
	logging.info("Starting tmaven {}".format(__version__))
	logging.info(platform.uname())
	logging.info("Platform: {}".format(platform.platform()))
	logging.info("Python path: \n   {}".format('\n   '.join(sys.path)))

	import pkg_resources
	these_pkgs = ['matplotlib','scipy','numpy','h5py','numba','PyQt5']
	packages = '\n   '.join(['{} {}'.format(d.project_name, d.version) for d in pkg_resources.working_set if d.project_name in these_pkgs])
	logging.info('packages:\n   '+packages)

	from .maven import maven_class
	maven = maven_class()

	return maven

def setup_gui(maven,args=[]):
	logging.info("Launching GUI")
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

	app = QApplication([])
	app.setApplicationName("tMAVEN")
	app.setDesktopFileName("t.MAVEN")

	app.setApplicationVersion(__version__)
	app.setAttribute(Qt.AA_DontShowIconsInMenus)

	gui = main_window(maven,app)
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

	# Restore the previous session
	if not '--safe_mode' in args:
		gui.restore_session()
	for a in args:
		if a.startswith('--startup='):
			script_name = a[10:]
			maven.scripts.run(script_name)
		else: ### This is where other command line options are added
			pass

	sys.exit(app.exec_())
