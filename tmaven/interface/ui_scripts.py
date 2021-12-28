import logging
logger = logging.getLogger(__name__)

def run(gui):
	from PyQt5.QtWidgets import QFileDialog
	fname = QFileDialog.getOpenFileName(gui,'Choose script (.py) to run ','./')[0]
	if fname == "":
		return
	gui.maven.scripts.run(fname)

def input_run(gui):
	from PyQt5.QtWidgets import QInputDialog
	text = QInputDialog.getMultiLineText(gui,'Enter code to run', "Code", "")[0]
	logger.info('input script: %s'%(text))
	if text == "" :
		return False
	try:
		logger.info('Script: running \n%s'%(text))
		code = compile(text,'<string>','exec')
		context = {'maven':gui.maven} ## this will hold all of the elements in the plugin file... eg functions, global variables.
		exec(code,context)
	except Exception as e:
		logger.error(str(e))
		return False
	return True
