from PyQt5.QtGui import QPixmap, QIcon
from importlib_resources import files

def path(fname, resource_dir="tmaven.interface.resources"):
	f = files(resource_dir).joinpath(fname)
	return str(f)

def load_icon(fname):
	"""Load an icon from the resources directory."""
	return QIcon(path(fname))


def load_pixmap(fname):
	"""Load a pixmap from the resources directory."""
	return QPixmap(path(fname))
