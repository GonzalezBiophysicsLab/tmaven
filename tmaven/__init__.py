"""
tMAVEN - Modeling, Analysis and Visualization ENvironment for N-dimensional single-molecule time series.
"""

__title__ = "tMAVEN"
__version__ = "0.1.0"

__description__ = "tMAVEN - Modeling, Analysis and Visualization ENvironment for N-dimensional single-molecule time series."

__license__ = "GPLv3"
__url__ = ""

__author__ = ""
__email__ = ""



from .app import run_app
def __main__():
	import sys

	run_app(sys.argv)

if __name__ == '__main__':
	__main__()
