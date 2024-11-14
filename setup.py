import io
import os
import re
import setuptools


NAME = "tmaven"
META_PATH = os.path.join("tmaven","__init__.py")
KEYWORDS = ["gui", "single molecule","science","chemistry","physics","biology"]
CLASSIFIERS = [
	"Programming Language :: Python :: 3",
	"License :: OSI Approved :: License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
	"Operating System :: OS Independent",
	"Topic :: Scientific/Engineering :: Biology",
	"Topic :: Scientific/Engineering :: Chemistry",
	"Topic :: Scientific/Engineering :: Physics",
	"Environment :: X11 Applications :: Qt",
]
INSTALL_REQUIRES = [
	"numpy>=1.21.0",
	"numba>=0.51.0",
	"scipy>=1.7.0",
	"matplotlib>=3.5.0",
	"h5py>=3.6.0",
	"PyQt5>=5.15.0",
	"appdirs",
	"importlib_resources",
	"biasd @ git+https://github.com/ckinzthompson/biasd.git@main#egg=biasd",
]
EXTRAS_REQUIRE = {
	"docs": [
	],
	"tests": [
	],
	"release": [
	],
}
EXTRAS_REQUIRE["dev"] = EXTRAS_REQUIRE["tests"] + EXTRAS_REQUIRE["docs"]

###############################################################################

HERE = os.path.abspath(os.path.dirname(__file__))

def read(*parts):
	"""
	Build an absolute path from *parts* and and return the contents of the
	resulting file.  Assume UTF-8 encoding.
	"""
	with io.open(os.path.join(HERE, *parts), encoding="utf-8") as f:
		return f.read()

META_FILE = read(META_PATH)

def find_meta(meta):
	"""
	Extract __*meta*__ from META_FILE.
	"""
	meta_match = re.search(
		r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta),
		META_FILE, re.M
	)
	if meta_match:
		return meta_match.group(1)
	raise RuntimeError("Unable to find __{meta}__ string.".format(meta=meta))


if __name__ == "__main__":
	setuptools.setup(
		name=NAME,
		description=find_meta("description"),
		license=find_meta("license"),
		url=find_meta("url"),
		version=find_meta("version"),
		author=find_meta("author"),
		keywords=KEYWORDS,
		long_description=read("readme.md"),
		long_description_content_type='text/markdown',
		packages=setuptools.find_packages(where=".")+ ["tmaven.bin"],
		package_dir={"": "."},
		package_data={
			"": ["*.png","*.svg"],
		},
		zip_safe=False,
		classifiers=CLASSIFIERS,
		install_requires=INSTALL_REQUIRES,
		extras_require=EXTRAS_REQUIRE,
		entry_points={
				'console_scripts': [
					'tmaven=tmaven.bin.tmaven_gui:main',
				],
		},
	)
