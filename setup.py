import os
import re
from setuptools import setup


base_dir = os.path.dirname(__file__)


DUNDER_ASSIGN_RE = re.compile(r"""^__\w+__\s*=\s*['"].+['"]$""")
about = {}
with open(os.path.join(base_dir, "tmaven", "__init__.py"), encoding="utf8") as f:
    for line in f:
        if DUNDER_ASSIGN_RE.search(line):
            exec(line, about)

with open(os.path.join(base_dir, "readme.md"), encoding="utf8") as f:
    readme = f.read()

with open(os.path.join(base_dir, "changelog.md"), encoding="utf8") as f:
    changes = f.read()

install_requires = [
    "numpy>=1.21.0,<2.0.0",
    "numba>=0.51.0",
    "toml",
    "scipy>=1.7.0",
    "matplotlib>=3.5.0",
    "h5py>=3.6.0",
    "PyQt5>=5.15.0",
    "appdirs",
    "importlib_resources",
    "biasd @ git+https://github.com/ckinzthompson/biasd.git@main"
]


extras_require = {
}

extras_require["all"] = list(
    {req for extra, reqs in extras_require.items() for req in reqs}
)

setup(
    name=about["__title__"],
    version=about["__version__"],
    description=about["__description__"],
    long_description="{}\n\n{}".format(readme, changes),
    author=about["__author__"],
    url=about["__url__"],
    license=about["__license__"],
    packages=[
        "tmaven",
        "tmaven.controllers",
        "tmaven.controllers.analysis_plots",
        "tmaven.controllers.corrections",
        "tmaven.controllers.io_special",
        "tmaven.controllers.modeler",
        "tmaven.controllers.modeler.fxns",
        "tmaven.controllers.photobleaching",
        "tmaven.controllers.trace_filter",
        "tmaven.interface",
        "tmaven.interface.hdf5_view",
        "tmaven.interface.modeler",
        "tmaven.interface.resources",
        "tmaven.pysmd",
        "tmaven.trace_plot",
    ],
    python_requires=">=3.6",
    install_requires=install_requires,
    extras_require=extras_require,
    package_data={"tmaven.interface.resources": ["*.png", "*.svg"]},
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Environment :: X11 Applications :: Qt"
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Biology",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    entry_points={"console_scripts": ["tmaven = tmaven.__main__:main"]},
)