#! /usr/bin/env python
# -*- coding: utf-8 -*-

from sys import argv, exit
import os, re

if "upl" in argv[1:]:
    os.system("python setup.py register -r pypi")
    os.system("python setup.py sdist upload -r pypi")
    exit()

m = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "soif", "_version.py")).read()
version = re.findall(r"__version__ *= *\"(.*?)\"", m)[0]

try:
    from setuptools import setup
    setup
except ImportError:
    from distutils.core import setup
    setup

setup(
    name = "soif",
    version = version,
    author = "Guillaume Schworer",
    author_email = "guillaume.schworer@obspm.fr",
    packages = ["soif"],
    url = "https://github.com/ceyzeriat/soif/",
    license = "GNU General Public License v3 or later (GPLv3+)",
    description = "Software for Optical Interferometry Fitting",
    long_description = open("README.rst").read() + "\n\n"
                    + "Changelog\n"
                    + "---------\n\n"
                    + open("HISTORY.rst").read(),
    package_data = {"": ["LICENSE", "AUTHORS.rst", "HISTORY.rst"]},
    include_package_data = True,
    install_requires = ["numpy>1.6.1","emcee","corner","patiencebar","MCres"],
    download_url = 'https://github.com/ceyzeriat/soif/tree/master/dist',
    keywords = ['astronomy','interferometry','data','processing','reduction','model','fitting','optical'],
    package_data={"": ["README.rst", "LICENSE", "HISTORY.rst"]},
    include_package_data=True,
    classifiers = [
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        'Intended Audience :: Education',
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Documentation :: Sphinx",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Astronomy"
    ],
)

