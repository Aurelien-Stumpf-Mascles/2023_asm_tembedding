# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2020
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# Module current version
version_major = 1
version_minor = 0
version_micro = 0

# Expected by setup.py: string of form "X.Y.Z"
__version__ = "{0}.{1}.{2}".format(version_major, version_minor, version_micro)

# Expected by setup.py: the status of the project
CLASSIFIERS = ["Development Status :: 5 - Production/Stable",
               "Environment :: Console",
               "Environment :: X11 Applications :: Qt",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering",
               "Topic :: Utilities"]

# Project descriptions
description = (
    "NeuroImaging Connectivity Package to compute fonctional static or " 
    "dynamic connectivity.\n")
SUMMARY = (
    "NeuroImaging Connectivity Package to compute fonctional static or " 
    "dynamic connectivity.\n")
long_description = (
    "NeuroImaging Connectivity Package to compute fonctional static or " 
    "dynamic connectivity.\n")

# Main setup parameters
NAME = "nicon"
ORGANISATION = "CEA"
MAINTAINER = "Antoine Grigis"
MAINTAINER_EMAIL = "antoine.grigis@cea.fr"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "https://github.com/neurospin/nicon"
DOWNLOAD_URL = "https://github.com/neurospin/nicon"
LICENSE = "CeCILL-B"
CLASSIFIERS = CLASSIFIERS
AUTHOR = "nicon developers"
AUTHOR_EMAIL = "antoine.grigis@cea.fr"
PLATFORMS = "OS Independent"
ISRELEASE = True
VERSION = __version__
PROVIDES = ["nicon"]
REQUIRES = [
    "elasticsearch>=7.0.5",
    "progressbar2>=3.39.3"
]
EXTRA_REQUIRES = {
}
SCRIPTS = [
]
