# -*- coding: utf-8 -*-
###############################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
###############################################################################

import os
import json
import numpy as np
import glob
from scipy.io import loadmat
from pprint import pprint

labels_file = "/neurospin/lbi/monkeyfmri/tmp/Resting_state/structural_matrix/RM_RegionNames.txt"
names_file = "/neurospin/lbi/monkeyfmri/tmp/Resting_state/coconombres.json"
#matrix_file = "/neurospin/lbi/monkeyfmri/tmp/Resting_state/structural_matrix/cocoSCmia.mat"
matrix_file = "/neurospin/lbi/monkeyfmri/tmp/Resting_state/macaque_FCSC_distances.mat"
output_file = "/neurospin/nsap/processed/dynamic_networks/data/structural.txt"


labels = np.loadtxt(labels_file)[:, 0]
with open(names_file, "rt") as open_file:
    names = json.load(open_file)
mapping = dict((k1, k2) for k1, k2 in zip(names, labels))
pprint(mapping)
mat = loadmat(matrix_file)
labels = [mapping[name[0]] for name in mat["ROIlabels"].squeeze()]
print(labels)
order = np.argsort(labels)
matrix = mat["cocoSC"]
order_matrix = matrix[:, order][order, :]
np.savetxt(output_file, order_matrix)

