import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
import cebra_v2 as cebra2
from collections import defaultdict
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA,FastICA
from torch.utils.data.sampler import BatchSampler
import package
import pandas as pd
import matplotlib.animation as animation
import tembedding
import scipy as sc
from sklearn.cluster import KMeans,OPTICS
import math
import time
import multiprocessing
import queue 
import os
import sys

def STRUCTURE(x,y):
    return 1 - package.preprocessing.structure_batch(x,y)

def main(argv):

    path_dfc = argv[0]
    path_meta = argv[1]
    path_distance = argv[2]
    session = int(argv[3])
    print("Session {}".format(session))

    dfc = np.load(path_dfc)
    print("dfc data loaded")
    meta = pd.read_csv(path_meta, sep="\t")
    print("meta data loaded")

    n_runs = len(set(meta["unique_id"] + meta["monkey"]))
    n_wins = 464
    dfc_all = dfc.reshape((-1, n_wins, 82, 82))
    dfc_all_tensor = torch.from_numpy(dfc_all).type(torch.float32)
    dfc_all_tensor_norm = dfc_all_tensor - torch.mean(dfc_all_tensor,dim = [-2,-1]).reshape(156,464,1,1)
    print("dfc processed")

    for j in range(464):
        print("Element {}".format(j))
        for i in range(156):
            print("Session {} : ok".format(i))
            accu = STRUCTURE(dfc_all_tensor_norm[session,j,:,:],dfc_all_tensor_norm[i,:,:,:]).detach().numpy()
            distance[session,j,i,:] = accu
            distance[i,:,session,j] = accu

    distance = np.load(path_distance)
    print("distance data loaded")

    distance = np.save(path_distance,arr = distance)
    print("distance data saved")

if __name__ == "__main__":
   main(sys.argv[1:])