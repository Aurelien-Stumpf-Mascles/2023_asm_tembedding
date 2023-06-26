# -*- coding: utf-8 -*-
"""
BSk doscovery on synthetic dataset
==================================

Credit: A Grigis

The goal here is to run discovery-driven (only time) or hypothesis-driven
(time + distance between dFC using SSIM) expermiments. The inputs are the
dyncamic functional conectivites (dFCs) computed using a sliding windows
strategy on the synthetic dataset (generated with simtb).
"""

import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tembedding.datasets import (
    Dataset, SimpleMultiSessionDataset, MultiSessionLoader)
from tembedding.models import CEBRAModel
from tembedding.solvers import SingleSessionSolver
from tembedding.losses import EuclideanInfoNCE, CosineInfoNCE


############################################################################
# Load the dataset
# ----------------
#
# Load the synthetic data with 4 brain states and high additional noise.

datadir = os.getenv("TEMBEDDING_DIR")
if datadir is None:
    raise ValueError("Please specify the dataset directory in the "
                     "TEMBEDDING_DIR variable.")
datafile = os.path.join(datadir, "sub-200_states-4_noise-high_synth",
                        "dataset.npz")
dataset = Dataset(
    name="synthetic", test_size=0.1, decimate_ratio=0, random_state=42,
    dataset_path=datafile, dfc_key="dfc45", label_key="labels45")
X_train, y_train = dataset.get_train_data()
X_test, y_test = dataset.get_test_data()
X_train = dataset.unflatten(X_train).astype(np.single)
X_test = dataset.unflatten(X_test).astype(np.single)
print(f"train: {X_train.shape}, {y_train.shape}")
print(f"test: {X_test.shape}, {y_test.shape}")


############################################################################
# Discovery-driven training
# -------------------------
#
# Use a MLP architecture and train CEBRA by using only the time.

input_dim = X_train.shape[-1]
output_dim = 3
distance = "cosine"
model = CEBRAModel(
    nn.Linear(input_dim, 256),
    nn.Dropout(p=0.2),
    nn.GELU(),
    nn.Linear(256, output_dim),
    num_input=input_dim,
    num_output=output_dim,
    normalize = True)
print(model)
ms_dataset = SimpleMultiSessionDataset(X_train)
loader = MultiSessionLoader(ms_dataset, num_steps=3, batch_size=2048,
                            distance=None, time_delta=20, matrix_delta=4)
if distance == "euclidean":
    criterion_klass = EuclideanInfoNCE
else:
    criterion_klass = CosineInfoNCE    
criterion = criterion_klass(temperature=1, beta=1)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
solver = SingleSessionSolver(model=model, criterion=criterion,
                             optimizer=optimizer)
solver.fit(loader)
device = ms_dataset.neural.device
solver.model.eval()
with torch.no_grad():
    embeddings_train = solver.model(dataset.flatten(
        torch.from_numpy(X_train).to(device))).detach().cpu().numpy()
    embeddings_test = solver.model(dataset.flatten(
        torch.from_numpy(X_test).to(device))).detach().cpu().numpy()
print(f"train embeddings: {embeddings_train.shape}")
print(f"test embeddings: {embeddings_test.shape}")

clf = LogisticRegression(random_state=42).fit(embeddings_train, y_train)
y_pred_train = clf.predict(embeddings_train)
y_pred_test = clf.predict(embeddings_test)
acc_train = accuracy_score(y_pred_train, y_train)
acc_test = accuracy_score(y_pred_test, y_test)
print(f"train accuracy: {acc_train}")
print(f"test accuracy: {acc_test}")

fig = plt.figure()
colors = ["black", "red", "green", "blue", "purple"]
cmap = matplotlib.colors.ListedColormap(colors)
ax = fig.add_subplot(projection="3d")
ax.set_title("Train embedding")
ax.scatter(embeddings_train[:, 0], embeddings_train[:, 1],
           embeddings_train[:, 2], cmap=cmap, c=y_train, s=1)
plt.show()

############################################################################
# Hypothesis-driven training
# --------------------------
#
# Same study but add an auxiliary viariable to the model: compare the dFC
# structes (SSIM-like metric) to allow loops in the latent space.

# TODO: speedup
#distances = euclidean_distances(dataset.flatten(X_train))
#print(f"use euclidean distance between dFCs: {distances.shape}")
#distance = torch.from_numpy(distances)

