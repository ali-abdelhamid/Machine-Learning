#RBE539 ML Team Assignment 2 E2.4

import numpy as np
import pandas as pd

import seaborn as sns
sns.set()

import logging
from time import time

from numpy.random import RandomState
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_olivetti_faces
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

# Load faces data
rng = RandomState(0)
faces, _ = fetch_olivetti_faces(return_X_y=True, shuffle=True, random_state=rng)
n_samples, n_features = faces.shape
n_components = np.arange(0, 2, 1) 

# global centering
faces_centered = faces - faces.mean(axis=0)

# local centering
faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)

data = faces_centered 

#fitting the model
pca = PCA()
pca.fit(data)

fa = FactorAnalysis()
fa.fit(data)

#compare PCA and FA based on covariance:
pca_scores, fa_scores = [], []
for n in n_components:
	pca.n_components = n
	fa.n_components = n
	
	pca_scores.append(np.mean(cross_val_score(pca, data)))
	fa_scores.append(np.mean(cross_val_score(fa, data)))
	

plt.figure()
plt.plot(n_components, pca_scores, "b", label="PCA scores")
plt.plot(n_components, fa_scores, "r", label="FA scores")
plt.title("Cross Validation Scores over nb_components")
plt.xlabel("Number of Components")
plt.ylabel("Mean of Cross Validation Scores")
plt.legend(loc="lower right")
plt.show()
