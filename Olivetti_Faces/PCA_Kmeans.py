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
faces, target = fetch_olivetti_faces(return_X_y=True, shuffle=True, random_state=rng)
n_samples, n_features = faces.shape

# global centering
faces_centered = faces - faces.mean(axis=0)

# local centering
faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)

data = faces_centered 

#fitting the model
pca = PCA()
pca.fit(data)

#plot variance to show how many principal components preserve at least 90% of the variance effect
pca.explained_variance_ratio_
plt.figure(figsize = (10,8))
plt.plot(range(1,401), pca.explained_variance_ratio_.cumsum(), marker = 'o', linestyle = '--')
plt.title("Cumulative Variance by Components")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance")
plt.show()

#Pick PCA with justified nb of components (80% or higher cumulative variance) and then perform Kmeans clustering on pca scores for principal components
pca.n_components = 50
pca.fit(data)
scores_pca = pca.transform(data)

#Within Cluster sum of squares (wcss) to compare K cluster number applied on PCA transformed data
wcss = []
rn=np.arange(5, 50, 5)

for i in rn:
	kmeans_pca = KMeans(n_clusters = i, init = 'k-means++', random_state=42)
	kmeans_pca.fit(scores_pca)
	wcss.append(kmeans_pca.inertia_)
	
plt.figure(figsize = (10,8))
plt.plot(range(5, 50, 5), wcss, marker = 'o', linestyle='--')
plt.title("KMeans for PCA")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()



#Test/Train split (40 individual faces in the test set). Simple for loop to extract first instance of every target label from 0 to 39. Notice that it does NOT matter which image of the same person we take so long as it is the only image of that person available in the test set

data_test = []
target_test = []
target_shrunk = target.tolist()
data_shrunk=data.tolist()
i=0

while len(target_test)<40:
	for z in range(0, 400):
		if target[z] == i:
			target_test.append(target[z])
			del target_shrunk[z-i]
			data_test.append(data[z])
			del data_shrunk[z-i]
			i = i+1
			continue

#refit pca to new data
pca.n_components = 40 #note that we have to limit our PCA to 40 components as that is all we have in the test set
pca.fit(data_shrunk)
scores_pca = pca.transform(data_shrunk)
pca.fit(data_test)
scores_pca_test = pca.transform(data_test)

#evaluate performance with K=40 clusters:
kmeans_pca = KMeans(n_clusters = 40, init = 'k-means++', random_state=42)
train_clusters = kmeans_pca.fit_predict(scores_pca)
#apply kmeans prediction (provides cluster indeces) to test set of 40 individuals to get their predicted clusters
test_clusters = kmeans_pca.fit_predict(scores_pca_test)

#count how many times our scores_pca prediction (cluster assignment) matches the actual target label by way of the scores_pca_test clustering assignment. Recall that we know the exact order (0 to 39) of labels/targets of the test set and therefore are able to match our clusters for the test with the actual target values
count=0
for w in range(0,360):
	for h in range(0,40):
		#check cluster key
		if train_clusters[w] == test_clusters[h]:
			#check if cluster key matches actual target
			if target_test[h] == target_shrunk[w]:
				count = count+1
				break
				
print("TEST SET ACCURACY:", 100*(count/40))
print("Test Clusters", test_clusters)
print("Test Labels", target_test)
