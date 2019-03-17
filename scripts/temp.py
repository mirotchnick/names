# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
from matplotlib import pyplot as plt
from sklearn import manifold, datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_blobs
#%matplotlib inline


X1, y1 = make_blobs(n_samples=50, centers=[[2,2], [-2,-1], [3,1], [10,4]], cluster_std=0.8)

plt.scatter(X1[:, 0], X1[:, 1], marker='o')

agglom=AgglomerativeClustering(n_clusters=4, linkage = 'average')
agglom.fit(X1,y1)

plt.figure(figsize=(6,4))
x_min, x_max = np.min(X1, axis=0), np.max(X1, axis=0)
X1 = (X1 - x_min) / (x_max - x_min)

for i in range(X1.shape[0]):
    plt.text(X1[i, 0], X1[i, 1], str(y1[i]),
             color=plt.cm.nipy_spectral(agglom.labels_[i] / 10.),
             fontdict={'weight': 'bold', 'size': 9})
    
plt.xticks([])
plt.yticks([])

plt.scatter(X1[:, 0], X1[:, 1], marker='.')
plt.show()

dist_matx = distance_matrix(X1,X1)
print(dist_matx)

Z = hierarchy.linkage(dist_matx, 'complete')
dendro = hierarchy.dendrogram(Z)

Z = hierarchy.linkage(dist_matx, 'average')
dendro = hierarchy.dendrogram(Z)

!wget -O cars_clus.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/cars_clus.csv
filename = 'cars_clus.csv'
pdf = pd.read_csv(filename)
print('shape of dataset: ', pdf.shape)
pdf.head(5)

print('shape of dataset before cleaning: ', pdf.size)
list(pdf)
pdf[['sales', 'resale', 'type', 'price', 'engine_s', 'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg', 'lnsales']] = pdf[['sales', 'resale', 'type', 'price', 'engine_s', 'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce')
pdf = pdf.dropna()
pdf = pdf.reset_index(drop=True)
print('shape of dataset after cleaning: ', pdf.size)
pdf.head(5)

features = pdf[['engine_s', 'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']]
from sklearn.preprocessing import MinMaxScaler
x=features.values
featmat = MinMaxScaler().fit_transform(x)
featmat[0:5]

import scipy
leng = featmat.shape[0]
leng
D = scipy.zeros([leng,leng])
for i in range(leng):
    for j in range(leng):
        D[i,j] = scipy.spatial.distance.euclidean(featmat[i], featmat[j])

import pylab
import scipy.cluster.hierarchy
Z = hierarchy.linkage(D, 'complete')

from scipy.cluster.hierarchy import fcluster
max_d = 3
clusters = fcluster(Z, max_d, criterion='distance')
clusters

k = 5
clusters = fcluster(Z, k, criterion='maxclust')
clusters

fig = pylab.figure(figsize=(18,50))
def llf(id):
    return '[%s %s %s]' % (pdf['manufact'][id], pdf['model'][id], int(float(pdf['type'][id])) )

dendro = hierarchy.dendrogram(Z, leaf_label_func=llf, leaf_rotation=0, leaf_font_size=12, orientation='right')
