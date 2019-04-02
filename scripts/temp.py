# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
from matplotlib import pyplot as plt
from sklearn import manifold, datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_blobs
%matplotlib inline


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


#hierarchical clustering with scikit-learn

dist_mat = distance_matrix(featmat, featmat)
print(dist_mat)

agglom = AgglomerativeClustering(n_clusters=6, linkage='complete')
agglom.fit(featmat)
agglom.labels_

pdf['cluster_'] = agglom.labels_
pdf.drop('cluster', axis=1)
pdf.head()

import matplotlib.cm as cm
n_clusters = max(agglom.labels_) + 1
colors = cm.rainbow(np.linspace(0, 1, n_clusters))
cluster_labels = list(range(0, n_clusters))

plt.figure(figsize=(16,14))

for color, label in zip(colors, cluster_labels):
        subset = pdf[pdf.cluster_ == label]
        for i in subset.index:
                plt.text(subset.horsepow[i], subset.mpg[i], str(subset['model'][i]), rotation = 25)
        plt.scatter(subset.horsepow, subset.mpg, s=subset.price*10, c=color, label='cluster'+str(label), alpha=0.5)

plt.legend()
plt.title('Clusters')
plt.xlabel('horsepower')
plt.ylabel('mpg')
plt.show()

pdf.groupby(['cluster_', 'type'])['cluster_'].count()

agg_cars = pdf.groupby(['cluster_', 'type'])['horsepow', 'engine_s', 'mpg', 'price'].mean()
agg_cars

plt.figure(figsize=(16,10))
for color, label in zip(colors, cluster_labels):
        subset = agg_cars.loc[(label,),]
        for i in subset.index:
                plt.text(subset.loc[i][0]+5, subset.loc[i][2], 'type='+str(int(i)) + ', price ='+str(int(subset.loc[i][3]))+'k')
        plt.scatter(subset.horsepow, subset.mpg, s=subset.price*20, c=color, label='cluster'+str(label))
plt.legend()
plt.title('Clusters')
plt.xlabel('horsepower')
plt.ylabel('mpg')


#density-based clustering
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def createDataPoints(centroidLocation, numSamples, clusterDeviation):
        X, y = make_blobs(n_samples=numSamples, centers=centroidLocation, cluster_std=clusterDeviation)
        X = StandardScaler().fit_transform(X)
        return X, y

X, y = createDataPoints([[4,3], [2,-1], [-1,4]], 1500, 0.5)

eps = 0.3
minSamps = 7
db = DBSCAN(eps=eps, min_samples=minSamps).fit(X)
labels = db.labels_
labels

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_clusters

unique_labels = set(labels)
unique_labels

colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
colors

for k, col in zip(unique_labels, colors):
        if k == -1:
                col = 'k'

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.scatter(xy[:, 0], xy[:, 1], s=50, c=col, marker=u'o', alpha=0.5)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.scatter(xy[:, 0], xy[:, 1], s=50, c=col, marker=u'o', alpha=0.5)


