#weather station clustering

!wget -O weather-stations20140101-20141231.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/weather-stations20140101-20141231.csv

import csv
import pandas as pd
filename='weather-stations20140101-20141231.csv'

pdf = pd.read_csv(filename)
pdf.head(5)

pdf = pdf[pd.notnull(pdf['Tm'])]
pdf = pdf.reset_index(drop=True)
pdf.head(5)

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from pylab import rcParams
%matplotlib inline

#4-Visualization

rcParams['figure.figsize'] = (14,20)

llon=-140
ulon=-50
llat=50
ulat=65

pdf = pdf[(pdf['Long'] > llon) & (pdf['Long'] < ulon) & (pdf['Lat'] > llat) & (pdf['Lat'] < ulat)]

my_map = Basemap(projection='merc',
                resolution='l', area_thresh=1000.0,
                llcrnrlon=llon, llcrnrlat=llat,
                urcrnrlon=ulon, urcrnrlat=ulat)

my_map.drawcoastlines()
my_map.drawcountries()
my_map.fillcontinents(color='white', alpha=0.3)
my_map.shadedrelief()

xs,ys = my_map(np.asarray(pdf.Long), np.asarray(pdf.Lat))
pdf['xm'] = xs.tolist()
pdf['ym'] = ys.tolist()

for index,row in pdf.iterrows():
    my_map.plot(row.xm, row.ym, markerfacecolor=([1,0,0]), marker='o', markersize=5, alpha=0.75)
    
plt.show()


#5- Clustering of stations based on their location i.e. Lat & Lon

from sklearn.cluster import DBSCAN
import sklearn.utils
from sklearn.preprocessing import StandardScaler
sklearn.utils.check_random_state(1000)
Clus_dataset = pdf[['xm','ym']]
Clus_dataset = np.nan_to_num(Clus_dataset)
Clus_dataset = StandardScaler().fit_transform(Clus_dataset)

db = DBSCAN(eps=0.15, min_samples=10).fit(Clus_dataset)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
pdf['Clus_Db'] = labels

realClusterNum = len(set(labels)) - (1 if -1 in labels else 0)
clusterNum = len(set(labels))

pdf[['Stn_Name', 'Tx', 'Tm', 'Clus_Db']].head(5)


#6- Visualization of clusters based on location

my_map.drawcoastlines()
my_map.drawcountries()
my_map.fillcontinents(color='white', alpha=0.3)
my_map.shadedrelief()

colors = plt.get_cmap('jet')(np.linspace(0.0, 1.0, clusterNum))

for clust_num in set(labels):
    c=(([0.4, 0.4, 0.4]) if clust_num == -1 else colors[np.int(clust_num)])
    clust_set=pdf[pdf.Clus_Db == clust_num]
    my_map.scatter(clust_set.xm, clust_set.ym, color=c, marker='o', s=20, alpha=0.85)
    if clust_num != -1:
        cenx=np.mean(clust_set.xm)
        ceny=np.mean(clust_set.ym)
        plt.text(cenx, ceny, str(clust_num), fontsize=25, color='red',)
        print('Cluster '+str(clust_num)+'Avg Temp: '+str(np.mean(clust_set.Tm)))

