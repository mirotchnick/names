import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pylab as pl
import tensorflow as tf
import matplotlib.patches as mpatch
%matplotlib inline
plt.rcParams['figure.figsize'] = (10,6)

X = np.arange(0.0, 5.0, 0.1)

a = 1
b = 0

Y = a * X + b

plt.plot(X,Y)
plt.ylabel('dependant')
plt.xlabel('indie')
plt.title('reg')
plt.show()

!wget -O FuelConsumption.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv

df = pd.read_csv('FuelConsumption.csv')