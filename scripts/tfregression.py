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

#!wget -O FuelConsumption.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv

df = pd.read_csv('FuelConsumption.csv')
df.head()

train_x = np.asanyarray(df[['ENGINESIZE']])
train_y = np.asanyarray(df[['CO2EMISSIONS']])

a = tf.Variable(20.0)
b = tf.Variable(30.2)
y = a * train_x + b

loss = tf.reduce_mean(tf.square(train_y - y))

optimizer = tf.train.GradientDescentOptimizer(0.05)

train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sesh = tf.Session()
sesh.run(init)

loss_values = []
train_data = []
for step in range(100):
    _, loss_val, a_val, b_val = sesh.run([train, loss, a, b])
    loss_values.append(loss_val)
    if step % 5 == 0:
        print (step, loss_val, a_val, b_val)
        train_data.append([a_val, b_val])

plt.plot(loss_values, 'ro')

cr, cg, cb = (1.0, 1.0, 0.0)
for f in train_data:
    cb += 1.0 / len(train_data)
    cg -= 1.0 / len(train_data)
    if cb > 1.0: cb = 1.0
    if cg < 0.0: cg = 0.0
    [a, b] = f
    f_y = np.vectorize(lambda x: a*x + b)(train_x)
    line = plt.plot(train_x, f_y)
    plt.setp(line, color=(cr,cg,cb))

plt.plot(train_x, train_y, 'ro')

green_line = mpatch.Patch(color='red', label='Data Points')
plt.legend(handles=[green_line])
plt.show()