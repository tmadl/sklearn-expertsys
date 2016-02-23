from sklearn.datasets import load_iris
from Discretization.MDLP import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

data = load_iris()
X = data.data
y = data.target

X = X[:, :3]

discretizer = MDLP_Discretizer(dataset=pd.DataFrame(X), class_label=y)
D = discretizer._data
print D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2], c=y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(D[:,0], D[:,1], D[:,2], c=y)

plt.show()