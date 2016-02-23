from sklearn.datasets import load_iris
from Discretization.MDLP import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

data = load_iris()
Xraw = data.data
yraw = data.target

X = np.hstack(( Xraw[:, :3], yraw.reshape((len(yraw), 1)) ))
X = pd.DataFrame(X, columns=["a","b","c","y"])

discretizer = MDLP_Discretizer(dataset=X, class_label="y")
D = discretizer._data
for f in range(len(D.columns)):
    vals = list(set(np.array(D[D.columns[f]]).flatten()))
    for i in range(len(vals)):
        if type(vals[i]) == str:
            D[D.columns[f]] = D[D.columns[f]].replace(vals[i], i)

D = np.array(D)[:, :3].astype(int)
print D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Xraw[:,0], Xraw[:,1], Xraw[:,2], c=yraw)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(D[:,0], D[:,1], D[:,2], c=yraw)

plt.show()