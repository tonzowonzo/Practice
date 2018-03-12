# Chapter 8 of machine learning book - Dimensionality reduction.
import numpy as np
import os

# Numpy seed
np.random.seed(42)

# For figures.
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Define dataset
np.random.seed(4)
m = 60
w1, w2 = 0.1, 0.3
noise = 0.1

angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
X = np.empty((m, 3))
X[:, 0] = np.cos(angles) + np.sin(angles) / 2 + noise * np.random.randn(m) / 2
X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)

#PCA
X_center = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(X_center)
c1 = Vt.T[:, 0]
c2 = Vt.T[:, 1]

# Projects training set onto hyperplane defined by 2 principle components
W2 = Vt.T[:, 2]
X2D = X_center.dot(W2)
X2D_using_svd = X2D

# PCA with sklearn
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X2D = pca.fit_transform(X)
print(X2D[:5])

# PCA for compression.
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
mnist = fetch_mldata('MNIST original')
X = mnist['data']
y = mnist['target']

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Compress mnist data.
pca = PCA()
pca.fit(X_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1
print(d)

# Compress with maintaining 95% of variance using hyperparameter
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_train)
print(pca.n_components_)
# This outputs 154, the same as doing it with numpy above

# Decompress the PCA
X_recovered = pca.inverse_transform(X_reduced)
