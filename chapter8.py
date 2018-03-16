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

# Incremental PCA
'''
Does PCA on mini batches
'''
from sklearn.decomposition import IncrementalPCA
n_batches = 100
inc_pca = IncrementalPCA(n_components=154)

for X_batch in np.array_split(X_train, n_batches):
    inc_pca.partial_fit(X_batch)

X_reduced = inc_pca.transform(X_train)

## Same as above with numpy memmap, allows you to manipulate a file in memory only.
#X_mm = np.memmap(filename, dtype='float32', mode='readonly', shape=(m, n))
#batch_size = m // n_batches
#inc_pca = IncrementalPCA(n_components=154, batch_size=batch_size)
#inc_pca.fit(X_mm)

# Randomised PCA
'''
Stochastic algorithm which finds an approximation of the first d principal 
components. Computational complexity = O(m * d^2) + O(d^3) instead of O(m * n^2)
+ O(n^3) so it is significantly faster when d is much smaller than n.
'''
rnd_pca = PCA(n_components=154, svd_solver='randomized')
X_reduced = rnd_pca.fit_transform(X_train)

# Kernel PCA
'''
Uses the same kernel trick used in SVMs. 
'''
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import KernelPCA

X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
rbf_pca = KernelPCA(n_components=2, kernel='rbf', gamma=0.04)
X_reduced = rbf_pca.fit_transform(X)

# Plot the kernel pca with diff hyperparameters
from sklearn.decomposition import KernelPCA

lin_pca = KernelPCA(n_components = 2, kernel="linear", fit_inverse_transform=True)
rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.0433, fit_inverse_transform=True)
sig_pca = KernelPCA(n_components = 2, kernel="sigmoid", gamma=0.001, coef0=1, fit_inverse_transform=True)

y = t > 6.9

plt.figure(figsize=(11, 4))
for subplot, pca, title in ((131, lin_pca, "Linear kernel"), (132, rbf_pca, "RBF kernel, $\gamma=0.04$"), (133, sig_pca, "Sigmoid kernel, $\gamma=10^{-3}, r=1$")):
    X_reduced = pca.fit_transform(X)
    if subplot == 132:
        X_reduced_rbf = X_reduced
    
    plt.subplot(subplot)
    #plt.plot(X_reduced[y, 0], X_reduced[y, 1], "gs")
    #plt.plot(X_reduced[~y, 0], X_reduced[~y, 1], "y^")
    plt.title(title, fontsize=14)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
    plt.xlabel("$z_1$", fontsize=18)
    if subplot == 131:
        plt.ylabel("$z_2$", fontsize=18, rotation=0)
    plt.grid(True)
plt.show()

# Selecting a kernel and tuning hyperparameters.
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

clf = Pipeline([
        ("kpca", KernelPCA(n_components=2)),
        ("log_reg", LogisticRegression())
    ])

# Param grid for searching hyperparams
param_grid = [{
        "kpca__gamma": np.linspace(0.03, 0.05, 10),
        "kpca__kernel": ["rbf", "sigmoid"]
    }]
        
grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X, y)

# Print the best parameters
print(grid_search.best_params_)

# Inversely transforming a kernel PCA.
rbf_pca = KernelPCA(n_components=2, kernel='rbf', gamma=0.0443,
                    fit_inverse_transform=True)
X_reduced = rbf_pca.fit_transform(X)
X_preimage = rbf_pca.inverse_transform(X_reduced)

# Compute error of reconstruction.
from sklearn.metrics import mean_squared_error
mean_squared_error(X, X_preimage)

# LLE - Locally Linear Embedding.
'''
Manifold learning technique that doesn't rely on projections like PCA. It measures
how each training instance linearly relates to its closest neighbours, and then looks
for a low dimensional version of the training set where these relations are best
conserved.
'''
from sklearn.manifold import LocallyLinearEmbedding
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
X_reduced = lle.fit_transform(X)

# Other dimensionality reduction techniques
'''
1. Multidimensional Scaling (MDS) reduces dimensionality while trying to preserve
the distances between the instances.

2. Isomap creates a graph by connecting each instance to its nearest neighbours then
reduces dimensionality while trying to preserve geodesic distances between the
instances.

3. t-Distributted Stochastic Neighbour Embedding (t-SNE) reduces the dimensionality
while trying to keep similar instances close and dissimilar instances apart.
It's mostly used for visualisation, in particular to visualise clusters of instances
in high-dimensional space.

4. Linear Discriminant Analysis (LDA) is actually a classification algorithm but
during training it learns the most discriminative axes between the classes,
and these axes are then used to define a hyperplane onto which to project the
data. The benefit is that the projection will keep classes as far apart as possible,
so LDA is a good technique to reduce dimensionality before running another classification
algorithm such as an SVM classifier.
'''

# Exercises