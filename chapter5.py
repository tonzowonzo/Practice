# SVMs
# Linear SVM classification
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
X = iris['data'][:, (2, 3)] # Petal length, petal width.
y = (iris['target'] == 2).astype(np.float64) # Iris-Virginica.

svm_clf = Pipeline((
                    ('scaler', StandardScaler()),
                    ('linear_svc', LinearSVC(C=1, loss='hinge')),
                    ))

svm_clf.fit(X, y)

svm_clf.predict([[5.5, 1.7]])

# Polynomial classification.
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

X, y = make_moons()
polynomial_svm_clf = Pipeline((
                               ('poly_features', PolynomialFeatures(degree=3)),
                               ('scaler', StandardScaler()),
                               ('svm_clf', LinearSVC(C=10, loss='hinge'))
                               ))
polynomial_svm_clf.fit(X, y)

# Polynomial kernel.
from sklearn.svm import SVC
poly_kernel_svm_clf = Pipeline((
                                ('scaler', StandardScaler()),
                                ('svm_clf', SVC(kernel='poly', degree=3, coef0=1, C=5))
                                ))
poly_kernel_svm_clf.fit(X, y)

# SVR - regression.
np.random.seed(42)
m = 50
X = 2 * np.random.rand(m, 1)
y = (4 + 3 * X + np.random.randn(m, 1)).ravel()

from sklearn.svm import LinearSVR
svm_reg = LinearSVR(epsilon=1.5, random_state=42) # Epsilon is the size of the margin.
svm_reg.fit(X, y)

# Polynomial SVR.
from sklearn.svm import SVR
svm_poly_reg = SVR(kernel='poly', degree=2, C=100, epsilon=0.1)
svm_poly_reg.fit(X, y)

# Train a linear SVC on a linearly separable dataset.
iris = datasets.load_iris()
X = iris['data'][:, (2, 3)] # Petal length, petal width.
y = iris['target']

# Setosa or versicolor.
setosa_or_versicolor = (y == 0) | (y == 1)
X = X[setosa_or_versicolor]
y = y[setosa_or_versicolor]

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

lin_clf = LinearSVC(loss='hinge', C=5)
svc_clf = SVC(kernel='linear', C=5)
sgd_clf = SGDClassifier(loss='hinge', n_iter=100000)

# Scale X.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit classifiers.
lin_clf.fit(X_scaled, y)
svc_clf.fit(X_scaled, y)
sgd_clf.fit(X_scaled, y)

# Print line equations
print("LinearSVC:                   ", lin_clf.intercept_, lin_clf.coef_)
print("SVC:                         ", svc_clf.intercept_, svc_clf.coef_)
print("SGDClassifier(alpha={:.5f}):".format(sgd_clf.alpha), sgd_clf.intercept_, sgd_clf.coef_)

# As you can see the equations are very similar.
