# SVMs
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
