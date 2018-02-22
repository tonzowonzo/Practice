# Decision trees.

# Imports
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data[:, 2:]
y = iris.target

# Define classifier.
tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)

# Visualise the tree
from sklearn.tree import export_graphviz

export_graphviz(
                tree_clf,
                feature_names = iris.feature_names[2:],
                class_names=iris.target_names,
                rounded=True,
                filled=True
                )

# Making predictions.
# Estimating probabilities of individual classes
tree_clf.predict_proba([[5, 1.5]])

# Predict
tree_clf.predict([[5, 1.5]])