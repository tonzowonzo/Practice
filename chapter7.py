# Ensemble methods.
# Imports
import numpy as np
import os

# Random seed.
np.random.seed(42)

# For plotting.
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save figures to.
PROJECT_ROOT_DIR = 'C:\\Users\\Tim\\pythonscripts\\MLbook'
CHAPTER_ID = 'ensembles'

def image_path(fig_id):
    return os.path.join(PROJECT_ROOT_DIR, 'images', CHAPTER_ID, fig_id)
    
def save_fig(fig_id, tight_layout=True):
    print('Saving figure', fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(image_path(fig_id) + '.png', format='png', dpi=300)
    

# Voting classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Import dataset
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Classifiers.
log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()

voting_clf = VotingClassifier(
                              estimators=[('lr', log_clf), ('rf', rnd_clf),
                                          ('svc', svm_clf)],
                                          voting='hard'
                                          )
voting_clf.fit(X_train, y_train)

# Show accuracy of each classifier.
from sklearn.metrics import accuracy_score

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
    
# Soft voting
'''
Uses the probabilities of each class instead of simple hard voting.
'''
# Classifiers.
log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC(probability=True)

voting_clf = VotingClassifier(
                              estimators=[('lr', log_clf), ('rf', rnd_clf),
                                          ('svc', svm_clf)],
                                          voting='soft'
                                          )
voting_clf.fit(X_train, y_train)

# Show accuracy of each classifier.
from sklearn.metrics import accuracy_score

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
    
# Bagging and pasting.
'''
Bagging is when the same training algorithm is used on different subsets of data
which is short for bootstrap aggregating, if this sampling is performed without
replacement it is called 'pasting' instead.
'''
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Bagging classifier which runs on all cores, set bootstrap to False for a pasting.
bag_clf = BaggingClassifier(
                            DecisionTreeClassifier(), n_estimators=10,
                            max_samples=100, bootstrap=True, n_jobs=1)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)

# Accuracy of bagging
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

# Plot the decision trees
from matplotlib.colors import ListedColormap

def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.5, -1, 1.5], alpha=0.5, contour=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]   
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap, linewidth=10)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], 'yo', alpha=alpha)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], 'bs', alpha=alpha)
    plt.axis(axes)
    plt.xlabel(r'$x_1$', fontsize=18)
    plt.ylabel(r'$x_2$', fontsize=18, rotation=0)
    
plt.figure(figsize=(11, 4))
plt.subplot(121)
plot_decision_boundary(bag_clf, X, y)
plt.title('Decision trees with bagging', fontsize=14)
save_fig('decision_tree_with_bagging')
plt.show()

# Out of the bag evaluation.
'''
Because when bagging the training algorithm only uses approximately 2/3 of the
data to train on it is possible to evaluate each algorithm with the left over
data.
'''
bag_clf = BaggingClassifier(
                            DecisionTreeClassifier(), n_estimators=10,
                            bootstrap=True, oob_score=True,
                            n_jobs=1) # Shows out of the bag score.
bag_clf.fit(X_train, y_train)
bag_clf.oob_score_

from sklearn.metrics import accuracy_score
y_pred = bag_clf.predict(X_test)
accuracy_score(y_test, y_pred)

# Random forests
from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators=10, max_leaf_nodes=16, n_jobs=1)
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)

# Bagging classifier similar to random forest above.
bag_clf = BaggingClassifier(
                            DecisionTreeClassifier(splitter='random', max_leaf_nodes=16),
                            n_estimators=10, max_samples=1.0, bootstrap=True, n_jobs=1)

# Extra trees classifier
'''
Makes the trees more random than random forest, this increases bias and decreases
variance.
'''
from sklearn.ensemble import ExtraTreesClassifier
extra_clf = ExtraTreesClassifier(n_estimators=10, n_jobs=1)
extra_clf.fit(X_train, y_train)
extra_clf_pred = extra_clf.predict(X_test)
accuracy_score(y_test, extra_clf_pred)

# Feature importance in random classifier
from sklearn.datasets import load_iris
iris = load_iris()
rnd_clf = RandomForestClassifier(n_estimators=10, n_jobs=1)
rnd_clf.fit(iris['data'], iris['target'])
for name, score in zip(iris['feature_names'], rnd_clf.feature_importances_):
    print(name, score)
    
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
rnd_clf = RandomForestClassifier(random_state=42)
rnd_clf.fit(mnist['data'], mnist['target'])

# Plot MNIST digit
def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap=matplotlib.cm.hot,
               interpolation='nearest')
    plt.axis('off')
    
# Plot feature importances
plot_digit(rnd_clf.feature_importances_)
cbar = plt.colorbar(ticks=[rnd_clf.feature_importances_.min(), rnd_clf.feature_importances_.max()])
cbar.ax.set_yticklabels(['not important', 'very important'])
plt.show()

# Boosting
'''
Refers to any ensemble method that can combine several weak learners into
a strong learner.
'''
# Adaboost.
'''
Focuses more on unterfitted instances. Results in a higher focus on hard cases.
Ie: A base classifier is trained and used to make predictions on the training
set, then the weight of misclassified training instances is increased and a new
classifier is trained.
'''
from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(
                             DecisionTreeClassifier(max_depth=1), n_estimators=100,
                             algorithm='SAMME.R', learning_rate=0.5)
ada_clf.fit(X_train, y_train)
y_pred = ada_clf.predict(X_test)
accuracy_score(y_test, y_pred)

# Gradient boosting.
'''
This does the same as Adaboost but instead of iteratively changing the weights
of specific inputs it tries to fit a new predictor to the residual errors made
by the previous predictor.
'''

from sklearn.tree import DecisionTreeRegressor
np.random.seed(42)
X = np.random.rand(100, 1) - 0.5
y = 3*X[:, 0]**2 + 0.05 * np.random.randn(100)

# Fit a decision tree to the data.
tree_reg1= DecisionTreeRegressor(max_depth=2)
tree_reg1.fit(X, y)

# Train a 2nd tree on the residual errors made by the first predicter.
y2 = y - tree_reg1.predict(X)
tree_reg2 = DecisionTreeRegressor(max_depth=2)
tree_reg2.fit(X, y2)

# Iterate the solution again
y3 = y2 - tree_reg2.predict(X)
tree_reg3 = DecisionTreeRegressor(max_depth=2)
tree_reg3.fit(X, y3)

# Predict
X_new = np.array([[0.8]])
y_pred = sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))
print(y_pred)

# Another way to do it.
from sklearn.ensemble import GradientBoostingRegressor
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0)
gbrt.fit(X, y)

# Plot decision boundaries
def plot_predictions(regressors, X, y, axes, label=None, style='r-', data_style='b.', data_label=None):
    x1 = np.linspace(axes[0], axes[1], 500)
    y_pred = sum(regressor.predict(x1.reshape(-1, 1)) for regressor in regressors)
    plt.plot(X[:, 0], y, data_style, label=data_label)
    plt.plot(x1, y_pred, style, linewidth=2, label=label)
    if label or data_label:
        plt.legend(loc='upper center', fontsize=16)
    plt.axis(axes)
    
plt.figure(figsize=(11, 11))

plt.subplot(321)
plot_predictions([tree_reg1], X, y, axes=[-0.5, 0.5, -0.1, 0.8], 
                 label='$h_1(x_1)$', style='g-', data_label='training set')
plt.ylabel('$y$', fontsize=16, rotation=0)
plt.title('Residuals and tree predictions', fontsize=16)
plt.subplot(322)
plot_predictions([tree_reg1], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1)$", data_label="Training set")
plt.ylabel("$y$", fontsize=16, rotation=0)
plt.title("Ensemble predictions", fontsize=16)

plt.subplot(323)
plot_predictions([tree_reg2], X, y2, axes=[-0.5, 0.5, -0.5, 0.5], label="$h_2(x_1)$", style="g-", data_style="k+", data_label="Residuals")
plt.ylabel("$y - h_1(x_1)$", fontsize=16)

plt.subplot(324)
plot_predictions([tree_reg1, tree_reg2], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1) + h_2(x_1)$")
plt.ylabel("$y$", fontsize=16, rotation=0)

plt.subplot(325)
plot_predictions([tree_reg3], X, y3, axes=[-0.5, 0.5, -0.5, 0.5], label="$h_3(x_1)$", style="g-", data_style="k+")
plt.ylabel("$y - h_1(x_1) - h_2(x_1)$", fontsize=16)
plt.xlabel("$x_1$", fontsize=16)

plt.subplot(326)
plot_predictions([tree_reg1, tree_reg2, tree_reg3], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1) + h_2(x_1) + h_3(x_1)$")
plt.xlabel("$x_1$", fontsize=16)
plt.ylabel("$y$", fontsize=16, rotation=0)

plt.show()

