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
PROJECT_ROOT_DIR = r'C:/users/Tim/pythonscripts/MLbook'
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
                            DecisionTreeClassifier(), n_estimators=500,
                            max_samples=100, bootstrap=True, n_jobs=-1)
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
                            DecisionTreeClassifier(), n_estimators=500,
                            bootstrap=True, oob_score=True,
                            n_jobs=-1) # Shows out of the bag score.
bag_clf.fit(X_train, y_train)
bag_clf.oob_score_

from sklearn.metrics import accuracy_score
y_pred = bag_clf.predict(X_test)
accuracy_score(y_test, y_pred)