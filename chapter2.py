# Chapter 2 Example
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import dataset
df = pd.read_csv(r'C:\Users\Tim\pythonscripts\datasets\CaliforniaHousing\cal_housing.csv')
df.columns = ['Longitude', 'Latitude', 'House_age', 'Total_rooms', 'Total_bedrooms',
              'Population', 'Households', 'Salary', 'Price']

# Create train and test sets
#from sklearn.model_selection import train_test_split
#train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

# Feature engineering an income category variable
df['Income_cat'] = np.ceil(df['Salary'] / 1.5)
df['Income_cat'].where(df['Income_cat'] < 5, 5.0, inplace=True)

# Histogram of df income categories
df['Income_cat'].hist(bins=5)
plt.xlabel('Income category')
plt.ylabel('Count')

# Look at ratios of each income category
df['Income_cat'].value_counts() / len(df)

# Stratified sampling
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df['Income_cat']):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]

# Plot data to gain insights :D
# Copy the train set
housing = strat_train_set.copy()

# Visualising geographical data
housing.plot(kind='scatter', x='Longitude', y='Latitude', alpha=0.1)

# Visualising geographical data with customisation
housing.plot(kind='scatter', x='Longitude', y='Latitude', alpha=0.4,
             s=housing['Population']/100, label='Population', figsize=(10, 7)
             , c='Price', cmap=plt.get_cmap('jet'), colorbar=True)
plt.legend()

# Looking for correlations
corr_matrix = housing.corr()
corr_matrix['Price'].sort_values(ascending=False)

# Using a scatter matrix to check for correlation
from pandas.tools.plotting import scatter_matrix
attributes = ['Price', 'Salary', 'Total_rooms', 'House_age']
scatter_matrix(housing[attributes], figsize=(12,8))

# Zoom in on Price vs Salary
housing.plot(kind='scatter', x='Salary', y='Price', alpha=0.1)

# Creating new features
housing['Rooms_per_household'] = housing['Total_rooms']/housing['Households']
housing['Bedrooms_per_room'] = housing['Total_bedrooms']/housing['Total_rooms']
housing['Population_per_household'] = housing['Population']/housing['Households']
# New correlation including new variables
corr_matrix = housing.corr()
corr_matrix['Price'].sort_values(ascending=False)

# Preparing data for ML algorithms
housing = strat_train_set.drop('Price', axis=1)
housing_labels = strat_train_set['Price'].copy()

# Dropping null values
'''
Can do it in these ways:
    housing.dropna(subset=['Total_bedrooms'])
    housing.drop('Total_bedrooms', axis=1)
    median = housing['Total_bedrooms'].median()
    housing['Total_bedrooms'].fillna(median, inplace=True)
'''
# You can also drop NA's using SKLEARN preprocessing.
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy='median')
housing_num = housing.copy()
imputer.fit(housing_num)
imputer.statistics_
housing_num.median().values

X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)

'''
# Encoding categorical labels
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
housing_cat = housing['ocean_proximity']
housing_cat_encoded = encoder.fit_transform(housing_cat)
housing_cat_encoded
# To look at the classes
print(encoder.classes_)

# We can also onehotencode instead
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
housing_cat_1hot.toarray()

# Or transform to binary classes like:
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
housing_cat_1hot = encoder.fit_transform(housing_cat)
'''

# Creating transformer class, does all of the work above but is reuseable :D
from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): 
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix]/X[:, household_ix]
        population_per_household = X[:, population_ix]/X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix]/X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

# Transformation feature scaling Pipleine
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([('imputer', Imputer(strategy='median')),
                         ('attribs_adder', CombinedAttributesAdder()),
                         ('std_scaler', StandardScaler())])

housing_num_tr = num_pipeline.fit_transform(housing_num)

# Allow a dataframe to be put directly into pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    '''
    Transforms the data by selecting the desired attributes, dropping the rest
    and converting the resulting DataFrame into an np array.
    '''
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

from sklearn.preprocessing import LabelBinarizer
num_attributes = list(housing_num)
#cat_attributes = ['Ocean_proximity']
# Pipelines for numerical and categorical variables :D
num_pipeline = Pipeline([('selector', DataFrameSelector(num_attributes)),
                         ('imputer', Imputer(strategy='median')),
                         ('attribs_adder', CombinedAttributesAdder()),
                         ('std_scaler', StandardScaler())])

#cat_pipeline = Pipeline([('selector', DataFrameSelector(cat_attribs)),
#                         ('label_binarizer', LabelBinarizer())])

'''
You can also merge these two pipelines with FeatureUnion
from sklearn.pipline import FeatureUnion
full_pipeline = FeatureUnion(transformer_list=[
('num_pipeline', num_pipeline),
('cat_pipeline', cat_pipeline)])
'''

# Final dataframe
housing_prepared = num_pipeline.fit_transform(housing)

# Select and train a model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# Test the model
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = num_pipeline.transform(some_data)
print('Predictions: ', lin_reg.predict(some_data_prepared))
print('Labels: ', list(some_labels))

# Calcing RMSE's
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rsme = np.sqrt(lin_mse)
print(lin_rsme)

# Trying a different model (decision tree regressor)
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rsme = np.sqrt(tree_mse)
print(tree_rsme)

# Evaluation with cross-validation :D
'''
K-fold cross validation, it randomly splits the training set into 10 subsets
called folds, then it trains and evaluates the decision tree model 10 times.
'''
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring='neg_mean_squared_error', cv=10)
tree_rsme_scores = np.sqrt(-scores)

# Display the scores
def display_scores(scores):
    print('Scores: ', scores)
    print('Mean: ', scores.mean())
    print('Standard Deviation: ', scores.std())

# Tree scores
display_scores(tree_rsme_scores)

# Linear model scores
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring='neg_mean_squared_error', cv=10)
lin_rsme_scores = np.sqrt(-scores)
display_scores(lin_rsme_scores)

# How about a random forest regressor
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rsme = np.sqrt(forest_mse)
print(forest_rsme)

# Display scores
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                         scoring='neg_mean_squared_error', cv=10)
forest_rsme_scores = np.sqrt(-forest_scores)
display_scores(forest_rsme_scores)

## How about SVMs?
#from sklearn.svm import SVR
#svr_rbf = SVR(kernel='rbf', C=1000, epsilon=0.01)
#svr_poly = SVR(kernel='poly', C=100, epsilon=0.01)
#svr_lin = SVR(kernel='linear', C=100, epsilon=0.01)
#
## Fit the svms
#svr_rbf.fit(housing_prepared, housing_labels)
#svr_poly.fit(housing_prepared, housing_labels)
#svr_lin.fit(housing_prepared, housing_labels)
#
## Display scores
#rbf_scores = cross_val_score(svr_rbf, housing_prepared, housing_labels,
#                         scoring='neg_mean_squared_error', cv=10)
#rbf_rsme_scores = np.sqrt(-rbf_scores)
#display_scores(rbf_rsme_scores)
#
#poly_scores = cross_val_score(svr_poly, housing_prepared, housing_labels,
#                         scoring='neg_mean_squared_error', cv=10)
#poly_rsme_scores = np.sqrt(-poly_scores)
#display_scores(poly_rsme_scores)
#
#linear_scores = cross_val_score(svr_lin, housing_prepared, housing_labels,
#                         scoring='neg_mean_squared_error', cv=10)
#linear_rsme_scores = np.sqrt(-linear_scores)
#display_scores(linear_rsme_scores)

# Fine tune model
# Grid search
from sklearn.model_selection import GridSearchCV
param_grid = [{'n_estimators': [3, 10, 30, 100], 'max_features': [2, 4, 6, 8, 12], 
             'bootstrap':[False], 'n_estimators':[3,10, 20], 'max_features':[2, 3, 4, 5]}]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)

# Display the best params
grid_search.best_params_

# To show the evaluation parameters
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score), params)
    
# Randomised search instead
#from sklearn.model_selection import RandomizedSearchCV
#forest_reg = RandomForestRegressor()
#param_grid = [{'n_estimators': [3:1000], 'max_features': [2:30], 'bootstrap':[False],
#               'max_features':[2:20]}]
#rand_grid_search = RandomizedSearchCV(forest_reg, param_grid, cv=5,
#                                      scoring='neg_mean_squared_error', n_iter=50)

# Ensemble methods - groups of models - Covered more later

# Analyse the best models and their erros
feature_importances = grid_search.best_estimator_.feature_importances_
print(feature_importances)

# Run model
final_model = grid_search.best_estimator_
X_test = strat_test_set.drop('Price', axis=1)
y_test = strat_test_set['Price'].copy()
X_test_prepared = num_pipeline.transform(X_test)
final_predictions = final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rsme = np.sqrt(final_mse)
print(final_rsme)

# Creating a full pipeline including the predictor
full_pipeline_with_predictor = Pipeline([
        ("preparation", num_pipeline()),
        ("linear", LinearRegression())
    ])

# Random grid search
from sklearn.model_selection import RandomizedSearchCV
param_grid = [{'n_estimators': [3, 10, 30, 100], 'max_features': [2, 4, 6, 8, 12], 
             'bootstrap':[False], 'n_estimators':[3,10, 20], 'max_features':[2, 3, 4, 5]}]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)

# Display the best params
grid_search.best_params_

# To show the evaluation parameters
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score), params)

# Creating a transformer which selects only the important attributes ;D
from sklearn.base import BaseEstimator, TransformerMixin

def indices_of_top_k(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])
    
class TopFeatureSelector(BaseEstimator, TransformerMixin):
    '''
    k is the amount of features we want to keep.
    '''
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k
    def fit(self, X, y=None):
        self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)
        return self
    def transform(self, X):
        return X[:, self.feature_indices_]

# New pipeline with feature selector included
prep_and_feature_selection_pipeline = Pipeline([
                                                ('preparation', num_pipeline),
                                                ('feature_selection', TopFeatureSelector(feature_importances, k))])

