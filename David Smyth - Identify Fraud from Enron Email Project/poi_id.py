#!/usr/bin/python
# -*- coding: utf-8 -*-

# import statements
import matplotlib.pyplot
import pandas
import pickle
import sys

from feature_format import featureFormat, targetFeatureSplit
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tester import dump_classifier_and_data

sys.path.append("../tools/")

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# error: key  deferred_payments  not present
# Precision: 0.32915      Recall: 0.36750
features_list = ['poi', 'from_messages', 'from_poi_to_this_person', 'from_this_person_to_poi', 
                 'shared_receipt_with_poi', 'to_messages', 'salary', 'bonus', 'long_term_incentive', 
                 'loan_advances', 'other', 'expenses', 'director_fees', 'total_payments', 
                 'exercised_stock_options', 'restricted_stock', 'restricted_stock_deferred', 
                 'total_stock_value']

# Financial List
# Precision: 0.33125      Recall: 0.37100
#features_list = ['poi', 'salary', 'bonus', 'long_term_incentive', 'loan_advances', 'other', 
#'expenses', 'director_fees', 'total_payments', 'exercised_stock_options', 'restricted_stock', 
#'restricted_stock_deferred', 'total_stock_value']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
 
# Adapted from Pickling pandas df Discussion Forum
df = pandas.DataFrame.from_records(list(data_dict.values()))
employees = pandas.Series(list(data_dict.keys()))

# Set the index of df to be the employees series:
df.set_index(employees, inplace=True)

# You will have code here to add columns, i.e. new features,
# to the df, or remove rows, i.e. employees, from the df

# From Project fear! Strugging with Machine learning project Disucssion Forum
df.replace('NaN', 0, inplace = True)

# After you create features, the column names will be your new features
# Create a list of column names:
new_features_list = df.columns.values

# Create a dictionary from the dataframe
df_dict = df.to_dict('index')

# Compare the original dictionary 
# with the dictionary reconstructed from the dataframe:  
print df_dict == data_dict

### Task 2: Remove outliers

data = featureFormat(df_dict, ["salary", "bonus"])

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

# Take out, spreadsheet quirk
df_dict.pop('TOTAL', 0)
df_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)

# Adapted from Project: Identify Fraud from Enron Email Discussion Forum
# Leave in as valid data points
print("Salary and Bonus outliers: ")
for data in df_dict:
    if df_dict[data]['salary'] != 'NaN' and df_dict[data]['salary'] > 1000000:
        if df_dict[data]['bonus'] != 'NaN' and df_dict[data]['bonus'] > 5000000:
            print(data)

# of poi
n_poi = 0
for data in df_dict:
    if df_dict[data]['poi'] == True:
        n_poi += 1 
print "n_poi: ", n_poi

# of non-poi
n_non_poi = 0
for data in df_dict:
    if df_dict[data]['poi'] == False:
        n_non_poi += 1 
print "n_non_poi: ", n_non_poi

### Task 3: Create new feature(s)

# Adapted from Starting the final project Discussion Forum
# New features added
# Precision: 0.33214      Recall: 0.37250
for employee in df_dict:
    if df_dict[employee]['poi'] == True:
        restricted_stock_to_total_stock_value_ratio = float(df_dict[employee]['restricted_stock']
        )/float(df_dict[employee]['total_stock_value'])
        df_dict[employee].update({'restricted_stock_to_total_stock_value_ratio':restricted_stock_to_total_stock_value_ratio})
        
        bonus_to_total_payments_ratio = float(df_dict[employee]['bonus']
        )/float(df_dict[employee]['total_payments'])
        df_dict[employee].update({'bonus_to_total_payments_ratio':bonus_to_total_payments_ratio})
        
        expenses_to_total_payments_ratio = float(df_dict[employee]['expenses']
        )/float(df_dict[employee]['total_payments'])
        df_dict[employee].update({'expenses_to_total_payments_ratio':expenses_to_total_payments_ratio})
        
for employee in df_dict:
    if df_dict[employee]['poi'] == True:
        non_poi_restricted_stock_to_total_stock_value_ratio = float(data_dict[employee]['restricted_stock']
        )/float(data_dict[employee]['total_stock_value'])
        df_dict[employee].update({'non_poi_restricted_stock_to_total_stock_value_ratio':non_poi_restricted_stock_to_total_stock_value_ratio})
        
        non_poi_bonus_to_total_payments_ratio = float(data_dict[employee]['bonus']
        )/float(data_dict[employee]['total_payments'])
        df_dict[employee].update({'non_poi_bonus_to_total_payments_ratio':non_poi_bonus_to_total_payments_ratio})
        
        non_poi_expenses_to_total_payments_ratio = float(data_dict[employee]['expenses']
        )/float(data_dict[employee]['total_payments'])
        df_dict[employee].update({'non_poi_expenses_to_total_payments_ratio':non_poi_expenses_to_total_payments_ratio})
        
### Store to my_dataset for easy export below.
my_dataset = df_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size= 0.3,
                                                                            random_state = 42)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()

# svr = SVC(kernel='linear')
# clf = GridSearchCV(svr, parameters)

# clf = GaussianNB() GaussianNB()below results in Precision: 0.23989      Recall: 0.44500
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()

# clf = tree.DecisionTreeClassifier() below resulted in Precision: 0.24440      Recall: 0.24000
# from sklearn import tree
# clf = tree.DecisionTreeClassifier()

# clf = svm.SVC() resulted in "Precision or recall may be undefined due to a lack of true positive
#predicitons."

# from sklearn import svm
# clf = svm.SVC()

# clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0) below 
#resulted in Precision: 0.35585      Recall: 0.10800

# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)

# clif = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(features_train, labels_train) resulted in
#Precision: 0.35585      Recall: 0.10800

# from sklearn.neighbors import NearestNeighbors
# clif = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(features_train, labels_train)

# clf = AdaBoostClassifier(n_estimators=500), results in Precision: 0.32707      Recall: 0.24350
# clf = AdaBoostClassifier(n_estimators=400) results in Precision: 0.35458      Recall: 0.26150
# clf = AdaBoostClassifier(n_estimators=100) results in Precision: 0.35438      Recall: 0.26100
# clf = AdaBoostClassifier(n_estimators=10), results in Precision: 0.37017      Recall: 0.23950
# clf = AdaBoostClassifier(n_estimators=5), results in Precision: 0.36131      Recall: 0.20450
# clf = AdaBoostClassifier(n_estimators=4), results in Precision: 0.39049      Recall: 0.27100
# clf = AdaBoostClassifier(n_estimators=3), results in Precision: 0.46354      Recall: 0.17800
# clf = AdaBoostClassifier(n_estimators=2), results in Precision: 0.35699      Recall: 0.16600
# clf = AdaBoostClassifier(n_estimators=1), results in Precision: 0.26692      Recall: 0.07100

# clf = AdaBoostClassifier(n_estimators=4)

# Adapted from Project fear! Strugging with Machine learning project Discussion Forum    
PL = Pipeline(steps=[('MMS', MinMaxScaler()),
                     #__init__(feature_range=(0, 1), copy=True)
                       ('PCA', PCA()),
                       #__init__(n_components=None, copy=True, whiten=False, svd_solver=’auto’, tol=0.0, 
                                 #iterated_power=’auto’, random_state=None)
                       ('SKB', SelectKBest()),
                       #__init__(score_func=<function f_classif>, k=10)
                       ('GNB', GaussianNB())])
                        #__init__(priors=None)

# GridSearchCV parameters
param = {
# Number of top features to select, “all” bypasses selection
'SKB__k': range(6, 10),
# Wider range but life is short...
#'SKB__k': range(1, 18),
# Number of components to keep
#'PCA__n_components': range(3, 6),
# Can sometime improve predictive accuracy of downstream estimators
'PCA__whiten': [True, False]
}

# Default # of re-shuffling & splitting iterations, 10% of dataset included in test split,
# dataset not included in train split, use RandomState instance used by np.random as random number generator
SSS = StratifiedShuffleSplit(
n_splits=10, test_size=0.1, train_size=None, random_state=None
)

clf = GridSearchCV(
# The estimator object
PL,
# Parameter names as keys and lists of parameter settings as values
param_grid = param,
# A weighted average of precision and recall
scoring = 'f1',
# Number of jobs to run in parallel
n_jobs = 1,
# Cross-validation generator, utilizing StratifiedShuffleSplit
cv = SSS,
# Controls verbosity: the higher, the more messages
verbose = 1,
# Value to assign score if error in estimator fitting, if set to ‘raise’, error is raised
error_score = 'raise'
)

# Adapted from: https://stackoverflow.com/questions/44999289/
#print-feature-names-for-selectkbest-where-k-value-is-inside-param-grid-of-gridse

# Fit classifier
clf = clf.fit(features_train, labels_train)

print 'SElectKBest() scores...'
skb_step = clf.best_estimator_.named_steps['SKB']

# Get SelectKBest scores, rounded to 2 decimal places, name them "feature_scores"
feature_scores = ['%.2f' % elem for elem in skb_step.scores_ ]

# Get SelectKBest pvalues, rounded to 3 decimal places, name them "feature_scores_pvalues"
feature_scores_pvalues = ['%.3f' % elem for elem in  skb_step.pvalues_ 
]
# Get SelectKBest feature names, whose indices are stored in 'skb_step.get_support',
# create a tuple of feature names, scores and pvalues, name it "features_selected_tuple"
features_selected_tuple=[(features_list[i+1], feature_scores[i], 
feature_scores_pvalues[i]) for i in skb_step.get_support(indices=True)]

# Sort the tuple by score, in reverse order
features_selected_tuple = sorted(features_selected_tuple, key=lambda 
feature: float(feature[1]) , reverse=True)

print ' '
print 'Selected Features, Scores, P-Values'
print features_selected_tuple

# Create vector of predictions
#pred = clf.predict(features_test)
pred = clf.best_estimator_

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
    
# No. of correctly labeled items/all items
#accuracy = clf.score(features_test, labels_test)

# Probability to identify provided actually is
#recall = recall_score(labels_test, pred)

# Probability if identified, actually is
#precision = precision_score(labels_test, pred)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

#dump_classifier_and_data(clf, my_dataset, features_list)
dump_classifier_and_data(clf.best_estimator_, my_dataset, features_list)