#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi','salary','bonus'] # You will need to use more features

features_payments = ['salary','bonus','long_term_incentive','deferred_income','deferral_payments','loan_advances','other','expenses','director_fees','total_payments']
features_stock_value = ['exercised_stock_options','restricted_stock','restricted_stock_deferred','total_stock_value']
features_email = ['to_messages','from_messages','from_this_person_to_poi','from_poi_to_this_person']
features_receipt = ['shared_receipt_with_poi']
features_list = ['poi'] + features_payments + features_stock_value + features_email + features_receipt

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
	
### Task 2: Remove outliers

import pandas as pd
# We will use pandas many times in our project so turn this into a function:
def to_pandas(data_dict):
    df = pd.DataFrame(data_dict)
    #df = df.transpose()
    df = df.replace('NaN',0)
    #df.reset_index(level=0, inplace=True)
    # Renaming with df.rename(columns={'index': 'name'}) does not work
    # pandas bug??
    #columns = list(df.columns)
    #columns[0] = 'name'
    #df.columns = columns
    return(df)

df = to_pandas(data_dict)

print df.head()

for k in df:
    print dk

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

exit()

data_dict.pop( 'TOTAL')
data_dict.pop( 'LOCKHART EUGENE E')

data = featureFormat(data_dict, features_list[1:])

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

# Add ratio of POI messages to total.

features_financial = [
                "bonus",
                "deferral_payments",
                "deferred_income",
                "director_fees",
                "exercised_stock_options",
                "expenses",
                "loan_advances",
                "long_term_incentive",
                "other",
                "restricted_stock",
                "restricted_stock_deferred",
                "salary",
                "total_payments",
                "total_stock_value"
                ]

for name in data_dict:
    try:
        total_messages = data_dict[name]['from_messages'] + data_dict[name]['to_messages']
        poi_related_messages = data_dict[name]["from_poi_to_this_person"] +\
                                    data_dict[name]["from_this_person_to_poi"]
        poi_ratio = 1.* poi_related_messages / total_messages
        data_dict[name]['poi_ratio_messages'] = poi_ratio
    except:
        data_dict[name]['poi_ratio_messages'] = 'NaN'
    
    for feat in features_financial:
        try:
            data_dict[name][feat + '_squared'] = data_dict[name][feat]*data_dict[name][feat]
        except:
            data_dict[name][feat + '_squared'] = 'NaN'

my_dataset = data_dict

features_list = features_list + ['poi_ratio_messages'] + [feat + '_squared' for feat in features_financial]

print 'features_list:', features_list

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features, labels)


from sklearn.model_selection import cross_val_predict
predicted = cross_val_predict(clf, features, labels, cv=10)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

print classification_report(labels, predicted)
print confusion_matrix(labels, predicted, labels=range(2))
print precision_score(labels, predicted)
print recall_score(labels,predicted)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(features, labels)
predicted = cross_val_predict(clf, features, labels, cv=10)
print classification_report(labels, predicted)
print confusion_matrix(labels, predicted, labels=range(2))
print precision_score(labels, predicted)
print recall_score(labels,predicted)

from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators = 1000, random_state = 202, learning_rate = 1.0, algorithm = "SAMME.R")
clf.fit(features, labels)
predicted = cross_val_predict(clf, features, labels, cv=10)
print classification_report(labels, predicted)
print confusion_matrix(labels, predicted, labels=range(2))
print precision_score(labels, predicted)
print recall_score(labels,predicted)

exit()

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

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
