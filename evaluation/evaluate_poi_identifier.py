#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

# data = featureFormat(data_dict, features_list)
data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(features,labels)
acc = clf.score(features,labels)
print 'Accuracy: %f' % (acc)

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
clf.fit(features_train,labels_train)
acc = clf.score(features_test,labels_test)
print 'Accuracy: %f' % (acc)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

labels_pred = clf.predict(features_test)
print classification_report(labels_test, labels_pred)
print confusion_matrix(labels_test, labels_pred, labels=range(2))
print precision_score(labels_test,labels_pred)
print recall_score(labels_test,labels_pred)


predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1] 
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
print classification_report(true_labels, predictions)
print confusion_matrix(true_labels, predictions, labels=range(2))
print precision_score(true_labels, predictions)
print recall_score(true_labels, predictions)


