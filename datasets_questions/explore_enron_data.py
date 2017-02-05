#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print(len(enron_data))
print(len(enron_data[enron_data.keys()[0]]))
print(enron_data[enron_data.keys()[0]])
print(sum([enron_data[k]['poi'] for k in enron_data.keys()]))
print(sorted(enron_data.keys()))
print(enron_data['PRENTICE JAMES']['total_stock_value'])
print(enron_data['SKILLING JEFFREY K']['exercised_stock_options'])

print(enron_data['SKILLING JEFFREY K']['total_payments'])
print(enron_data['FASTOW ANDREW S']['total_payments'])
print(enron_data['LAY KENNETH L']['total_payments'])

print(len([enron_data[k]['salary'] for k in enron_data.keys() if enron_data[k]['salary'] != 'NaN']))
print(len([enron_data[k]['email_address'] for k in enron_data.keys() if enron_data[k]['email_address'] != 'NaN']))
print([enron_data[k]['total_payments'] for k in enron_data.keys() if enron_data[k]['poi'] == True ])
print(len([enron_data[k]['total_payments'] for k in enron_data.keys() if enron_data[k]['total_payments'] == 'NaN']))

print(len([enron_data[k]['total_payments'] for k in enron_data.keys() if enron_data[k]['poi'] == True ]))

print(sorted([enron_data[k]['salary'] for k in enron_data.keys() if enron_data[k]['salary']!= 'NaN' and k != 'TOTAL']))
print([k for k in enron_data.keys() if enron_data[k]['salary'] == 1111258 or enron_data[k]['salary'] == 1072321])


