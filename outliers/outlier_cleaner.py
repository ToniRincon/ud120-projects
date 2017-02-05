#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    import numpy as np
    errors = np.absolute(predictions-net_worths)
    indexes = np.argsort(errors,axis=0)[0:len(ages)*0.9]
    cleaned_data = [(ages[i],net_worths[i],predictions[i]-net_worths[i]) for i in indexes]
       
    return cleaned_data

