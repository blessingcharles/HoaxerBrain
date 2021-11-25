import numpy as np

def entropy(y):
    """
    calculate the entropy of the given set of values
    e = Summation[ p(x) * log(1/p(x)) ] =>  -Summation[ p(x) * log(p(x)) ]
     
    """
    class_count = np.bincount(y) # returns bincount over the range of (y_min , y_max)

    Px = class_count/len(y)
    entropy = -np.sum([p*np.log2(p) for p in Px if p > 0])

    return entropy