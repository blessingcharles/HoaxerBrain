import numpy as np 

def accuracy(y_true : np.ndarray , y_pred : np.ndarray):
    acc = np.sum(y_true == y_pred)/y_pred.shape[0]
    return acc

def logg_loss(y_true : np.ndarray , y_pred : np.ndarray, epsilon = 1e-15):

    """
        log loss  = -1/n * Summation[ ylog(y_pred) + (1-y)*log(1-y_pred) ]

        As log(0) tends to -inf we need to change it to very small like 0.00000001 and also log(1) to 0.99999
    """

    y_pred_new = np.array([max(ele , epsilon) for ele in y_pred ]) # for changing 0 to very small
    y_pred_new = np.array([min(ele , 1-epsilon) for ele in y_pred_new]) # for changing 1 to 1- 10^-15 

    result = -np.mean(y_true*np.log(y_pred_new) + (1-y_true)*np.log(1-y_pred_new))

    return result

def sigmoid(z : np.ndarray):
    exp = 1/(1+np.exp(-z))
    return exp

