import numpy as np
from pandas.core.indexing import convert_from_missing_indexer_tuple

from Regression.utils import logg_loss, sigmoid

class LogisticRegression:
    def __init__(self ,epochs : int = 1000 , lr : int = 0.01 , verbose : bool = False , bias : int = 0) -> None:
        self.epochs = epochs
        self.lr = lr
        self.verbose = verbose
        self.bias = bias
        self.cost_list = []
        self.epoch_list = []
        self.note_epoch = 10

    def fit(self , X_train : np.ndarray, y_train : np.ndarray):
        self.X : np.ndarray = X_train 
        self.y : np.ndarray = y_train

        self.total_samples = self.X.shape[0]
        self.total_features = self.X.shape[1]
        
        self.weights = np.ones(shape=(self.total_features))

        """
            applying loggloss cost function with gradient descent for it
            to reduce the loss

            log loss  = -1/n * Summation[ ylog(y_pred) + (1-y)*log(1-y_pred) ]

        """

        for i in range(self.epochs):
            y_pred = np.dot(self.weights , X_train.T) + self.bias

            y_pred = sigmoid(y_pred)
            cf = logg_loss(y_train,y_pred)

            cf_gd = np.dot(self.X.T,(y_pred-y_train))/y_train.shape[0]

            self.weights = self.weights - self.lr*cf_gd
            # self.bias = self.weights - self.lr*cf_gd

            if i%self.note_epoch:
                self.cost_list.append(cf)
                self.epoch_list.append(i)


    def predict(self , x_test):
        y_pred = np.dot(self.weights , x_test.T) + self.bias

        y_pred = sigmoid(y_pred)
        
        for i in range(len(y_pred)):
            if y_pred[i] < 0.5 :
                y_pred[i] = 0
            else :
                y_pred[i] = 1

        return y_pred