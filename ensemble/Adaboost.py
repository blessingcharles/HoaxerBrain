from typing import List
import numpy as np


class DecisionStump:
    def __init__(self , feature_idx : int = None , threshold :any = None , polarity : int = 1 , amount_to_say : int = None) -> None:
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.polarity = polarity
        self.amount_to_say = amount_to_say

    def predict(self , X : np.ndarray):
        X_c = X[: , self.feature_idx]
        n_sample = X.shape[0]
        y_pred = np.ones(n_sample)

        if self.polarity == 1:
            y_pred[X_c < self.threshold] = -1
        else :
            y_pred[X_c > self.threshold] = -1
        
        return y_pred

            
class Adaboost:
    
    def __init__(self , n_classifiers : int = 5,verbose : bool = False ) -> None:
        """
            Initialise weights to be 1/N
            find a best feature to split
            Amount of say = 0.5 * log[1-TE/TE]   {TE = total error(summation of weights of sample)}
            We decrease the weights of sample which are correctly classified by W = W/* e[(-Amount of say) * y * y_pred] 
                                                                        --> divide by Z = summation of updated weights(to normalize everthing from 0 to 1)
    
        """

        self.n_classifiers = n_classifiers
        self.verbose = verbose
        self.classifiers = []

    def fit(self , X : np.ndarray , y : np.ndarray ):
        n_samples , n_features = X.shape
        self.X = X
        # initialise the error weights of each sample to 1/n_samples
        w = np.full(n_samples , (1/n_samples))

        """ 
            build n decision stumpy classifiers by
            finding a best feature by iterating through each feature and try each unique threshold splitting and split 
            which give less error 
        """

        for _ in range(self.n_classifiers):

            min_error = float('inf')
            stump = DecisionStump()

            for feature_idx in range(n_features):
                X_c = self.X[: , feature_idx]
                unique_features = np.unique(X_c)
            
                for threshold in unique_features:
                    p = 1

                    predictions = np.ones(n_samples)
                    predictions[X_c < threshold] = -1
                    total_error   = sum(w[predictions != y])

                    # # if total error is greater than 0.5 reverse the polarity the left classified becomes 1
                    if total_error > 0.5:
                        p = -1   
                        total_error = 1 - total_error

                    if total_error < min_error:
                        min_error = total_error
                        stump.feature_idx = feature_idx
                        stump.threshold = threshold
                        stump.polarity = p

            # change the error for the next stump to get more accurate on the sample which is currently wrongly classified
            
            EPS = 1e-15           # to avoid underflow
            stump.amount_to_say = 0.5 * np.log((1-total_error + EPS)/(total_error+EPS))
            y_pred = stump.predict(X)       

            w = w*np.exp(-stump.amount_to_say*y*y_pred)
            # to normalise between 0 to 1 of sample error weights
            w = w/np.sum(w)

            self.classifiers.append(stump)

    def predict(self , X):
        stumps_pred = [stump.amount_to_say * stump.predict(X) for stump in self.classifiers]
        y_pred = np.sum(stumps_pred , axis=0)
        y_pred = np.sign(y_pred)
        return y_pred

