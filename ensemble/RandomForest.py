import numpy as np
from collections import Counter

from Classification.DecisionTree import DecisionTree

class RandomForest:
    def __init__(self , n_tress : int = 10 , min_samples_split : int = 2 ,max_depth : int = 100, n_features = None) -> None:
        self.n_tress = n_tress
        self.n_features = n_features
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.trees = []


    def bootstrap(self , X : np.ndarray , y: np.ndarray):

        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples , size=n_samples , replace=True)

        return X[idxs] , y[idxs]

    def fit(self , X : np.ndarray , y : np.ndarray):
        for _ in range(self.n_tress):
            tree = DecisionTree(n_features=self.n_features , max_depth=self.max_depth , min_sample_to_split=self.min_samples_split)
            x_sample , y_sample = self.bootstrap(X , y)
            tree.fit(x_sample ,y_sample)
            self.trees.append(tree)

    def predict(self , x_test : np.ndarray):
        
        predictions = np.array([tree.predict(x_test) for tree in self.trees ])
        predictions = np.swapaxes(predictions , 0 ,1)
        y_pred = [self.__most_common_label(labels) for labels in predictions]
        return np.array(y_pred)

    def __most_common_label(self , samples):
        counter = Counter(samples)
        return counter.most_common(1)[0][0]
