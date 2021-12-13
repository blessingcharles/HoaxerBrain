import numpy as np
from collections import Counter

from Classification.utils import entropy

class Node:
    
    def __init__(self , feature=None , threshold = None , left_child = None , right_child = None , * ,value=None) -> None:
       
        self.feature = feature 
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.value = value
    
    def is_leaf_node(self):
        return True if self.value is not None else False

class DecisionTree:

    def __init__(self , n_features : int = None , max_depth : int = 10 ,min_sample_to_split : int = 2  ,verbose : bool = False) -> None:
        """
            Decision tree based on id3 algorithm split on high information gain by iteratively analyzing each features and then by each
            unnique label in that feature .
            
        """
        self.n_features = n_features
        self.max_depth = max_depth
        self.root = None
        self.min_sample_to_split = min_sample_to_split  # minimum sample to split a given group
        self.verbose = verbose

    def fit(self , X : np.ndarray , y : np.ndarray) -> None:
        n_samples , n_features = X.shape
        self.n_features = n_features if self.n_features is None else min(self.n_features , n_features)
        self.root = self._build_decision_tree(X,y)
    
    def predict(self , x_test : np.ndarray) -> np.ndarray:
        y_pred = [self._traverse_decision(x , self.root) for x in x_test]

        return np.array(y_pred)

    def _traverse_decision(self , x : np.ndarray , node : Node):
        
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_decision(x,node.left_child)
        else :
            return self._traverse_decision(x,node.right_child)


    def _build_decision_tree(self , X : np.ndarray , y : np.ndarray , depth : int = 0) -> Node:

        n_samples , n_feat = X.shape
        unique_class = np.unique(y)
        n_labels= len(unique_class)

        # make a leaf node if it satisfies following condition
        if (n_labels == 1 or depth >= self.max_depth or n_samples < self.min_sample_to_split ):

            label = self.__most_common_label(y)
            leaf_node = Node(value=label)
            return leaf_node
        
        features_idx = np.random.choice(n_feat, self.n_features  , replace=False) #random array of feature index
        
        best_information_gain = -1
        best_feature_split = None
        best_threshold_split = None
        
        for feature in features_idx:
            X_c = X[:,feature]
            thresholds = np.unique(X_c)  #get all the unique values in that feature

            for threshold in thresholds:
                ig = self._get_information_gain(X_c ,y , threshold)
                if ig > best_information_gain:
                    best_information_gain = ig
                    best_feature_split = feature
                    best_threshold_split = threshold

        left_idx , right_idx = self._split_on_threshold(X[: , best_feature_split] , best_threshold_split)
        
        root = Node(feature=best_feature_split , threshold=best_threshold_split)
        
        root.left_child = self._build_decision_tree(X[ left_idx , :] , y[left_idx] , depth+1)
        root.right_child = self._build_decision_tree(X[right_idx , :] , y[right_idx] , depth+1)

        return root

    def _get_information_gain(self ,X_c : np.ndarray ,y: np.ndarray , threshold):
        
        """
            information gain = parent entrophy - child entrophy

            child entrophy = weighted sum of all entrophy based on threshold split
        """

        left_idx , right_idx = self._split_on_threshold(X_c , threshold)

        parent_entrophy = entropy(y)
        n_y = len(y) 
        n_left = len(left_idx)
        n_right = len(right_idx)
        entropy_left_split = entropy(y[left_idx])
        entropy_right_split = entropy(y[right_idx])

        child_entropy = ( (n_left/n_y)*entropy_left_split ) + ( (n_right/n_y)*entropy_right_split )

        return parent_entrophy - child_entropy


    def _split_on_threshold(self , X_c : np.ndarray , threshold):
        left_idx = np.argwhere(X_c <= threshold).flatten()
        right_idx = np.argwhere(X_c > threshold).flatten()

        return left_idx , right_idx

    def __most_common_label(self , samples):
        try:
            counter = Counter(samples)
            return counter.most_common(1)[0][0]
        except:
            return 0
