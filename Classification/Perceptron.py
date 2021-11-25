import numpy as np

class Perceptron:
    def __init__(self , epochs : int = 1000 , lr : int = 0.01 , random_state : int = 0 , bias : int = 0) -> None:
        self.epochs = epochs
        self.lr = lr
        self.random_state = random_state
        self.threshold = 0
        self.activation_func = self._activation
        self.bias = bias
       
    def fit(self , X : np.ndarray , y : np.ndarray):
        """
        
            assign random weights to each feature 
            loop for n epoch times and update the weight and bias using 
            w = learning rate *(y - y_pred)*x
            bias = learning rate *(y - y_pred)

            where y_pred = w1*x1 + w2*x2 .... +wn*xn + bias

        """
        n_samples , n_features = X.shape

        np.random.seed(self.random_state)
        self._weights = np.random.randn(n_features)

        for _ in range(self.epochs):
            for idx , x in enumerate(X):
                y_pred = self.activation_func(np.dot(x ,self._weights ) + self.bias)
                change = self.lr *(y[idx] - y_pred)

                self._weights = self._weights + change*x
                self.bias = self.bias + change


    def predict(self , x_test : np.ndarray):

        y_pred = np.dot(x_test , self._weights) + self.bias
        return self.activation_func(y_pred)
        
    def _activation(self , y : np.ndarray ):
        return np.where(y > self.threshold , 1 ,0)