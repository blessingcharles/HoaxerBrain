class BaseRegression:
    
    def __init__(self , epochs : int , lr : int , verbose : bool , bias : int) -> None:
        self.epochs = epochs
        self.lr = lr
        self.verbose = verbose
        self.bias = bias
    
    def predict(self,y_test):
        return self._predict

    def _predict(self,y_test):
        raise NotImplementedError
