class BaseRegression:
    
    def __init__(self ,epochs : int = 1000 , lr : int = 0.01 , verbose : bool = False , bias : int = 0) -> None:
        self.epochs = epochs
        self.lr = lr
        self.verbose = verbose
        self.bias = bias
        self.cost_list = []
        self.epoch_list = []
        self.note_epoch = 10
