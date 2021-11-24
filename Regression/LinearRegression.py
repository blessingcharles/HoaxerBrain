import numpy as np
import matplotlib.pyplot as plt

from Regression import BaseRegression

class LinearRegression(BaseRegression):
    
    def fit(self , X : np.ndarray , y : np.ndarray ):
        self.X : np.ndarray = X 
        self.y : np.ndarray = y

        self.total_samples = X.shape[0]
        self.total_features = X.shape[1]
        
        self.weights = np.ones(shape=(self.total_features))

        """
            applying batch gradient descent
            W = [w1,w2]
            y_pred = w1.x1 + w2.x2 + c 

            cf = 1/n*(y - y_pred)^2
            MSE = cf
            w_grad = d(MSE)/dw = -2/n[ X * ( y - y_pred )]

            W = W - lr * w_grad

        """

        for i in range(self.epochs):

            y_pred = np.dot(self.weights,self.X.T) + self.bias
            diff = self.y-y_pred
            cf = np.mean(np.square(diff))

            w_grad = -(2/self.total_samples)*(self.X.T.dot(diff))
            bias_grad = -(2/self.total_samples)*np.sum(diff)

            # print(self.weights - w_grad * self.lr)
            self.weights = self.weights - (self.lr * w_grad) 
            self.bias = self.bias - (self.lr * bias_grad)

            if i%self.note_epoch==0:
                self.cost_list.append(cf)
                self.epoch_list.append(i)
                
    def predict(self , X_test : np.ndarray):
        y_pred = np.dot(self.weights , X_test.T) + self.bias
        return y_pred

    def results(self , y_real):
        pass

    def actual_vs_predicted_graph(self , y_test , predicted_value ):
        plt.figure(figsize=(10,10))
        plt.scatter(y_test, predicted_value, c='crimson')
        # plt.yscale('log')
        # plt.xscale('log')

        p1 = max(max(predicted_value), max(y_test))
        p2 = min(min(predicted_value), min(y_test))
        plt.plot([p1, p2], [p1, p2], 'b-')
        plt.xlabel('True Values', fontsize=15)
        plt.ylabel('Predictions', fontsize=15)
        plt.axis('equal')
        plt.show()
