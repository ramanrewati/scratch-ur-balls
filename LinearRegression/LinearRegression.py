import numpy as np

class LinearRegression:
    def __init__(self,lr=0.01,n_iters=10000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self,X,y):
        n_samples , n_features =  np.shape(X)
        self.weights= np.zeros(n_features)
        self.bias = 0

        #grad descent
        for _ in range(self.n_iters):
            y_pred = np.dot(X,self.weights) + self.bias
            #calculating gradients
            dw = (1/n_samples) * np.dot(X.T,(y_pred-y))
            db = (1/n_samples) * np.sum(y_pred-y)
            #adjusting weights and bias
            self.weights -= self.lr*dw
            self.bias -= self.lr*db

    def predict(self,X):
        y_pred = np.dot(X,self.weights) + self.bias
        return y_pred


