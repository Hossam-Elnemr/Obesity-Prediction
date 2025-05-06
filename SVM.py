import numpy as np

class SVM:

    def __init__(self, learning_rate = 0.001, lambda_parameter = 0.01, iterations = 1000 ):
        self.lr = learning_rate
        self.lp = lambda_parameter
        self.i = iterations
        self.w = None
        self.b = None

    def fit(self, X, y):
        samples, features = X.shape
        y_ = np.where(y <= 0, -1, 1)

        self.w = np.random.randn(features)
        self.b = 0

        for _ in range(self.i):
            for indx, x_i in enumerate(X):
                condition = y_[indx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lp * self.w)
                else:
                    self.w -= self.lr * (2 * self.lp * self.w - np.dot(x_i, y_[indx]))
                    self.b -= self.lr * y_[indx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)