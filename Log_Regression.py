import numpy as np
import pandas as pd
import random


class MyLogReg:
    def __init__(self, n_iter, learning_rate):
        self.n_iter = n_iter
        self.learning_rate = learning_rate

    def __str__(self):
        return self.weights.mean()

    def fit(self, X, y, verbose=False):
        array_ones = np.ones((X.shape[0], 1))
        X = np.hstack([array_ones, X])
        self.weights = np.array([1] * X.shape[1])
        for i in range(self.n_iter):
            y_predict = 1 / (1 + np.e ** -(X @ self.weights))
            grad = ((y_predict - y) @ X) / X.shape[0]
            self.weights = self.weights - self.learning_rate * grad

    def get_coef(self):
        return self.weights[1:]

    def predict(self, X):
        array_ones = np.ones((X.shape[0], 1))
        X = np.hstack([array_ones, X])
        y_predict = 1 / (1 + np.e ** -(X @ self.weights))
        y_predict[y_predict > 0.5] = 1
        y_predict[y_predict < 0.5] = 0
        y_predict = y_predict.astype(int)
        return y_predict

    def predict_proba(self, X):
        proba = self.predict(X)
        array_ones = np.ones((X.shape[0], 1))
        X = np.hstack([array_ones, X])
        y_predict = 1 / (1 + np.e ** -(X @ self.weights))
        return y_predict
