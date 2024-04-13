import numpy as np
import pandas as pd
import random


class MyLogReg:
    def __init__(self, n_iter=10, learning_rate=0.1, metric=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.metric = metric
        self.y, self.X_no_ones = 0, 0

    def __str__(self):
        return self.weights.mean()

    def fit(self, X, y, verbose=False):
        X_copy = X.copy()
        array_ones = np.ones((X.shape[0], 1))
        X = np.hstack([array_ones, X])
        self.weights = np.array([1] * X.shape[1])
        for i in range(self.n_iter):
            y_predict = 1 / (1 + np.e ** -(X @ self.weights))
            grad = ((y_predict - y) @ X) / X.shape[0]
            self.weights = self.weights - self.learning_rate * grad
        self.y = y
        self.X_no_ones = X_copy

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

    def error_matrix(self, y_true, predict):
        y_true_list = list(y_true.astype(int))
        predict_list = list(predict.astype(int))
        data = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
        for i in range(len(predict_list)):
            if (predict_list[i] == 1) and (predict_list[i] == y_true_list[i]):
                data['TP'] = data.get('TP', 0) + 1
            elif (predict_list[i] == 1) and (predict_list[i] != y_true_list[i]):
                data['FP'] = data.get('FP', 0) + 1
            elif (predict_list[i] == 0) and (predict_list[i] == y_true_list[i]):
                data['TN'] = data.get('TN', 0) + 1
            else:
                data['FN'] = data.get('FN', 0) + 1
        return data

    def accuracy(self):
        predict = self.predict(self.X_no_ones)
        slov = self.error_matrix(self.y, predict)
        ac = (slov['TP'] + slov['TN']) / (slov['TP'] + slov['TN'] + slov['FP'] + slov['FN'])
        return ac

    def precision(self):
        predict = self.predict(self.X_no_ones)
        slov = self.error_matrix(self.y, predict)
        precis = slov['TP'] / (slov['TP'] + slov['FP'])
        return precis

    def recall(self):
        predict = self.predict(self.X_no_ones)
        slov = self.error_matrix(self.y, predict)
        rec = slov['TP'] / (slov['TP'] + slov['FN'])
        return rec

    def f1(self):
        predict = self.predict(self.X_no_ones)
        slov = self.error_matrix(self.y, predict)
        f_mer = (2 * self.precision() * self.recall()) / (self.precision() + self.recall())
        return f_mer

    def roc_auc(self):
        ...

    def get_best_score(self):
        if self.metric == 'accuracy':
            return self.accuracy()
        elif self.metric == 'precision':
            return self.precision()
        elif self.metric == 'recall':
            return self.recall()
        elif self.metric == 'f1':
            return self.f1()
        else:
            ...
