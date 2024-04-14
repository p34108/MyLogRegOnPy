import numpy as np
import pandas as pd
import random


class MyLogReg:
    def __init__(self, n_iter=10, learning_rate=0.1, metric=None, reg=None, l1_coef=0, l2_coef=0, sgd_sample=None,
                 random_state=42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.metric = metric
        self.y, self.X_no_ones = 0, 0
        self.reg = reg
        self.l1_coef, self.l2_coef = l1_coef, l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state
        if isinstance(self.learning_rate, float):
            self.flag = 0
        else:
            self.f = self.learning_rate
            self.flag = 1

    def __str__(self):
        return self.weights.mean()

    def fit(self, X, y, verbose=False):
        random.seed(self.random_state)
        X_for_d = X.copy()
        X_copy = X.copy()
        array_ones = np.ones((X.shape[0], 1))
        X = np.hstack([array_ones, X])
        self.weights = np.array([1] * X.shape[1])
        for i in range(1, self.n_iter + 1):
            if self.flag:
                self.learning_rate = self.f(i)
            if self.sgd_sample:
                if isinstance(self.sgd_sample, float):
                    index_X = random.sample(range(X_for_d.shape[0]), int(np.round(self.sgd_sample * X_for_d.shape[0])))
                    X_new = X_for_d.iloc[index_X]
                    y_new = y.iloc[index_X]
                else:
                    index_X = random.sample(range(X_for_d.shape[0]), self.sgd_sample)
                    X_new = X_for_d.iloc[index_X]
                    y_new = y.iloc[index_X]
                array_ones = np.ones((X_new.shape[0], 1))
                X_new = np.hstack([array_ones, X_new])
                y_predict = 1 / (1 + np.e ** -(X_new @ self.weights))
                self.gradient_descent(X_new, y_predict, y_new)
            else:
                y_predict = 1 / (1 + np.e ** -(X @ self.weights))
                self.gradient_descent(X, y_predict, y)
        self.y = y
        self.X_no_ones = X_copy

    def gradient_descent(self, X, y_predict, y):
        if self.reg == 'l1':
            grad = ((y_predict - y) @ X) / X.shape[0] + self.l1_coef * np.sign(self.weights)
            self.weights = self.weights - self.learning_rate * grad
        elif self.reg == 'l2':
            grad = ((y_predict - y) @ X) / X.shape[0] + self.l2_coef * 2 * self.weights
            self.weights = self.weights - self.learning_rate * grad
        elif self.reg == 'elasticnet':
            grad = ((y_predict - y) @ X) / X.shape[0] + self.l1_coef * np.sign(
                self.weights) + self.l2_coef * 2 * self.weights
            self.weights = self.weights - self.learning_rate * grad
        else:
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
        predict_class = self.y
        predict_pr = self.predict_proba(self.X_no_ones)
        data = {'p': predict_pr, 'class': predict_class}
        X = pd.DataFrame(data)
        X = X.sort_values(by='p', ascending=False).reset_index(drop=True)
        p = list(X['p'])
        cl = list(X['class'])
        array = []
        for i in range(len(p)):
            if cl[i] == 0:
                sp = cl[:i]
                s = sp.count(1)
                c = list(X[(X['p'] == p[i]) & (X['class'] == 1)]['class'])
                if c:
                    array.append((len(c) + s) / 2)
                else:
                    array.append(s)
        return np.round(sum(array) / (cl.count(1) * cl.count(0)), 10)

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
            return self.roc_auc()
