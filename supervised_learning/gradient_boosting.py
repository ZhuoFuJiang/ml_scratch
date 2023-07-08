# -*- coding: utf-8 -*-
# @Time    : 2023/7/4 17:07
# @Author  : Zhuofu Jiang
# @FileName: gradient_boosting.py
# @Software: PyCharm


import numpy as np
import progressbar

from supervised_learning.decision_tree import RegressionTree
from utils.data_manipulation import to_categorical
from utils.misc import bar_widgets
from deep_learning.loss_functions import SquareLoss, CrossEntropy


class GradientBoosting(object):
    def __init__(self, n_estimators, learning_rate, min_samples_split, min_impurity, max_depth, regression):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.regression = regression
        self.bar = progressbar.ProgressBar(widgets=bar_widgets)

        self.loss = SquareLoss()
        if not self.regression:
            self.loss = CrossEntropy()

        self.trees = []
        # 梯度提升做分类的时候，其实是在拟合每个类别的对数几率
        for _ in range(n_estimators):
            tree = RegressionTree(
                min_samples_split=min_samples_split,
                min_impurity=min_impurity,
                max_depth=max_depth
            )
            self.trees.append(tree)

    def fit(self, X, y):
        # 每棵树只拟合上个轮次每个样本的预测值对损失函数的梯度
        # y_pred = np.full(np.shape(y), np.mean(y, axis=0))
        y_pred = np.zeros_like(y)
        for i in self.bar(range(self.n_estimators)):
            gradient = self.loss.gradient(y, y_pred)
            self.trees[i].fit(X, gradient)
            update = self.trees[i].predict(X)
            # 因为拟合的是负梯度
            y_pred -= np.multiply(self.learning_rate, update)

    def predict(self, X):
        y_pred = np.array([])
        for tree in self.trees:
            update = tree.predict(X)
            update = np.multiply(self.learning_rate, update)
            y_pred = -update if not y_pred.any() else y_pred - update

        if not self.regression:
            y_pred = np.exp(y_pred) / np.expand_dims(np.sum(np.exp(y_pred), axis=1), axis=1)
            y_pred = np.argmax(y_pred, axis=1)
        return y_pred


class GradientBoostingRegressor(GradientBoosting):
    def __init__(self, n_estimators=200, learning_rate=0.5, min_samples_split=2,
                 min_var_red=1e-7, max_depth=4, debug=False):
        super(GradientBoostingRegressor, self).__init__(n_estimators=n_estimators,
                                                        learning_rate=learning_rate,
                                                        min_samples_split=min_samples_split,
                                                        min_impurity=min_var_red,
                                                        max_depth=max_depth,
                                                        regression=True)


class GradientBoostingClassifier(GradientBoosting):
    def __init__(self, n_estimators=200, learning_rate=.5, min_samples_split=2,
                 min_info_gain=1e-7, max_depth=2, debug=False):
        super(GradientBoostingClassifier, self).__init__(n_estimators=n_estimators,
                                                         learning_rate=learning_rate,
                                                         min_samples_split=min_samples_split,
                                                         min_impurity=min_info_gain,
                                                         max_depth=max_depth,
                                                         regression=False)

    def fit(self, X, y):
        y = to_categorical(y)
        super(GradientBoostingClassifier, self).fit(X, y)
