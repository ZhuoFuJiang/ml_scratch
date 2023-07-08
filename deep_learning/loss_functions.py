# -*- coding: utf-8 -*-
# @Time    : 2023/7/7 15:34
# @Author  : Zhuofu Jiang
# @FileName: loss_functions.py
# @Software: PyCharm


import numpy as np

from utils.data_operation import accuracy_score


class Loss(object):
    def loss(self, y_true, y_pred):
        return NotImplementedError()

    def gradient(self, y, y_pred):
        raise NotImplementedError

    def acc(self, y, y_pred):
        return 0


class SquareLoss(Loss):
    def __init__(self):
        pass

    def loss(self, y, y_pred):
        return 0.5 * np.power((y - y_pred), 2)

    def gradient(self, y, y_pred):
        return -(y - y_pred)

    def hess(self, y, y_pred):
        return np.ones(np.shape(y_pred))


class CrossEntropy(Loss):
    def __init__(self):
        pass

    def loss(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -y * np.log(p) - (1 - y) * np.log(1 - p)

    def acc(self, y, p):
        return accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))

    def gradient(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -(y / p) + (1 - y) / (1 - p)

    def hess(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return (p * (1 - p) - (1 - 2 * p) * (p - y)) / (p ** 2 * (1 - p) ** 2)
