# -*- coding: utf-8 -*-
# @Time    : 2023/7/7 16:20
# @Author  : Zhuofu Jiang
# @FileName: decision_tree_classifier.py
# @Software: PyCharm


from __future__ import division, print_function
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import sys
import os

# Import helper functions
from utils.data_manipulation import train_test_split, standardize
from utils.misc import Plot
from utils.data_operation import mean_squared_error, calculate_variance, accuracy_score
from supervised_learning.decision_tree import ClassificationTree


def main():

    print("-- Classification Tree --")

    data = datasets.load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    clf = ClassificationTree()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)

    Plot().plot_in_2d(X_test, y_pred, title="Decision Tree", accuracy=accuracy, legend_labels=data.target_names)


if __name__ == "__main__":
    main()
