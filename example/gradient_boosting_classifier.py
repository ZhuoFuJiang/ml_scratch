# -*- coding: utf-8 -*-
# @Time    : 2023/7/7 16:36
# @Author  : Zhuofu Jiang
# @FileName: gradient_boosting_classifier.py
# @Software: PyCharm


from __future__ import division, print_function
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# Import helper functions
from utils.data_manipulation import train_test_split
from utils.data_operation import accuracy_score
from deep_learning.loss_functions import CrossEntropy
from utils.misc import Plot
from supervised_learning.gradient_boosting import GradientBoostingClassifier


def main():

    print("-- Gradient Boosting Classification --")

    data = datasets.load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    clf = GradientBoostingClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)

    Plot().plot_in_2d(X_test, y_pred, title="Gradient Boosting", accuracy=accuracy, legend_labels=data.target_names)


if __name__ == "__main__":
    main()