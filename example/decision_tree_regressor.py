# -*- coding: utf-8 -*-
# @Time    : 2023/7/7 16:27
# @Author  : Zhuofu Jiang
# @FileName: decision_tree_regress.py
# @Software: PyCharm


from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils.data_manipulation import train_test_split, standardize
from utils.data_operation import mean_squared_error, calculate_variance, accuracy_score
from utils.misc import Plot
from supervised_learning.decision_tree import RegressionTree


def main():

    print("-- Regression Tree --")

    # Load temperature data
    data = pd.read_csv('../data/TempLinkoping2016.txt', sep="\t")

    time = np.atleast_2d(data["time"].values).T
    temp = np.atleast_2d(data["temp"].values).T

    X = standardize(time)        # Time. Fraction of the year [0, 1]
    y = temp[:, 0]  # Temperature. Reduce to one-dim

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    model = RegressionTree()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_pred_line = model.predict(X)

    # Color map
    cmap = plt.get_cmap('viridis')

    mse = mean_squared_error(y_test, y_pred)

    print("Mean Squared Error:", mse)

    # Plot the results
    # Plot the results
    m1 = plt.scatter(366 * X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(366 * X_test, y_test, color=cmap(0.5), s=10)
    m3 = plt.scatter(366 * X_test, y_pred, color='black', s=10)
    plt.suptitle("Regression Tree")
    plt.title("MSE: %.2f" % mse, fontsize=10)
    plt.xlabel('Day')
    plt.ylabel('Temperature in Celcius')
    plt.legend((m1, m2, m3), ("Training data", "Test data", "Prediction"), loc='lower right')
    plt.show()


if __name__ == "__main__":
    main()