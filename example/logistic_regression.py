from __future__ import print_function
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

# Import helper functions
from utils.data_manipulation import make_diagonal, normalize, train_test_split
from utils.data_operation import accuracy_score
from deep_learning.activation_functions import Sigmoid
from utils.misc import Plot
from supervised_learning.logistic_regression import LogisticRegression


def main():
    # Load dataset
    data = datasets.load_iris()
    X = normalize(data.data[data.target != 0])
    y = data.target[data.target != 0]
    y[y == 1] = 0
    y[y == 2] = 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, seed=1)

    clf = LogisticRegression(gradient_descent=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Reduce dimension to two using PCA and plot the results
    Plot().plot_in_2d(X_test, y_pred, title="Logistic Regression", accuracy=accuracy)


if __name__ == "__main__":
    main()