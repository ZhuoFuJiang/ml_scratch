from __future__ import print_function
from sklearn import datasets
import numpy as np

from supervised_learning.multi_class_lda import MultiClassLDA
from utils.data_manipulation import normalize


def main():
    # Load the dataset
    data = datasets.load_iris()
    X = normalize(data.data)
    y = data.target

    # Project the data onto the 2 primary components
    multi_class_lda = MultiClassLDA()
    multi_class_lda.plot_in_2d(X, y, title="LDA")


if __name__ == "__main__":
    main()
