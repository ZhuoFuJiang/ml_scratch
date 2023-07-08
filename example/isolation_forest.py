from random import random
from unsupervised_learning.isolation_forest import IsolationForest


def main():
    print("Comparing average score of X and outlier's score...")
    # Generate a dataset randomly
    n = 100
    X = [[random() for _ in range(5)] for _ in range(n)]
    # Add outliers
    X.append([10]*5)
    # Train model
    clf = IsolationForest()
    clf.fit(X, n_samples=500)
    # Show result
    print("Average score is %.2f" % (sum(clf.predict(X)) / len(X)))
    print("Outlier's score is %.2f" % clf._predict(X[-1]))


if __name__ == "__main__":
    main()