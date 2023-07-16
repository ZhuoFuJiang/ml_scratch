import numpy as np
from utils.data_operation import calculate_covariance_matrix


class LDA():
    def __init__(self):
        self.w = None

    def fit(self, X, y):
        X1 = X[y == 0]
        X2 = X[y == 1]

        cov1 = calculate_covariance_matrix(X1)
        cov2 = calculate_covariance_matrix(X2)
        cov_tot = cov1 + cov2

        mean1 = X1.mean(0)
        mean2 = X2.mean(0)
        mean_diff = np.atleast_1d(mean1 - mean2)

        self.w = np.linalg.pinv(cov_tot).dot(mean_diff)

    def transform(self, X, y):
        self.fit(X, y)
        X_transform = X.dot(self.w)
        return X_transform

    def predict(self, X):
        # 这里理解为w的向量方向为u2指向u1(w的方向和u1-u2一致)，自然u1的方向和w的方向夹角小于90，而u2和w的方向夹角大于90
        # 如果y大于0，则类别为0，y小于0，则类别为1
        y_pred = []
        for sample in X:
            h = sample.dot(self.w)
            y = 1 * (h < 0)
            y_pred.append(y)
        return y_pred
