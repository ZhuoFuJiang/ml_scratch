import numpy as np
from utils.data_operation import calculate_covariance_matrix


class PCA():
    def transform(self, X, n_components):
        covariance_matrix = calculate_covariance_matrix(X)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx][:n_components]
        eigenvectors = np.atleast_1d(eigenvectors[:, idx][:, :n_components])
        X_transformed = X.dot(eigenvectors)
        return X_transformed
