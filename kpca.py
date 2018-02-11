import numpy as np
from scipy.linalg import eigh
from numpy.linalg import matrix_power
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise_kernels


class Kernel:

    def __init__(self, kernel_name, gamma, degree):
        self.kernel_name = kernel_name
        self.gamma = gamma
        self.degree = degree
        self.kernels = {
            'rbf': self.rbf_kernel,
            'poly': self.poly_kernel,
            'linear': self.linear_kernel
        }

    def get_kernel(self, x):
        return self.kernels.get(self.kernel_name)(x)

    def rbf_kernel(self, x):
        # Gaussian kernel
        sq_dists = pdist(x, 'sqeuclidean')
        mat_sq_dists = squareform(sq_dists)
        kernel = -self.gamma * mat_sq_dists
        np.exp(kernel, kernel)
        return kernel

    def poly_kernel(self, x):
        return matrix_power(x.dot(x.T) + self.gamma, self.degree)

    def linear_kernel(self, x):
        return x.dot(x.T)


class KernelPCA:

    def __init__(self, kernel='rbf', gamma=10, degree=3, n_components=2):
        self.kernel = Kernel(kernel_name=kernel, gamma=gamma, degree=degree)
        self.gamma = gamma
        self.degree = degree
        self.kernel_matrix = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.fitted = None
        self.n_components = n_components

    def _center_kernel(self):
        n = self.kernel_matrix.shape[0]
        one_n = np.ones((n, n)) / n
        self.kernel_matrix -= one_n.dot(self.kernel_matrix)
        self.kernel_matrix -= self.kernel_matrix.dot(one_n)
        self.kernel_matrix += one_n.dot(self.kernel_matrix).dot(one_n)

    def _kernel_centerer_from_scikit_learn(self):
        n_samples = self.kernel_matrix.shape[0]
        self.K_fit_rows_ = np.sum(self.kernel_matrix, axis=0) / n_samples
        self.K_fit_all_ = self.K_fit_rows_.sum() / n_samples
        K_pred_cols = (np.sum(self.kernel_matrix, axis=1) /
        self.K_fit_rows_.shape[0])[:, np.newaxis]
        self.kernel_matrix -= self.K_fit_rows_
        self.kernel_matrix -= K_pred_cols
        self.kernel_matrix += self.K_fit_all_

    def transform(self, x):
        if self.fitted is None:
            self.kernel_matrix = self.kernel.get_kernel(x)
            self._center_kernel()
        else:
            # used in contour plotting
            self.kernel_matrix = pairwise_kernels(self.fitted, x, metric='rbf', gamma=self.gamma).T
            self._kernel_centerer_from_scikit_learn()

        # find eigenvectors and eigenvalues of Kernel matrix
        if self.eigenvectors is None:
            self.eigenvalues, self.eigenvectors = eigh(
                self.kernel_matrix,
                eigvals=(self.kernel_matrix.shape[0] - self.n_components, self.kernel_matrix.shape[0] - 1)
            )
            # sort in descending order
            indices = self.eigenvalues.argsort()[::-1]
            self.eigenvalues = self.eigenvalues[indices]
            self.eigenvectors = self.eigenvectors[:, indices]
            self.fitted = x
        # return projection
        return np.dot(self.kernel_matrix, self.eigenvectors / np.sqrt(self.eigenvalues))
