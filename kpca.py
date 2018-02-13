import numpy as np
from scipy.linalg import eigh
from numpy.linalg import matrix_power
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise_kernels


class Kernel:

    def __init__(self, kernel_type, gamma, degree):
        self.kernel_type = kernel_type
        self.kernels = {
            'rbf': self.rbf_kernel,
            'poly': self.poly_kernel,
            'linear': self.linear_kernel
        }
        self.kernels_params = {
            'rbf': {'gamma': gamma},
            'poly': {'gamma': gamma, 'degree': degree},
            'linear': {}
        }

    def get_kernel(self, x):
        """
        Return kernel matrix computed on x using specific kernel.
        Available kernel types: rbf, linear, poly
        """
        return self.kernels.get(self.kernel_type)(x, **self.get_specific_kernel_params())

    def get_specific_kernel_params(self):
        """
        Return specific kernel param. Ex. for `rbf` kernel - gamma
        """
        return self.kernels_params.get(self.kernel_type)

    @staticmethod
    def rbf_kernel(x, gamma):
        """
        Gaussian kernel
        :param x: data
        :return kernel matrix
        """
        sq_dists = pdist(x, 'sqeuclidean')
        mat_sq_dists = squareform(sq_dists)
        kernel = -gamma * mat_sq_dists
        np.exp(kernel, kernel)
        return kernel

    @staticmethod
    def poly_kernel(x, gamma, degree):
        """
        Polynomial kernel
        :param x: data
        :return: kernel matrix
        """
        return matrix_power(x.dot(x.T) + gamma, degree)

    @staticmethod
    def linear_kernel(x):
        """
        Linear kernel
        :param x: data
        :return kernel_matrix
        """
        return x.dot(x.T)


class KernelPCA:

    def __init__(self, kernel='rbf', gamma=10, degree=3, n_components=2):
        self.kernel = Kernel(kernel_type=kernel, gamma=gamma, degree=degree)
        self.gamma = gamma
        self.degree = degree
        self.fitted = None
        self.kernel_matrix = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.n_components = n_components

    def _center_kernel(self):
        """ Centralizing kernel matrix. Can be used for square kernel matrix only. """
        n = self.kernel_matrix.shape[0]
        one_n = np.ones((n, n)) / n
        self.kernel_matrix -= one_n.dot(self.kernel_matrix)
        self.kernel_matrix -= self.kernel_matrix.dot(one_n)
        self.kernel_matrix += one_n.dot(self.kernel_matrix).dot(one_n)

    def _kernel_centerer_from_scikit_learn(self):
        """ Kernel centerer taken from scikit-learn lib. Can be used for non-square kernel matrix. """
        n_samples = self.kernel_matrix.shape[0]
        self.K_fit_rows_ = np.sum(self.kernel_matrix, axis=0) / n_samples
        self.K_fit_all_ = self.K_fit_rows_.sum() / n_samples
        K_pred_cols = (np.sum(self.kernel_matrix, axis=1) /
        self.K_fit_rows_.shape[0])[:, np.newaxis]
        self.kernel_matrix -= self.K_fit_rows_
        self.kernel_matrix -= K_pred_cols
        self.kernel_matrix += self.K_fit_all_

    def get_projection(self, x):
        """ Compute projection of `x` onto the first `self.n_components` principal components. """
        self.kernel_matrix = self.kernel.get_kernel(x)
        self._center_kernel()

        # find eigenvectors and eigenvalues of Kernel matrix
        self.eigenvalues, self.eigenvectors = eigh(
            self.kernel_matrix,
            eigvals=(self.kernel_matrix.shape[0] - self.n_components, self.kernel_matrix.shape[0] - 1)
        )
        # sort in descending order
        indices = self.eigenvalues.argsort()[::-1]
        self.eigenvalues = self.eigenvalues[indices]
        self.eigenvectors = self.eigenvectors[:, indices]
        self.fitted = x
        print("Computed eigenvalues: {}".format(self.eigenvalues))
        # return projection
        return np.dot(self.kernel_matrix, self.eigenvectors / np.sqrt(self.eigenvalues))

    def get_out_of_sample_projection(self, new_data):
        """
        Compute projection of new data. Compute pairwise kernel matrix from already fitted and new data.
        Project new data onto existing principal components.
        """
        if self.fitted is None:
            raise Exception("KPCA is not fitted")
        self.kernel_matrix = pairwise_kernels(
            self.fitted,
            Y=new_data,
            metric=self.kernel.kernel_type,
            **self.kernel.get_specific_kernel_params()
        )
        self._kernel_centerer_from_scikit_learn()
        return np.dot(self.kernel_matrix.T, self.eigenvectors / np.sqrt(self.eigenvalues))
