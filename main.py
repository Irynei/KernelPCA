import numpy as np
from sklearn.datasets import make_moons, make_circles
import matplotlib.pyplot as plt
from kpca import KernelPCA

np.random.seed(0)
X, y = make_moons(n_samples=100, noise=.05)
# X, y = make_circles(n_samples=400, factor=.3, noise=.05)

kpca = KernelPCA(kernel='rbf', gamma=10)
X_kpca = kpca.transform(X)


def scatter_plot(X, reds, blues, x_label, y_label, title):
    plt.title(title)
    plt.scatter(X[reds, 0], X[reds, 1], c="red", s=20, edgecolor='k')
    plt.scatter(X[blues, 0], X[blues, 1], c="blue", s=20, edgecolor='k')
    plt.xlabel(x_label)
    plt.ylabel(y_label)


def plot_contour():
    X1, X2 = np.meshgrid(np.linspace(-1.5, 1.5, 50), np.linspace(-1.5, 1.5, 50))
    X_grid = np.array([np.ravel(X1), np.ravel(X2)]).T
    Z_grid = kpca.transform(X_grid)[:, 0].reshape(X1.shape)
    plt.contour(X1, X2, Z_grid, colors='grey', linewidths=1, origin='lower')


if __name__ == '__main__':
    plt.figure()
    plt.subplot(2, 2, 1, aspect='equal')
    # plot original space with contour
    scatter_plot(X, y == 0, y == 1, x_label='$x_1$', y_label='$x_1$', title='Original space')
    plot_contour()

    # plot kpca projection
    plt.subplot(2, 2, 2, aspect='equal')
    scatter_plot(X_kpca, y == 0, y == 1,
                 x_label='1st principal component in space induced by $\phi$',
                 y_label='2nd component', title='2nd component')

    plt.subplots_adjust(0.02, 0.10, 0.98, 0.94, 0.04, 0.35)
    plt.show()
