import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_moons, make_circles

from kpca import KernelPCA


def _scatter_plot(X1, Y1, X2, Y2, x_label, y_label, title):
    plt.title(title)
    plt.scatter(X1, Y1, c="red", s=20, edgecolor='k', alpha=0.8)
    plt.scatter(X2, Y2, c="blue", s=20, edgecolor='k', alpha=0.8)
    plt.xlabel(x_label)
    plt.ylabel(y_label)


def plot_contour(kpca):
    X1, X2 = np.meshgrid(np.linspace(-1.5, 1.5, 50), np.linspace(-1.5, 1.5, 50))
    X_grid = np.array([np.ravel(X1), np.ravel(X2)]).T
    Z_grid = kpca.get_out_of_sample_projection(X_grid)[:, 0].reshape(X1.shape)
    plt.contour(X1, X2, Z_grid, colors='grey', linewidths=1, origin='lower')


def plot_original_2_dim_data(X, y):
    _scatter_plot(X[y == 0, 0], X[y == 0, 1], X[y == 1, 0], X[y == 1, 1],
                  x_label='$x_1$', y_label='$x_1$', title='Original space')


def plot_projected_2_dim_data(X_kpca, y):
    _scatter_plot(X_kpca[y == 0, 0], X_kpca[y == 0, 1],
                  X_kpca[y == 1, 0], X_kpca[y == 1, 1],
                  x_label='1st principal component in space induced by $\phi$',
                  y_label='2nd component', title='Projection on 2 components')


def plot_projected_1_dim_data(X_kpca, y):
    _scatter_plot(X_kpca[y == 0, 0], np.zeros(X_kpca[y == 0, 0].shape),
                  X_kpca[y == 1, 0], np.zeros(X_kpca[y == 1, 0].shape),
                  x_label='1st principal component in space induced by $\phi$',
                  y_label='', title='Projection on 1 component')


if __name__ == '__main__':
    np.random.seed(0)
    X_1, y_1 = make_circles(n_samples=400, factor=.4, noise=.05)

    X_2, y_2 = make_moons(n_samples=100, noise=.01)

    kpca_1 = KernelPCA(kernel='rbf', gamma=10, degree=2, n_components=2)
    X_kpca_1 = kpca_1.get_projection(X_1)

    kpca_2 = KernelPCA(kernel='rbf', gamma=15, degree=2, n_components=2)
    X_kpca_2 = kpca_2.get_projection(X_2)

    plt.figure()
    # plt.subplot(3, 3, 1, aspect='equal')
    # # plot original space with contour
    # plot_original_2_dim_data(X_1, y_1)
    # plot_contour(kpca_1)
    #
    # # plot kpca projection
    # plt.subplot(3, 3, 2, aspect='equal')
    # plot_projected_2_dim_data(X_kpca_1, y_1)
    #
    # # plot kpca projection
    # plt.subplot(3, 3, 3, aspect='equal')
    # plot_projected_1_dim_data(X_kpca_1, y_1)

    plt.subplot(3, 3, 4, aspect='equal')
    # plot original space with contour
    plot_original_2_dim_data(X_2, y_2)
    plot_contour(kpca_2)

    # plot kpca projection
    plt.subplot(3, 3, 5, aspect='equal')
    plot_projected_2_dim_data(X_kpca_2, y_2)

    plt.subplot(3, 3, 6, aspect='equal')
    plot_projected_1_dim_data(X_kpca_2, y_2)

    plt.subplots_adjust(0.02, 0.10, 0.98, 0.94, 0.04, 0.35)
    plt.show()

    # Section:
    # data set - 3 circles
    plt.figure()
    X_11, y_11 = make_circles(n_samples=400, factor=.1, noise=.05)
    X_1 = np.concatenate((X_1, X_11[y_11 == 1, :]), axis=0)
    y_11[y_11 == 1] = 2
    y_1 = np.concatenate((y_1, y_11[y_11 == 2]), axis=0)
    X_kpca_1 = kpca_1.get_projection(X_1)

    # plot original data
    plt.scatter(X_1[y_1 == 0, 0], X_1[y_1 == 0, 1], c="blue", s=20, edgecolor='k', alpha=0.5)
    plt.scatter(X_1[y_1 == 1, 0], X_1[y_1 == 1, 1], c="green", s=20, edgecolor='k', alpha=0.5)
    plt.scatter(X_1[y_1 == 2, 0], X_1[y_1 == 2, 1], c="red", s=20, edgecolor='k', alpha=0.5)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.show()

    # plot feature space phi
    fig = plt.figure()
    ax = Axes3D(fig)
    z = sum(kpca_1.kernel.get_kernel(X_1))
    ax.scatter(X_1[:, 0], X_1[:, 1], z, c=y_1, cmap=plt.cm.rainbow)
    plt.show()

    # plot kPCA projection onto first 2 principal components
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.scatter(X_kpca_1[y_1 == 0, 0], X_kpca_1[y_1 == 0, 1], c="blue", s=20, edgecolor='k', alpha=0.5)
    plt.scatter(X_kpca_1[y_1 == 1, 0], X_kpca_1[y_1 == 1, 1], c="green", s=20, edgecolor='k', alpha=0.5)
    plt.scatter(X_kpca_1[y_1 == 2, 0], X_kpca_1[y_1 == 2, 1], c="red", s=20, edgecolor='k', alpha=0.5)
    plt.title("Projection by KPCA")
    plt.xlabel("1st principal component in space induced by $\phi$")
    plt.ylabel("2nd component")

    # plot PCA projection onto first 2 principal components
    plt.subplot(1, 2, 2)
    pca = PCA()
    X_pca = pca.fit_transform(X_1)
    plt.scatter(X_pca[y_1 == 0, 0], X_pca[y_1 == 0, 1], c="blue", s=20, edgecolor='k', alpha=0.5)
    plt.scatter(X_pca[y_1 == 1, 0], X_pca[y_1 == 1, 1], c="green", s=20, edgecolor='k', alpha=0.5)
    plt.scatter(X_pca[y_1 == 2, 0], X_pca[y_1 == 2, 1], c="red", s=20, edgecolor='k', alpha=0.5)
    plt.title("Projection by PCA")
    plt.xlabel("1st principal component")
    plt.ylabel("2nd component")
    plt.show()



    # # example with swiss roll
    # from sklearn.datasets.samples_generator import make_swiss_roll
    # from mpl_toolkits.mplot3d import Axes3D
    # X, color = make_swiss_roll(n_samples=800)
    #
    # fig = plt.figure(figsize=(7, 7))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.rainbow)
    # plt.title('Swiss Roll in 3D')
    # plt.show()
    #
    # kpca_3 = KernelPCA('rbf', gamma=0.1, n_components=2)
    # x_kpca_3 = kpca_3.get_projection(X)
    # plt.figure(figsize=(8, 6))
    # plt.scatter(x_kpca_3[:, 0], x_kpca_3[:, 1], c=color, cmap=plt.cm.rainbow)
    #
    # plt.title('First 2 principal components after RBF Kernel PCA')
    # plt.xlabel('PC1')
    # plt.ylabel('PC2')
    # plt.show()
