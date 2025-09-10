# Copyright (c) 2025 Javad Komijani

"""
Gaussian KDE in PyTorch

This module provides a multivariate Gaussian Kernel Density Estimation (KDE)
implementation in PyTorch, designed to mimic the behavior of SciPy's
`gaussian_kde`. It supports arbitrary dimensions and flexible bandwidth
selection.

Key features:
- Supports 1D, 2D, and higher-dimensional datasets.
- Bandwidth selection methods: 'scott', 'silverman', or a custom scalar.
- Efficient evaluation using broadcasting and PyTorch operations.
- Designed to produce results comparable to SciPy's gaussian_kde.

Classes
-------
GaussianKDE
    Class implementing multivariate Gaussian KDE in PyTorch with an
    `evaluate` method for computing KDE values at given points.

Usage
-----
>>> import torch, numpy as np
>>> from gaussian_kde_torch import GaussianKDETorch
>>> data = np.random.randn(100, 2)
>>> kde = GaussianKDETorch(data, bw_method='scott')
>>> points = torch.randn(10, 2)
>>> kde_values = kde.evaluate(points)
"""

# pylint: disable=invalid-name, too-many-locals

import torch
import numpy as np

from scipy.stats import gaussian_kde
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt


__all__ = ["GaussianKDE"]


class GaussianKDE:
    """
    Multivariate Gaussian KDE in PyTorch, mimicking SciPy's gaussian_kde.

    Supports arbitrary dimensions. Bandwidth can be 'scott', 'silverman', or
    a float scaling factor.

    Parameters
    ----------
    dataset : torch.Tensor, shape (n_samples, d)
        Input data points.
    bw_method : 'scott', 'silverman', or float
        Bandwidth scaling method.
    """
    def __init__(self, dataset, bw_method='scott'):
        # Convert input to PyTorch tensor if necessary
        if not isinstance(dataset, torch.Tensor):
            dataset = torch.from_numpy(np.asarray(dataset)).float()
        self.dataset = dataset
        self.n, self.d = self.dataset.shape  # number of samples and dimensions
        self.set_bandwidth(bw_method)  # compute bandwidth matrix

    def set_bandwidth(self, bw_method='scott'):
        """
        Compute the bandwidth (scalar for 1D, matrix for d>1).

        Parameters
        ----------
        bw_method : {'scott', 'silverman'} or float
            Bandwidth selection method or scaling factor.
        """
        # Compute sample covariance
        if self.d == 1:
            # cov_data: scalar variance
            cov_data = torch.var(self.dataset, dim=0, unbiased=True)
        else:
            # cov_data: matrix of shape (d, d)
            cov_data = torch.from_numpy(np.cov(self.dataset.T.numpy())).float()

        # Determine bandwidth scaling factor
        if bw_method == 'scott':
            factor = self.n ** (-1.0 / (self.d + 4))
        elif bw_method == 'silverman':
            factor = (self.n * (self.d + 2) / 4.0) ** (-1.0 / (self.d + 4))
        elif isinstance(bw_method, (float, int)):
            factor = float(bw_method)
        else:
            raise ValueError("bw_method must be scott, silverman, or float")

        # Scale covariance to obtain bandwidth matrix
        bandwidth = cov_data * factor**2
        self.bandwidth = bandwidth
        if self.d == 1:
            self.inv_bandwidth = 1.0 / bandwidth  # scalar inverse
            self.det_bandwidth = bandwidth  # scalar determinant
        else:
            self.inv_bandwidth = torch.inverse(bandwidth)  # matrix inverse
            self.det_bandwidth = torch.det(bandwidth)  # determinant

    def evaluate(self, points):
        """
        Evaluate KDE at given points.

        Parameters
        ----------
        points : torch.Tensor or np.ndarray, shape (m, d)
            Points where the KDE is evaluated.

        Returns
        -------
        kde_values : torch.Tensor, shape (m,)
            Estimated density at each evaluation point.
        """
        # Convert to tensor if needed
        if not isinstance(points, torch.Tensor):
            points = torch.from_numpy(np.asarray(points)).float()

        m, d_check = points.shape  # pylint: disable=unused-variable
        if d_check != self.d:
            raise ValueError(
                f"Dimensionality mismatch: data has {self.d} dims, "
                f"points have {d_check} dims"
            )

        # Broadcast points and dataset for pairwise differences:
        #   points: shape (m, d)  -> m evaluation points, d dimensions
        #   self.dataset: shape (n, d)  -> n data samples, d dimensions
        diff = points.unsqueeze(1) - self.dataset.unsqueeze(0)  # shape (m,n,d)

        if self.d == 1:
            # 1D case: reduce last dimension (d=1) to scalar differences
            diff = diff.squeeze(-1)  # shape (m, n)
            exponent = -0.5 * diff**2 * self.inv_bandwidth
            norm_factor = self.n * torch.sqrt(2 * np.pi * self.det_bandwidth)
            kde_values = torch.exp(exponent).sum(dim=1) / norm_factor
        else:
            # Multivariate case: use einsum for quadratic form
            inv_b = self.inv_bandwidth
            det_b = self.det_bandwidth
            exponent = -0.5 * torch.einsum('mnd,de,mne->mn', diff, inv_b, diff)
            norm_factor = self.n * torch.sqrt((2 * np.pi)**self.d * det_b)
            kde_values = torch.exp(exponent).sum(dim=1) / norm_factor

        return kde_values

    def marginal_evaluate(self, points, dims):
        """
        Evaluate the marginal KDE on a subset of dimensions.

        Parameters
        ----------
        points : torch.Tensor or np.ndarray, shape (m, len(dims))
            Points in the marginal space
        dims : int or array-like
            Indices of dimensions to keep (marginalized over the others)

        Returns
        -------
        kde_values : torch.Tensor, shape (m,)
        """
        if not isinstance(points, torch.Tensor):
            points = torch.from_numpy(np.asarray(points)).float()
        if isinstance(dims, int):
            dims = (dims,)

        _, d_check = points.shape
        if d_check != len(dims):
            raise ValueError(
                f"Dimensionality mismatch: points have {d_check}, "
                f"expected {len(dims)}"
            )

        if self.d == d_check:
            # Marginal is the KDE itself
            return self.evaluate(points)

        dataset_marg = self.dataset[:, dims]
        diff = points.unsqueeze(1) - dataset_marg.unsqueeze(0)

        if len(dims) == 1:
            # 1D marginal from multivariate
            diff = diff.squeeze(-1)
            inv_b = 1.0 / self.bandwidth[dims[0], dims[0]]
            det_b = self.bandwidth[dims[0], dims[0]]
            exponent = -0.5 * diff**2 * inv_b
            norm_factor = self.n * torch.sqrt(2 * np.pi * det_b)
            kde_values = torch.exp(exponent).sum(dim=1) / norm_factor
        else:
            # Multivariate marginal
            inv_b = torch.inverse(self.bandwidth[np.ix_(dims, dims)])
            det_b = torch.det(self.bandwidth[np.ix_(dims, dims)])
            exponent = -0.5 * torch.einsum('mnd,de,mne->mn', diff, inv_b, diff)
            norm_factor = self.n * torch.sqrt((2 * np.pi)**len(dims) * det_b)
            kde_values = torch.exp(exponent).sum(dim=1) / norm_factor

        return kde_values


# =============================================================================
# Below are three test functions for 1D, 2D, and 3D
# ------------------- 1D KDE Test -------------------
def test_kde_1d(n_samples=1000):
    """
    Test and compare 1D Kernel Density Estimates using Torch and SciPy.

    Generates either a standard Gaussian or a multimodal dataset, fits
    KDE using both Torch and SciPy, and visualizes the comparison.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate from the Gaussian distribution.

    Notes
    -----
    - The function plots histogram + KDE curves for both unimodal and
      multimodal cases.
    - Prints the maximum absolute difference between Torch and SciPy KDE.
    """
    plt.figure(figsize=(12, 4))

    # Test both unimodal and multimodal cases
    for with_multimodal in [False, True]:
        # Generate 1D Gaussian data
        data_np = np.random.normal(0, 1, n_samples)

        # Introduce multimodality if requested
        if with_multimodal:
            data_np += np.sign(data_np) * 8

        # Convert to Torch tensor for KDE
        data_torch = torch.from_numpy(data_np).float().unsqueeze(1)

        # Define evaluation points over a wider range
        eval_points = torch.linspace(
            data_np.min() - 5, data_np.max() + 5, 500
        ).unsqueeze(1)

        # Torch KDE
        kde_torch = GaussianKDE(data_torch)
        vals_torch = kde_torch.evaluate(eval_points).numpy()

        # SciPy KDE
        kde_scipy = gaussian_kde(data_np)
        vals_scipy = kde_scipy.evaluate(eval_points.numpy().T)

        # Print max absolute difference
        print(f"1D max abs diff (multimodal={with_multimodal})",
              np.max(np.abs(vals_torch - vals_scipy)))

        # Plot histogram and KDEs
        ind = 2 if with_multimodal else 1
        plt.subplot(1, 2, ind)
        plt.hist(data_np, bins=50, density=True, alpha=0.5, color='gray')
        plt.plot(eval_points.numpy(), vals_torch, label="Torch KDE")
        plt.plot(eval_points.numpy(), vals_scipy, '--', label="SciPy KDE")
        plt.title(f"1D KDE Comparison\n(multimodal={with_multimodal})")
        plt.legend()

    plt.tight_layout()
    plt.show()


# ------------------- 2D KDE Test -------------------
def test_kde_2d():
    """
    Test and compare 2D Kernel Density Estimates using Torch and SciPy.

    Generates a 2D Gaussian dataset, fits KDE using both Torch and SciPy,
    and visualizes the density estimates with filled contours. Overlays
    exact PDF contours for reference.

    Notes
    -----
    - Torch and SciPy KDE are compared on a regular 2D grid.
    - Maximum absolute difference between the two KDEs is printed.
    - Filled contour plot (contourf) shows KDE density; red contour lines
      show exact PDF.
    """
    # ------------------- Data Generation -------------------
    mean_2d = [0, 0]
    cov_2d = [[1, 0.5], [0.5, 1]]
    data_2d = np.random.multivariate_normal(mean_2d, cov_2d, 500)
    data_torch_2d = torch.from_numpy(data_2d).float()

    # Grid for evaluation
    x = np.linspace(data_2d[:, 0].min() - 3, data_2d[:, 0].max() + 3, 100)
    y = np.linspace(data_2d[:, 1].min() - 3, data_2d[:, 1].max() + 3, 100)
    X, Y = np.meshgrid(x, y)
    grid_points = torch.from_numpy(
        np.column_stack([X.ravel(), Y.ravel()])
    ).float()

    # ------------------- Torch KDE -------------------
    kde_torch_2d = GaussianKDE(data_torch_2d)
    Z_torch = kde_torch_2d.evaluate(grid_points).numpy().reshape(100, 100)

    # ------------------- SciPy KDE -------------------
    kde_scipy_2d = gaussian_kde(data_2d.T)
    Z_scipy = kde_scipy_2d(grid_points.numpy().T).reshape(100, 100)

    print("2D max abs difference:", np.max(np.abs(Z_torch - Z_scipy)))

    # ------------------- Exact PDF Overlay -------------------
    def overlay_exact_pdf_contour_lines():
        """Compute and overlay top-density contour lines of the exact PDF."""
        exact_pdf = multivariate_normal(mean=mean_2d, cov=cov_2d)
        Z_exact = exact_pdf.pdf(
            np.column_stack([X.ravel(), Y.ravel()])
        ).reshape(100, 100)

        # Top-density contour levels
        max_val = Z_exact.max()
        levels = [0.05 * max_val, 0.7 * max_val]  # two main contours
        plt.contour(X, Y, Z_exact, levels=levels, colors='red', linewidths=0.5)

    # ------------------- Plot Torch KDE -------------------
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.contourf(X, Y, Z_torch, levels=50, cmap='viridis')
    plt.title("Torch KDE (filled) & Exact PDF (red lines)")
    plt.colorbar(label='Density')
    overlay_exact_pdf_contour_lines()

    # ------------------- Plot SciPy KDE -------------------
    plt.subplot(1, 2, 2)
    plt.contourf(X, Y, Z_scipy, levels=50, cmap='viridis')
    plt.title("SciPy KDE (filled) & Exact PDF (red lines)")
    plt.colorbar(label='Density')
    overlay_exact_pdf_contour_lines()

    plt.tight_layout()
    plt.show()


# ------------------- 3D KDE Test -------------------
def test_kde_3d(n_samples=500):
    """
    3D KDE test (Torch vs SciPy).

    - Generates 3D Gaussian data
    - Computes KDE on a 3D grid
    - Prints max absolute difference between Torch and SciPy KDE
    """
    # ------------------- Data -------------------
    mean_3d = [0, 0, 0]
    cov_3d = [[1, 0.5, 0.2],
              [0.5, 1, 0.3],
              [0.2, 0.3, 1]]
    data_3d = np.random.multivariate_normal(mean_3d, cov_3d, n_samples)
    data_torch_3d = torch.from_numpy(data_3d).float()

    # ------------------- Grid -------------------
    grid_pts = 30
    x = np.linspace(data_3d[:, 0].min() - 3, data_3d[:, 0].max() + 3, grid_pts)
    y = np.linspace(data_3d[:, 1].min() - 3, data_3d[:, 1].max() + 3, grid_pts)
    z = np.linspace(data_3d[:, 2].min() - 3, data_3d[:, 2].max() + 3, grid_pts)
    X, Y, Z = np.meshgrid(x, y, z)
    grid_points = torch.from_numpy(
        np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    ).float()

    # ------------------- KDE -------------------
    kde_torch_3d = GaussianKDE(data_torch_3d)
    values_torch = kde_torch_3d.evaluate(grid_points).numpy()

    kde_scipy_3d = gaussian_kde(data_3d.T)
    values_scipy = kde_scipy_3d(grid_points.numpy().T)

    # ------------------- Compare -------------------
    max_abs_diff = np.max(np.abs(values_torch - values_scipy))
    print("3D max absolute difference (Torch vs SciPy):", max_abs_diff)


if __name__ == '__main__':
    test_kde_1d()
    test_kde_2d()
    test_kde_3d()
