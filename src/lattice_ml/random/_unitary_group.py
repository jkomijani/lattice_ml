# Copyright (c) 2021-2026 Javad Komijani

"""Classes and functions to generate random (special) unitary matrices."""
# Moved here from `normflow.lib.stats.unitary_group`, which now re-exports
# these for backward compatibility.

from math import log, lgamma, pi  # lgamma: log gamma

import torch
import numpy as np

from lattice_ml.linalg import haar_qr
from ._ginibre_dist import GinibreCMatrixDist


__all__ = [
    "rand_sun_group_like",
    "rand_diagonal_sun_group_like",
    "DiagonalSUnGroup",
    "SUnGroup",
    "UnGroup",
    "U1Group",
]


# =============================================================================
def rand_sun_group_like(x: torch.Tensor) -> torch.Tensor:
    """Return a tensor of random SU(n) matrices with the same shape as `x`.

    Each matrix in the last two dimensions is drawn independently and uniformly
    with respect to the Haar measure on SU(n), following the same construction
    as `normflow`'s `SUnGroup`.

    Parameters
    ----------
    x : torch.Tensor
        A complex tensor of shape (..., n, n); only its shape, dtype, and
        device are used to determine those of the output.

    Returns
    -------
    torch.Tensor
        A tensor of shape (..., n, n) of random SU(n) matrices.
    """
    assert x.shape[-2] == x.shape[-1], "x must have shape (..., n, n)"
    n = x.shape[-1]
    ginibre = torch.randn(x.shape, dtype=x.dtype, device=x.device)
    q = haar_qr(ginibre, q_only=True)
    det = torch.linalg.det(q).unsqueeze(-1).unsqueeze(-1)
    return q / torch.pow(det, 1 / n)


# =============================================================================
def rand_diagonal_sun_group_like(x: torch.Tensor) -> torch.Tensor:
    """Return a tensor of random diagonal SU(n) matrices, shaped like `x`.

    Each output matrix lies in the maximal torus of SU(n): the first `n - 1`
    phases are drawn i.i.d. uniform on the circle, and the last is fixed to
    minus their sum so that `det = 1`.

    Note that this is *not* the eigenvalue distribution of a Haar-random SU(n)
    matrix as produced by `rand_sun_group_like`: those eigenvalues repel each
    other under the Weyl integration formula (density proportional to the
    squared Vandermonde determinant, as in the circular unitary ensemble),
    whereas the phases drawn here are independent.

    Parameters
    ----------
    x : torch.Tensor
        A tensor of shape (..., n, n); only its shape, dtype, and device
        are used to determine those of the output (the output is always
        complex-valued, matching the floating-point precision of `x`).

    Returns
    -------
    torch.Tensor
        A tensor of shape (..., n, n) of random diagonal SU(n) matrices.
    """
    assert x.shape[-2] == x.shape[-1], "x must have shape (..., n, n)"
    n = x.shape[-1]
    free = torch.rand(*x.shape[:-2], n-1, dtype=x.real.dtype, device=x.device)
    last = -free.sum(dim=-1, keepdim=True)
    angles = 2 * torch.pi * torch.cat([free, last], dim=-1)
    return torch.diag_embed(torch.exp(1j * angles))


# =============================================================================
class UnGroup(GinibreCMatrixDist):
    """Generate random unitary matrices, i.e. random U(n).

    Simliar to `[scipy.stats.unitary_group]`_, we follow `[Mezzadri]`_ to
    generate randam unitary matrices.

    Parameters
    ----------
    n : int
        Specifies the dimension n of the U(n) matrices.

    shape : Tuple[int] | None (optional, default=None)
        Specifing the shape of tensor of random unitary matrices.
        Each sample would be of a tensor of size (*shape, n, n),
        where the last two dimensions construct unitary matrices.
        If None, shape will be set to (); shape can accept () too.

    .. _[scipy.stats.unitary_group]:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.unitary_group.html

    .. _[Mezzadri]:
        F. Mezzadri,
        "How to generate random matrices from the classical compact groups",
        :arXiv:`math-ph/0609050`.
    """

    def __init__(self, n, shape=None, drop_constant_log_prob=False):

        if shape is None:
            shape = ()

        super().__init__(n=n, shape=shape)

        self.drop_constant_log_prob = drop_constant_log_prob

        self.log_group_vol = self.calc_log_group_volume(n)
        self.log_tot_vol = self.log_group_vol + log(np.prod(shape))

    def sample(self, size=(1,)):  # this is the `sample` of dist (not prior)
        """Draw random samples."""
        samples = super().sample(size)  # Ginibre GL(n, C) matrices
        return haar_qr(samples, q_only=True)  # Unitary matrices

    def log_prob(self, x):
        """Return log_prob of each matrix."""  # one value for each matrix
        lat_shape = x.shape[:-2]  # one log_prop for each matrix
        if self.drop_constant_log_prob:
            return torch.zeros(lat_shape, device=x.device)
        return torch.zeros(lat_shape, device=x.device) - self.log_group_vol

    @staticmethod
    def calc_log_group_volume(n):
        """Return volume of U(n); we use eq (5.16) of `[Boya et. al.]`_

        .. _[Boya et. al.]:
            "Volumes of Compact Manifolds", arXiv:`math-ph/0210033`
        """
        logc = log(n) + (n+1) * log(2) + (n**2 + n) * log(pi)
        return 0.5 * logc + sum(-lgamma(1+k) for k in range(1, n))


# =============================================================================
class SUnGroup(UnGroup):
    """Generate random special unitary matrices, i.e. random SU(n).

    As a subgroup of unitary matrices, this class uses the `UnGroup` class
    to generate random SU(n) matrices.
    """

    def sample(self, size=(1,)):  # this is the `sample` of dist (not prior)
        """Draw random samples."""
        samples = super().sample(size)
        det = torch.linalg.det(samples).unsqueeze(-1).unsqueeze(-1)
        # Remarks:
        # 1. det \in U(1)
        # 2. det^(1/n) covers only 1/n-th of U(1) -> volume 1/n-th of U(1)
        # 3. determinant of det^(1/n) I_{n x n} covers U(1); stretching fac. n
        return samples / torch.pow(det, 1/self.n)

    @staticmethod
    def calc_log_group_volume(n):
        """Return volume of U(n); we use eq (5.13) of `[Boya et. al.]`_

        .. _[Boya et. al.]:
            "Volumes of Compact Manifolds", arXiv:`math-ph/0210033`
        """
        logc = log(n) + (n-1) * log(2) + (n**2 + n - 2) * log(pi)
        return 0.5 * logc + sum(-lgamma(1+k) for k in range(1, n))


# =============================================================================
class DiagonalSUnGroup:
    """Generate random diagonal SU(n) matrices, uniform on the maximal torus.

    Each output matrix lies in the maximal torus of SU(n): the first `n - 1`
    phases are drawn i.i.d. uniform on the circle, and the last is fixed to
    minus their sum so that `det = 1`.

    Note that this is *not* the eigenvalue distribution of a Haar-random SU(n)
    matrix as produced by `rand_sun_group_like`: those eigenvalues repel each
    other under the Weyl integration formula (density proportional to the
    squared Vandermonde determinant, as in the circular unitary ensemble),
    whereas the phases drawn here are independent.

    Parameters
    ----------
    n : int
        Dimension of the SU(n) matrices.
    shape : tuple (optional)
        Specifing the shape of tensor of random matrices.
    """

    def __init__(self, n, shape=None, drop_constant_log_prob=False):
        if shape is None:
            shape = ()
        low = torch.zeros(*shape, n - 1)
        high = torch.ones(*shape, n - 1) * (2 * pi)
        self.n = n
        self.shape = shape
        self.uniform_dist = torch.distributions.uniform.Uniform(low, high)
        self.drop_constant_log_prob = drop_constant_log_prob
        self.log_group_vol = (n - 1) * log(2 * pi)
        self.log_tot_vol = self.log_group_vol + log(np.prod(shape))

    def sample(self, size=(1,)):  # this is the `sample` of dist (not prior)
        """Draw random samples."""
        free = self.uniform_dist.sample(size)
        last = -free.sum(dim=-1, keepdim=True)
        angles = torch.cat([free, last], dim=-1)
        return torch.diag_embed(torch.exp(1j * angles))

    def log_prob(self, x):
        """Return log_prob of each matrix."""
        lat_shape = x.shape[:-2]
        if self.drop_constant_log_prob:
            return torch.zeros(lat_shape, device=x.device)
        return torch.zeros(lat_shape, device=x.device) - self.log_group_vol


# =============================================================================
class U1Group:
    r"""Generate random unitary matrices, i.e. random U(1).

    This is an implementation of random U(1), which is faster than UnGroup(1).

    For a random unitary variable, the probability distribution function
    is math:`p(z) = 1/(2 \pi i) 1/z` such that math:`p(z) dz` is always real
    and its integral over any circle about the origin amounts to unity.

    Parameters
    ----------
    shape : tuple (optional)
        Specifing the shape of multivariate unitary distribution.
    """

    def __init__(self, shape=(1,)):
        low = torch.zeros(shape)
        high = torch.ones(shape) * (2 * pi)
        self.shape = shape
        self.uniform_dist = torch.distributions.uniform.Uniform(low, high)
        self.log_group_vol = np.log(2 * pi)
        self.log_tot_vol = np.log(2 * pi) + np.log(np.prod(shape))

    def sample(self, size=(1,)):  # this is the `sample` of dist (not prior)
        """Draw random samples."""
        return torch.exp(1j * self.uniform_dist.sample(size))

    def log_prob(self, x):
        """This is the real part of math:`log(p(z))`, ie math:`log(|p(z)|)`."""
        return torch.zeros(x.shape, device=x.device) - self.log_group_vol


# =============================================================================
def test_spectrum(prior, n_samples=100, display=True, bins=30):
    # pylint: disable=all
    r"""Test the distribution of eigenvalues.

    Since Haar measure is the analogue of a uniform distribution, each set of
    eigenvalues must have the same weight, therefore the normalized eigenvalue
    density is :math:`\rho(\theta) = 1 / (2 \pi)` `[Mezzadri]`_.
    The histogram generated here must agree with Figure 2.a of `[Mezzadri]`_.

    .. _[Mezzadri]:
        F. Mezzadri,
        "How to generate random matrices from the classical compact groups",
        :arXiv:`math-ph/0609050`.
    """
    x = prior.sample(n_samples)
    eig, _ = torch.linalg.eig(x)
    phase = torch.angle(eig)

    if display:
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        axs[0].hist(phase.ravel().numpy(), bins=bins, density=True)
        axs[0].plot([-np.pi, np.pi], [1/2/np.pi]*2, ':k')
        axs[0].set_xlabel(r"$\theta$")
        axs[0].set_ylabel(r"$\rho(\theta)$")
        axs[0].set_xlim([-np.pi, np.pi])

        if len(phase.shape) > 1 and phase.shape[-1] > 1:
            phase = torch.sort(phase, dim=-1)[0]
            diff_phase = phase[..., 1:] - phase[..., :-1]
            for ind in range(diff_phase.shape[-1]):
                diff = diff_phase[..., ind]
                axs[1].hist(diff.ravel().numpy(), bins=bins, density=True)
            axs[1].set_xlabel(r"$\theta$")
            axs[1].set_ylabel(r"$\rho(\Delta \theta)$")
            axs[1].set_xlim([0, 2 * np.pi])

    out = dict(eig=eig, prod=torch.prod(eig, -1), phase=phase)

    if display:
        return axs, out
    else:
        return out
