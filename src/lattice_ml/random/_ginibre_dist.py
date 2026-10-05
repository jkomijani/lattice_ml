# Copyright (c) 2021-2026 Javad Komijani

"""
This module has a class to generate complex matrices from the Ginibre ensemble.
"""
# Moved here from `normflow.lib.stats.ginibre_dist.py`, which now re-exports
# these for backward compatibility.

import torch


__all__ = ["GinibreCMatrixDist"]


# =============================================================================
class GinibreCMatrixDist:
    """
    For generating matrices from the Ginibre ensemble, which is the ensemble of
    square, complex-valued matrices with i.i.d. normal entries.

    For the definition see `[Mezzadri]`_, which takes advantage of the Ginibre
    random matrices to generate unitary matrices.

    Parameters
    ----------
    n : int
        Specifies the dimension n of the Ginibre matrices.

    shape : Tuple[int] | None (optional, default=None)
        Specifing the shape of tensor of Ginibre matrices.
        Each sample would be of a tensor of size (*shape, n, n),
        where the last two dimensions construct GL(n, C) matrices.
        If None, shape will be set to (); shape can accept () too.

    .. _[Mezzadri]:
        F. Mezzadri,
        "How to generate random matrices from the classical compact groups",
        :arXiv:`math-ph/0609050`.
    """

    def __init__(self, *, n, shape=None, sigma=1):
        if shape is None:
            shape = ()
        self.n = n
        self.shape = shape
        shape_ = (*shape, n, n)  # the shape of underlying torch tensor
        loc = torch.zeros(shape_)  # i.e. mean
        scale = sigma * torch.ones(shape_)  # i.e. sigma
        self.normal_dist = torch.distributions.normal.Normal(loc, scale)

    def sample(self, size=(1,)):  # this is the `sample` of dist (not prior)
        """Draw random samples."""
        fnc = self.normal_dist.sample
        return fnc(size) + 1j * fnc(size)

    def log_prob(self, x):
        """Return log_prob for each matrix."""  # one value for each matrix
        fnc = self.normal_dist.log_prob
        return torch.sum(fnc(x.real) + fnc(x.imag), dim=(-2, -1))
