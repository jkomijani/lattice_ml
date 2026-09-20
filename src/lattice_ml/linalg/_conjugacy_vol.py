# Copyright (c) 2026 Javad Komijani

r"""Conjugacy-class volumes of a spectrum.

Diagonalizing a matrix trades its entries for a spectrum plus an eigenframe.
The eigenframe ranges over a coset of the group that conjugates the matrix,
and the volume of that orbit -- the *conjugacy volume* -- is the Jacobian of
recomposing the matrix from the two pieces. It depends on the spectrum alone,
through the differences of the eigenvalues.

Which group conjugates sets the exponent (the Dyson index): `|Delta|^2` for
the unitary orbit of a complex normal matrix, `|Delta|` for the orthogonal
orbit of a real symmetric one. The functions here are therefore named after
that group, not after the entries of the matrix: the dtype of `eigvals`
cannot distinguish the cases, since a complex Hermitian matrix and a real
symmetric matrix both have real eigenvalues.

Only the unitary case is implemented so far.
"""

# pylint: disable=invalid-name

import torch


__all__ = ['log_unitary_conjugacy_vol']


# =============================================================================
def log_unitary_conjugacy_vol(eigvals: torch.Tensor) -> torch.Tensor:
    r"""
    `log prod_{k<l} |lambda_k - lambda_l|^2`, up to an additive constant.

    This is the volume of the conjugacy class, i.e. the Jacobian of
    recomposing a normal matrix from its spectrum and eigenframe.

    Args:
        eigvals: Eigenvalues of shape `(..., N)`, real or complex.

    Returns:
        The log volume, shape `eigvals.shape[:-1]`.
    """
    log_vol = torch.zeros_like(eigvals[..., 0].real)
    for k in range(eigvals.shape[-1] - 1):
        diff = (eigvals[..., k:k+1] - eigvals[..., k+1:]).abs()
        log_vol = log_vol + 2 * torch.sum(torch.log(diff), dim=-1)
    return log_vol
