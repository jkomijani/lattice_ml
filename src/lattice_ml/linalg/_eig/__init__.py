# *** Functions imported here do not have reliable autograd. ***

import torch

from . import generic

from .eig_2x2 import eigh2x2, eigu2x2
from .eigh_3x3 import eigh3x3, eigvalsh3x3
from .eig_3x3 import eign3x3, eigvals3x3
from .eigsu_3x3 import eigvalssu3x3
from .eigh_jacobi import jacobi_diagonalization


def eigh(matrix: torch.Tensor, backend: str = "custom", **kwargs):
    """
    Compute the eigendecomposition of the Hermitian input matrix.

    Args:
        matrix (torch.Tensor): Hermitian matrix of shape (..., n, n).
        backend (str, optional): Which implementation to use.

    Returns:
        torch.Tensor: tensor of eigenvalues.
        torch.Tensor: tensor of eigenvectors.
    """
    n = matrix.shape[-1]

    if backend == "custom" and n == 2:
        return eigh2x2(matrix, **kwargs)
    if backend == "custom" and n == 3:
        return eigh3x3(matrix, **kwargs)

    # else use eigh
    return torch.linalg.eigh(matrix, **kwargs)


def eigu(matrix: torch.Tensor, backend: str = "custom", **kwargs):
    """
    Compute the eigendecomposition of the Unitary input matrix.

    Args:
        matrix (torch.Tensor): Unitary matrix of shape (..., n, n).
        backend (str, optional): Which implementation to use.

    Returns:
        torch.Tensor: tensor of eigenvalues.
        torch.Tensor: tensor of eigenvectors.
    """
    n = matrix.shape[-1]

    if backend == "custom" and n == 2:
        return eigu2x2(matrix, **kwargs)
    if backend == "custom" and n == 3:
        return eign3x3(matrix, **kwargs)  # use eign3x3, more precise

    # else use eigh
    return torch.linalg.eig(matrix, **kwargs)
