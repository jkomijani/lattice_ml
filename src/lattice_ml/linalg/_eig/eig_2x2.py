# Copyright (c) 2023 Javad Komijani

import torch
import numpy as np


# =============================================================================
def eigh2x2(matrix):
    """
    Return eigenvalues and eigenvectors of 2x2 hermitian matrices using closed
    form expressions.

    We use following parametrization of hermitian matrices

    .. math::

        H = [[t + z,    x - i y]
             [x + i y,   t - z ]]

    Then the eigenvalues are :math:`(t - r, t + r)` where
    :math:`r = \sqrt{x^2 + y^2 + z^2}` and the matrix of eigenvectors is

    .. math::

        \Omega = [[z - r,    x - i y]
                  [x + i y,   r - z ]] / c

    where :math:`c = \sqrt{2 r (r - z)}`.
    """
    
    a = torch.real(matrix[..., 0, 0])
    b = torch.real(matrix[..., 1, 1])
    x = torch.real(matrix[..., 1, 0])
    y = torch.imag(matrix[..., 1, 0])

    t = (a + b) * (1/2.)
    z = a - t
    r = (z**2 + x**2 + y**2)**0.5
    # Note that r >= |z|, thus r - z >= 0

    eigvals = torch.stack([t - r, t + r], dim=-1)

    vec_norm = (2 * r * (r - z)).reshape(*r.shape, 1, 1)
    eigvecs = torch.stack(
            [z - r, x - y*1j, x + y*1j, r - z], dim=-1
            ).reshape(*matrix.shape) / vec_norm**0.5

    # if vec_norm == 0, then eigvecs = eye(2)
    cond = vec_norm.ravel() == 0
    if torch.sum(cond) > 0:
        cond0 = cond & (z <= 0).ravel()
        cond1 = cond & (z > 0).ravel()
        mat0 = torch.tensor([[[1, 0], [0, 1]]]) + 0j
        mat1 = torch.tensor([[[0, 1], [1, 0]]]) + 0j
        eigvecs_ = eigvecs.reshape(-1, 2, 2)
        eigvecs_[cond0] = mat0.repeat(sum(cond0), 1, 1)
        eigvecs_[cond1] = mat1.repeat(sum(cond1), 1, 1)

    return eigvals, eigvecs


# =============================================================================
def eigsu2x2(matrix):
    """
    Return eigenvalues and eigenvectors of 2x2 special unitary matrices using
    closed form expressions.

    We use following parametrization of 2x2 special unitray matrices

    .. math::

        U = [[t + i z,   i x + y]
             [i x - y,   t - i z]]

    where the deteriminant, i.e., :math:`t^2 + x^2 + y^2 + z^2` is unity.
    (Here we do not check whether the determinat is in fact unity.)
    Then the eigenvalues are :math:`(t - i r, t + i r)` where
    :math:`r = \sqrt{x^2 + y^2 + z^2}` and the matrix of eigenvectors is

    .. math::

        \Omega = [[z - r,    x - i y]
                  [x + i y,   r - z ]] / c

    where :math:`c = \sqrt{2 r (r - z)}`.
    """
    
    t = torch.real(matrix[..., 0, 0])
    z = torch.imag(matrix[..., 0, 0])
    x = torch.imag(matrix[..., 0, 1])
    y = torch.real(matrix[..., 0, 1])

    r = (z**2 + x**2 + y**2)**0.5
    # Note that r >= |z|, thus r - z >= 0

    eigvals = torch.stack([t - r * 1j, t + r * 1j], dim=-1)

    vec_norm = (2 * r * (r - z)).reshape(*r.shape, 1, 1)
    eigvecs = torch.stack(
            [z - r, x - y*1j, x + y*1j, r - z], dim=-1
            ).reshape(*matrix.shape) / vec_norm**0.5

    # if vec_norm == 0, then eigvecs = eye(2)
    cond = vec_norm.ravel() == 0
    if torch.sum(cond) > 0:
        cond0 = cond & (z <= 0).ravel()
        cond1 = cond & (z > 0).ravel()
        mat0 = torch.tensor([[[1, 0], [0, 1]]]) + 0j
        mat1 = torch.tensor([[[0, 1], [1, 0]]]) + 0j
        eigvecs_ = eigvecs.reshape(-1, 2, 2)
        eigvecs_[cond0] = mat0.repeat(sum(cond0), 1, 1)
        eigvecs_[cond1] = mat1.repeat(sum(cond1), 1, 1)

    return eigvals, eigvecs


# =============================================================================
def eigu2x2(matrix):
    root_det = torch.det(matrix).unsqueeze(-1)**0.5
    u, v = eigsu2x2(matrix / root_det.unsqueeze(-1))
    return root_det * u, v
