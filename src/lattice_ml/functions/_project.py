# Copyright (c) 2025 Javad Komijani

"""
This module has functions for projection of matrices onto U(N) and SU(N).
"""

# pylint: disable=no-member

import torch
from lattice_ml.linalg import svd


__all__ = [
    "project_onto_unitary",
    "project_onto_special_unitary",
    "project_onto_su2",
    "naive_project_onto_su3"
]


def project_onto_unitary(matrix):
    """
    Project a square matrix onto the closest unitary matrix using SVD.

    This function computes the singular value decomposition (SVD) of the input
    matrix M and reconstructs the unitary matrix Q that maximizes Re Tr(Q† M).

    Parameters
    ----------
    matrix : torch.tensor
        Input complex or real matrix of shape (..., n, n).

    Returns
    -------
    torch.tensor
        The closest unitary matrix.
    """
    svd_ = svd(matrix)
    return svd_.U @ svd_.Vh


def project_onto_special_unitary(matrix):
    """
    Project a square matrix onto the closest special unitary matrix using SVD.

    This function first projects the input matrix onto the unitary group
    (using SVD), then rescales it by a global phase factor so that its
    determinant is 1. Equivalently, it finds Q ∈ SU(n) that maximizes
    Re Tr(Q† M).

    Parameters
    ----------
    matrix : torch.Tensor
        Input complex or real matrix of shape (..., n, n).

    Returns
    -------
    torch.Tensor
        The closest special unitary matrix.
    """
    q = project_onto_unitary(matrix)
    rdet_angle = torch.angle(torch.linalg.det(q)) / q.shape[-1]
    phase_factor = torch.exp(-1j * rdet_angle).unsqueeze(-1).unsqueeze(-1)
    return q * phase_factor


def project_onto_su2(matrix):
    """
    Project a 2x2 matrix onto SU(2) using a closed form relation.

    Parameters
    ----------
    matrix : torch.Tensor
        Input complex or real matrix of shape (..., n, n).

    Returns
    -------
    torch.Tensor
        The closest SU(2) matrix.
    """

    # The SU(2) matrix is represented as v0 + i * Sum_j (sigma_j * vj)
    #
    #     | A+iB     C+iD |
    # M = |               |
    #     | E+iF     G+iH |
    #
    #   = 1/2*[ (A+G)*I + i*(B-H)*sigma_z + i*(F+D)*sigma_x + i*(C-E)*sigma_y ]
    #   + i/2*[ (B+H)*I - i*(A-G)*sigma_z - i*(C+E)*sigma_x - i*(F-D)*sigma_y ]
    #
    # The second line does not contribute to `Re Tr (Q^\dagger W)`, so we drop
    # it. When the first line is identical to zero, we simply assign
    # the identity matrix as the projection of M onto SU(2).

    v0 = matrix[..., 0, 0].real + matrix[..., 1, 1].real
    v3 = matrix[..., 0, 0].imag - matrix[..., 1, 1].imag
    v1 = matrix[..., 0, 1].imag + matrix[..., 1, 0].imag
    v2 = matrix[..., 0, 1].real - matrix[..., 1, 0].real

    v_sq = v0**2 + v1**2 + v2**2 + v3**2

    r = 1 / v_sq**0.5  # for normalization

    # TODO: zero v_sq is not yet taken into account

    out_mat = torch.zeros_like(matrix)

    out_mat[..., 0, 0] = (v0 - v3*1J) * r
    out_mat[..., 0, 1] = (-v2 - v1*1J) * r
    out_mat[..., 1, 0] = (v2 - v1*1J) * r
    out_mat[..., 1, 1] = (v0 + v3*1J) * r

    return out_mat


def naive_project_onto_su3(y):
    """
    Naively projects a 3x3 complex matrix to SU(3) by orthonormalizing rows.

    This method assumes the input matrix is close to the identity. It first
    orthonormalizes the first two rows, then reconstructs the third row to
    enforce unitarity and determinant = 1.

    Notes:
    1. Although not necessary, the matrix is initially normalized to ensure
       determinnat 1.
    2. The changes are not in-place because PyTorch cannot handle
       backpropagation of derivatives (if the adjointstate method is not used).
    """
    # Normalize matrix to ensure determinant is 1 (special unitary)
    # Explicit calculation of determinant is faster than torch.linalg.det!
    y_00, y_01, y_02 = torch.unbind(y[..., 0, :], dim=-1)
    y_10, y_11, y_12 = torch.unbind(y[..., 1, :], dim=-1)
    y_20, y_21, y_22 = torch.unbind(y[..., 2, :], dim=-1)
    det = (
        y_20 * (y_01 * y_12 - y_02 * y_11)
        + y_21 * (y_02 * y_10 - y_00 * y_12)
        + y_22 * (y_00 * y_11 - y_01 * y_10)
    )

    y = y / det[..., None, None]**(1/3.)

    # Unbind rows for further calculations
    y_0, y_1, _ = torch.unbind(y, dim=-2)

    # Normalize the first row
    norm_sq = torch.sum(y_0.conj() * y_0, dim=-1, keepdim=True)
    y_0 = y_0 / torch.sqrt(norm_sq)

    # Compute inner product of first two rows
    vdot = torch.sum(y_0.conj() * y_1, dim=-1, keepdim=True)
    # Orthogonalize second row against the first
    y_1 = y_1 - y_0 * vdot

    # Normalize the second row
    norm_sq = torch.sum(y_1 * y_1.conj(), dim=-1, keepdim=True)
    y_1 = y_1 / torch.sqrt(norm_sq)

    # Reconstruct third row as complex conjugate of cross product of first two
    y_2 = torch.stack(
        ((y_0[..., 1] * y_1[..., 2] - y_0[..., 2] * y_1[..., 1]).conj(),
         (y_0[..., 2] * y_1[..., 0] - y_0[..., 0] * y_1[..., 2]).conj(),
         (y_0[..., 0] * y_1[..., 1] - y_0[..., 1] * y_1[..., 0]).conj()
         ),
        dim=-1
    )

    y = torch.stack((y_0, y_1, y_2), dim=-2)

    return y
