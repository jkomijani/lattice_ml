# Copyright (c) 2024-2025 Javad Komijani

"""
This module provides various functions, especially for mapping between unitary
matrices and their Lie algebra elements.
"""

# pylint: disable=invalid-name  # for matrices like U & A

import math
import torch

from lattice_ml.linalg import eigh, eigu, inverse_eign


__all__ = [
    "exp_unitary_algebra",
    "log_unitary_group",
    "log_special_unitary_group",
    "pow_special_unitary_group",
    "pow_special_unitary_group_",
    "matrix_exp1jh",
    "matrix_angleu",
    "enforce_zero_sum",
    "kronecker_product",
    "eyes_like"
]


# =============================================================================
def exp_unitary_algebra(A: torch.Tensor) -> torch.Tensor:
    r"""
    Compute the exponential map from the Lie algebra `u(n)` to its lie group.

    Given an anti-Hermitian matrix `A`, this function returns the unitary
    matrix `exp(A)`.  If `A` is also traceless, `U` will be in SU(n).

    Parameters
    ----------
    A : torch.Tensor
        Anti-Hermitian input matrix of shape `(..., n, n)`.

    Returns
    -------
    U : torch.Tensor
        The corresponding unitary matrix (same shape as `A`).

    Notes
    -----
    This function uses the spectral decomposition of `A`:

    .. math::

        A = \Omega \Lambda \Omega^\dagger, \quad
        U = \Omega \exp(\Lambda) \Omega^\dagger.
    """
    vals, vecs = eigh(-1j * A)
    f_vals = torch.exp(1j * vals)
    return inverse_eign(f_vals, vecs)


def log_unitary_group(U: torch.Tensor) -> torch.Tensor:
    r"""
    Compute the logarithm map from the lie group `U(n)` to its Lie algebra.

    Given a unitary matrix `U`, this function returns the anti-Hermitian
    matrix `log(U)`.

    Parameters
    ----------
    U : torch.Tensor
        Unitary input matrix of shape `(..., n, n)`.

    Returns
    -------
    A : torch.Tensor
        The corresponding anti-Hermitian matrix.

    Notes
    -----
    This function uses the spectral decomposition of `U`:

    .. math::

        U = \Omega \Lambda \Omega^\dagger, \quad
        A = \Omega \log(\Lambda) \Omega^\dagger.
    """
    vals, vecs = eigu(U)
    f_vals = 1j * torch.angle(vals)
    return inverse_eign(f_vals, vecs)


def log_special_unitary_group(U: torch.Tensor) -> torch.Tensor:
    r"""
    Compute the logarithm map from the lie group `SU(n)` to its Lie algebra.

    Given a special unitary matrix `U`, this function returns the traceless,
    anti-Hermitian matrix `log(U)`.

    Parameters
    ----------
    U : torch.Tensor
        Speical Unitary input matrix of shape `(..., n, n)`.

    Returns
    -------
    A : torch.Tensor
        The corresponding traceless anti-Hermitian matrix.

    Notes
    -----
    This function uses the spectral decomposition of `U`:

    .. math::

        U = \Omega \Lambda \Omega^\dagger, \quad
        A = \Omega \log(\Lambda) \Omega^\dagger,

    and enssures that the matrix `A` is traceless.
    """
    vals, vecs = eigu(U)
    f_vals = 1j * enforce_zero_sum(torch.angle(vals))
    return inverse_eign(f_vals, vecs)


def pow_special_unitary_group(
    U: torch.Tensor,
    t: float | torch.Tensor
) -> torch.Tensor:
    r"""
    Computes `U^t` for a special unitary matrix using its eigen-decomposition.

    Parameters
    ----------
    U : torch.Tensor
        Special unitary input matrix of shape (..., n, n).
    t : float or torch.Tensor
        Exponent. Can be a float or a tensor broadcastable to the shape
        of the eigenvalues.

    Returns
    -------
    torch.Tensor
        Matrix raised to the power t (same shape as U).

    Notes
    -----
    This function uses the spectral decomposition of `U`:

    .. math::

        U = \Omega \Lambda \Omega^\dagger, \quad
        U^t = \Omega \Lambda^t \Omega^\dagger,

    and enssures that `U^t` remains special unitary.
    """
    vals, vecs = eigu(U)
    log_vals = 1j * enforce_zero_sum(torch.angle(vals))
    f_vals = torch.exp(log_vals * t)
    return inverse_eign(f_vals, vecs)


def pow_special_unitary_group_(
    U: torch.Tensor,
    t: float | torch.Tensor
) -> torch.Tensor:
    r"""
    Computes `U^t` for a special unitary matrix using its eigen-decomposition.
    In addition to `U^t`, this function also returns `log(U)` that is needed
    for taking derivatices with respect to the exponent.

    Parameters
    ----------
    U : torch.Tensor
        Special unitary input matrix of shape (..., n, n).
    t : float or torch.Tensor
        Exponent. Can be a float or a tensor broadcastable to the shape
        of the eigenvalues.

    Returns
    -------
    torch.Tensor
        Matrix raised to the power t (same shape as U) and logarithm of U.

    Notes
    -----
    This function uses the spectral decomposition of `U`:

    .. math::

        U = \Omega \Lambda \Omega^\dagger, \quad
        U^t = \Omega \Lambda^t \Omega^\dagger,

    and enssures that `U^t` remains special unitary.
    """
    vals, vecs = eigu(U)
    log_vals = 1j * enforce_zero_sum(torch.angle(vals))
    f_vals = torch.exp(log_vals * t)
    return inverse_eign(f_vals, vecs), inverse_eign(log_vals, vecs)


def matrix_exp1jh(H: torch.Tensor) -> torch.Tensor:
    r"""
    Return :math:`U = \exp(i H)` with :math:`H` being a Hermitian matrix.

    This is a variant of `exp_unitary_algebra` with Hermitian input matrix.
    """
    vals, vecs = eigh(H)
    f_vals = torch.exp(1j * vals)
    return inverse_eign(f_vals, vecs)


def matrix_angleu(U: torch.Tensor) -> torch.Tensor:
    r"""
    Return :math:`H = -i \log(U)` with :math:`U` being a unitary matrix.

    This is a variant of `log_unitary_group` with Hermitian output matrix.
    """
    vals, vecs = eigu(U)
    f_vals = torch.angle(vals)
    return inverse_eign(f_vals, vecs)


# =============================================================================
def enforce_zero_sum(
    x: torch.Tensor,
    dim: int = -1,
    period: float = 2 * math.pi
):
    """
    Adjust a single entry along `dim` so the total sum becomes exactly zero.

    Assumes that the sum along `dim` is already an integer multiple of the
    period (e.g., 2π). The function removes the winding number by modifying
    one entry.

    SU(3) eigen-angle example:
    - If sum = -2π, the smallest angle is increased by 2π.
    - If sum = 2π, the largest angle is decreased by 2π.
    - If sum = 0, no change is needed.

    Algorithm
    ---------
    1. Compute the winding number as the rounded sum divided by period.
    2. Sort entries along `dim` and pick the entry at (winding + n//2) % n.
    3. Map that index back to the original tensor and subtract period*winding.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor (e.g., eigen-angles), can have batch dimensions.
    dim : int, optional
        Dimension along which to enforce zero-sum (default: -1).
    period : float, optional
        Period of the values (default: 2π).

    Returns
    -------
    torch.Tensor
        Copy of `x` with one entry modified so the sum along `dim` equals 0.
    """
    # Compute integer winding number along dim
    n_w = roundint(torch.sum(x, dim=dim) / period).unsqueeze(dim)

    n_d = x.size(dim)

    # Sort values and get original indices
    indices = torch.argsort(x, dim=dim)

    # Comment: For SU(3) eigen-angles, sorting is unnecessary since x is sorted
    # when n_w ≠ 0. We sort anyway for consistency and generality.

    # Determine which entry to adjust in sorted order
    designated_idx = (n_w + n_d // 2) % n_d

    # Map back to original tensor
    designated_orig_idx = torch.gather(indices, dim=dim, index=designated_idx)

    # Subtract winding number * period from the chosen entry
    correction = -n_w.to(x.dtype) * period
    y = x.clone()
    y.scatter_add_(dim, designated_orig_idx, correction)
    return y


def roundint(x, dtype=torch.int64):
    """Return the closest integer to `x`."""
    return torch.round(x).to(dtype)


def kronecker_product(mat1, mat2):
    """Return the Kronecker product of two input matrices."""
    shp1 = mat1.shape
    shp2 = mat2.shape
    assert shp1[:-2] == shp2[:-2], f"{shp1[:-2]} != {shp2[:-2]}"
    mat1 = mat1.repeat_interleave(shp1[-2], -2).repeat_interleave(shp1[-1], -1)
    mat2 = mat2.repeat(*[1]*(len(shp1) - 2) + list(shp1[-2:]))
    return mat1 * mat2


def eyes_like(matrix):
    """Return identity matrices of the same size of the input matrix."""
    eye = torch.zeros_like(matrix)
    for k in range(matrix.shape[-1]):
        eye[..., k, k] = 1
    return eye


# =============================================================================
def _test_enforce_zero_sum(n_samples):
    # pylint: disable=import-outside-toplevel
    from normflow.prior import SUnPrior
    samples = SUnPrior(n=3, shape=(1,)).sample(n_samples)
    vals, _ = eigu(samples)
    angs = torch.angle(vals)
    angs_p = enforce_zero_sum(angs)
    for x, y in zip(angs, angs_p):
        print(x[0], f"{x.sum():.4f}", y[0])
