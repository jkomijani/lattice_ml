# Copyright (c) 2026 Javad Komijani

r"""
Exact, closed-form solution of the SU(N) commutator equation.

Given any Z in SU(N), this module returns X, Y in SU(N) satisfying

    Z = [X, Y]

where `[X, Y] = X Y X^\dagger Y^\dagger`, in closed form.
"""

# pylint: disable=invalid-name

import math
import torch

from lattice_ml.linalg import eigu, inverse_eign

from ._matrix_func import enforce_zero_sum


__all__ = ["solve_sun_commutator"]


# =============================================================================
def solve_sun_commutator(Z: torch.Tensor):
    r"""
    Solve `Z = X Y X^\dagger Y^\dagger` for X, Y in SU(N), given Z in SU(N),
    in closed form.

    Parameters
    ----------
    Z : torch.Tensor
        Special unitary input matrix of shape `(..., N, N)`.

    Returns
    -------
    X, Y : torch.Tensor
        One particular pair of special unitary matrices (same shape as Z)
        satisfying `Z = X Y X^\dagger Y^\dagger`. This is *a* solution, not
        *the* solution -- the solution is **not unique**.

    Notes
    -----
    The construction proceeds in four steps:

    1. Diagonalize `Z = Omega D Omega^\dagger`, `D = diag(e^{i theta_k})`.
       Solve the diagonal-target problem `D = X0 Y0 X0^\dagger Y0^\dagger`
       and set `X = Omega X0 Omega^\dagger`, `Y = Omega Y0 Omega^\dagger`.

    2. Take `X0 = diag(e^{i phi_k})` and `Y0 = S`, the cyclic-shift
       permutation matrix. Conjugation by S just cyclically permutes
       `X0^\dagger`'s diagonal, reducing the equation to the linear system

           theta_k = phi_k - phi_{k+1 mod N}   <=>   theta = P phi,  P = I - S.

    3. P is circulant with a one-dimensional kernel (the all-ones vector);
       `sum_k theta_k = 0` (enforced via `enforce_zero_sum`, det `Z = 1`
       guarantees it's a multiple of `2 pi`) puts `theta` in P's range, so
       `phi = P^+ theta` solves it (`P^+` = Moore-Penrose pseudo-inverse).

    4. `det S = (-1)^(N-1)`, so S is only in SU(N) for odd N; for even N,
       `Y0` is rescaled by `e^{i pi/N}` to fix `det Y0 = 1`.

    Non-uniqueness: for any independently chosen N-th roots of unity a, b,
    `(a X, b Y)` is also a valid solution; so is `(Y, X^\dagger)`. More
    fundamentally, "Y built from X's own eigenbasis" isn't a well-defined
    function of X: relabeling which eigenvalue pairs with which eigenvector
    column before building `X0, Y0` reconstructs the identical X but a
    *different* Y (verified numerically for SU(3) -- a fresh `eigu(X)`
    call returns X's eigenvalues in a different column order than the one
    that built X). The full solution set for fixed Z is in fact a
    continuous `(N^2-1)`-dimensional family, of which both the center
    orbit and the `(X0, S)` ansatz used here are only small slices. See
    the paper (docs/exact_2d_holonomy_gauge_theory, App.~A) for the full
    derivation and why no branch-count correction is applied anywhere in
    this codebase.
    """
    N = Z.shape[-1]
    dtype, device = Z.dtype, Z.device

    vals, omega = eigu(Z)
    theta = torch.angle(vals)
    theta = enforce_zero_sum(theta, dim=-1)

    p_pinv = _circulant_difference_pinv(N, theta.dtype, device)
    phi = torch.einsum('jk,...k->...j', p_pinv, theta)

    y0 = _cyclic_shift_matrix(N, dtype, device)
    if N % 2 == 0:
        angle = torch.tensor(math.pi / N, dtype=theta.dtype)
        phase = torch.exp(1j * angle)
        y0 = y0 * phase.to(dtype)

    X = inverse_eign(torch.exp(1j * phi).to(dtype), omega)
    Y = omega @ y0 @ omega.adjoint()
    return X, Y


# =============================================================================
def _circulant_difference_pinv(N, dtype, device):
    """
    Moore-Penrose pseudo-inverse of `P = I - S`, the circulant matrix that
    the commutator equation reduces to (see `solve_sun_commutator`). P is
    normal with a one-dimensional kernel spanned by the all-ones vector.
    """
    shift = _cyclic_shift_matrix(N, dtype, device)
    p = torch.eye(N, dtype=dtype, device=device) - shift
    return torch.linalg.pinv(p)


def _cyclic_shift_matrix(N, dtype, device):
    """
    Return the N x N cyclic-shift permutation matrix S with
    `S_{k, k+1 mod N} = 1` (all other entries zero).
    """
    shift = torch.zeros(N, N, dtype=dtype, device=device)
    idx = torch.arange(N, device=device)
    shift[idx, (idx + 1) % N] = 1
    return shift
