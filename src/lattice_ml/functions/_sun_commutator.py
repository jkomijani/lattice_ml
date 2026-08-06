# Copyright (c) 2026 Javad Komijani

r"""
Exact, closed-form solution of the SU(N) commutator equation.

Given any ``Z`` in SU(N), this module returns ``X, Y`` in SU(N) satisfying

    Z = [X, Y]

where ``[X, Y] = X Y X^\dagger Y^\dagger``, in closed form.
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
    Solve ``Z = X Y X^\dagger Y^\dagger`` for ``X, Y`` in SU(N), given ``Z`` in
    SU(N), in closed form.

    Parameters
    ----------
    Z : torch.Tensor
        Special unitary input matrix of shape ``(..., N, N)``.

    Returns
    -------
    X, Y : torch.Tensor
        One particular pair of special unitary matrices (same shape as ``Z``)
        satisfying ``Z = X Y X^\dagger Y^\dagger``. Note that the output is
        *a* solution, not *the* solution as the solution is **not unique**.

    Notes
    -----
    The construction proceeds in four steps:

    1. Diagonalize ``Z = Omega D Omega^\dagger``, ``D = diag(e^{i theta_k})``.
       Under simultaneous conjugation, it suffices to solve the diagonal-target
       problem ``D = X0 Y0 X0^\dagger Y0^\dagger`` and set
       ``X = Omega X0 Omega^\dagger``, ``Y = Omega Y0 Omega^\dagger``.

    2. Take ``X0 = diag(e^{i phi_k})`` and ``Y0 = S``, the cyclic-shift
       permutation matrix. Since ``S`` only reorders the diagonal entries of
       ``X0^\dagger`` under conjugation, the equation reduces entry-by-entry
       to the linear system

           theta_k = phi_k - phi_{k+1 mod N}   <=>   theta = P phi,  P = I - S.

    3. ``P`` is circulant, with eigenvalues ``1 - e^{2 pi i m/N}``; the
       ``m = 0`` mode gives a one-dimensional kernel spanned by the all-ones
       vector. Because ``det Z = 1``, ``sum_k theta_k`` is guaranteed to be
       an integer multiple of ``2 pi``, but it is fixed here to zero via
       ``enforce_zero_sum``. This satisfies the Fredholm solvability condition,
       so ``P`` is genuinely invertible on the subspace that matters. We use
       ``torch.linalg.pinv`` to calculate the Moore-Penrose pseudo-inverse --
       though this could equally be solved via FFT since ``P`` is circulant.

    4. The permutation matrix has ``det S = (-1)^(N-1)``, so ``S`` is in SU(N)
       only for odd N; for even N, ``Y0`` is rescaled by the global phase
       ``e^{i pi/N}``, which leaves the solution unaffected while fixing
       ``det Y0 = 1``.

    This is *a* solution, not *the* solution as the solution is **not unique**.
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
    Moore-Penrose pseudo-inverse of ``P = I - S``, the circulant matrix that
    the commutator equation reduces to (see ``solve_sun_commutator``). ``P``
    is normal with a one-dimensional kernel spanned by the all-ones vector.
    """
    shift = _cyclic_shift_matrix(N, dtype, device)
    p = torch.eye(N, dtype=dtype, device=device) - shift
    return torch.linalg.pinv(p)


def _cyclic_shift_matrix(N, dtype, device):
    """
    Return the N x N cyclic-shift permutation matrix ``S`` with
    ``S_{k, k+1 mod N} = 1`` (all other entries zero).
    """
    shift = torch.zeros(N, N, dtype=dtype, device=device)
    idx = torch.arange(N, device=device)
    shift[idx, (idx + 1) % N] = 1
    return shift
