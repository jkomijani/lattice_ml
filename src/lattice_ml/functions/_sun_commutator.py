# Copyright (c) 2026 Javad Komijani

r"""
Exact, closed-form solution of the SU(N) commutator equation.

Given any Z in SU(N), this module returns X, Y in SU(N) satisfying

    Z = [X, Y]

where `[X, Y] = X Y X† Y†`, in closed form.
"""

# pylint: disable=invalid-name

import torch

from lattice_ml.linalg import eigu, inverse_eign

from ._matrix_func import enforce_zero_sum


__all__ = ["solve_sun_commutator"]


# =============================================================================
def solve_sun_commutator(Z: torch.Tensor, random_twist: bool = False):
    r"""
    Solve `Z = X Y X† Y†` for X, Y in SU(N), given Z in SU(N), in closed form.

    Parameters
    ----------
    Z : torch.Tensor
        Special unitary input matrix of shape `(..., N, N)`.

    random_twist : bool, default=False
        If True, draw X and Y independently and uniformly at random from
        their `(N-1)`-parameter families of equally-valid solutions (see
        "Non-uniqueness exploited by `random_twist`" below).
        If False, return the single canonical solution the construction below
        builds directly.

    Returns
    -------
    X, Y : torch.Tensor
        One pair of special unitary matrices satisfying `Z = X Y X† Y†`.
        This is *a* solution; the solution is **not unique** (see below).


    Construction:
    -------------
    1. Diagonalize `Z = Ω Λ Ω†`.

    2. Solve `Λ = X0 Y0 X0† Y0†` via `X0 = diag(e^{iφ_k})` and `Y0 = S`
       (cyclic-shift, rescaled by `e^{iπ/N}` for even N so`det Y0 = 1`).

    3. Conjugating by `S` permutes `X0†`'s diagonal, reducing this to
       `θ = P φ`, `P = I - S`. Since `sum_k θ_k = 0` (because `det Z = 1`),
       `θ` lies in `P`'s range, so `φ = P⁺ θ` solves it (also `sum_k φ_k = 0`).

    4. Set `X = Ω X0 Ω†`, `Y = Ω Y0 Ω†`. This `(X, Y)` is already a valid
       solution, but `Y`'s eigenvalues are always bound to the permutation
       matrix's spectrum. Since `(X, Y X^α)` is also valid for any real `α`,
       return `(X, Y X)` (`α = 1`) instead, so `Y`'s eigenvalues vary too
       when `random_twist=True`.

    Non-uniqueness exploited by `random_twist`:
    -------------------------------------------
    Right-multiplying by any `D` that *commutes* with the other matrix also
    solves it (`[X D, Y] = X D Y D†X†Y† = [X,Y]` when `DY=YD`). `y0 -> y0 @ D`
    for any diagonal `D` in SU(N) commutes with `X0` (both diagonal);
    and `X -> X @ D'` for any `D'` diagonal in `Y`'s own eigenbasis commutes
    with `Y`. Each is an `(N-1)`-real-parameter family, and together they
    already cover the *discrete* scalar-twist freedom too (`X -> c X` for `c`
    an N-th root of unity is the special case `D' = c I`, trivially diagonal
    in any basis) -- no separate mechanism for that discrete case is needed.

    Other known solutions:
    ----------------------
    The provided solutions are only one slice of the full solution set for
    fixed `Z`. (For instance we do not explore `(X, Y) -> (Y, X†)`.)
    """
    N = Z.shape[-1]
    dtype, device = Z.dtype, Z.device
    batch_shape = Z.shape[:-2]

    # Diagonalize Z = Ω Λ Ω†, Λ = diag(e^{iθ_k}).
    vals, omega = eigu(Z)
    theta = torch.angle(vals)
    theta = enforce_zero_sum(theta, dim=-1)

    # Solve the reduced linear system θ = P φ, P = I - S, via the
    # Moore-Penrose pseudo-inverse (P has a 1D kernel: the all-ones vector).
    p_pinv = _circulant_difference_pinv(N, theta.dtype, device)
    phi = torch.einsum('jk,...k->...j', p_pinv, theta)

    # Y0 = S, the cyclic-shift matrix; rescale for even N to keep det = 1.
    y0 = _cyclic_shift_matrix(N, dtype, device)
    if N % 2 == 0:
        angle = torch.tensor(torch.pi / N, dtype=theta.dtype)
        y0 = y0 * torch.exp(1j * angle)

    # For random_twist, right-multiply y0 by a random diagonal SU(N)
    # matrix -- it commutes with X0 (both diagonal), giving Y its full
    # (N-1)-parameter continuous freedom.
    if random_twist:
        y0 = y0 @ _random_diagonal_sun(batch_shape, N, theta.dtype, device)

    # X = Ω X0 Ω†, Y = Ω Y0 Ω†.
    X = inverse_eign(torch.exp(1j * phi), omega)
    Y = omega @ y0 @ omega.adjoint()

    # For random_twist, also right-multiply X by a random matrix diagonal
    # in Y's own eigenbasis -- it commutes with Y, so [X @ D', Y] = [X, Y]
    # still holds, giving X the same kind of continuous freedom as Y.
    if random_twist:
        _, v = eigu(Y)
        d_prime = _random_diagonal_sun(batch_shape, N, theta.dtype, device)
        X = X @ (v @ d_prime @ v.adjoint())

    # (X, Y) is already a valid solution, but Y's eigenvalues are always bound
    # to the permutation matrix's own spectrum. Since (X, Y X^alpha) is also
    # valid for any real alpha, return (X, Y X) (alpha=1) so Y's eigenvalues
    # vary too when random_twist=True.
    #
    # This mirror-twist-and-shift step (the two blocks above, from
    # `if random_twist:` through here) could be repeated to further
    # reshape the eigenangle distribution of X, Y; empirically it never
    # makes that distribution match Z's exactly, so it isn't done here.
    return X, Y @ X


# =============================================================================
def _random_diagonal_sun(batch_shape, N, real_dtype, device):
    """
    A uniformly random diagonal SU(N) matrix per batch element: the
    first `N-1` phases are drawn i.i.d. uniform on the circle; the
    last is fixed as minus their sum, so `det = 1`.
    """
    # Unlike subtracting the mean of all N phases, this keeps each phase
    # uniform on the full circle instead of concentrating it near zero.
    free = torch.rand(
        *batch_shape, N - 1, dtype=real_dtype, device=device
    ) * (2 * torch.pi)
    last = -free.sum(dim=-1, keepdim=True)
    angles = torch.cat([free, last], dim=-1)
    return torch.diag_embed(torch.exp(1j * angles))


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
