# Copyright (c) 2026 Javad Komijani

r"""
Exact, closed-form solution of the SU(N) group commutator equation.

Given any Z in SU(N), this module returns X, Y in SU(N) satisfying

    Z = [X, Y]

where `[X, Y] = X Y X† Y†`, in closed form.
"""

# pylint: disable=invalid-name

import torch

from ._unitary_group import rand_sun_group_like, rand_diagonal_sun_group_like


__all__ = ["sample_sun_group_commutator"]


# =============================================================================
def sample_sun_group_commutator(Z: torch.Tensor, n_mix: int = 10):
    """
    Sample `Z = X Y X† Y†` for X, Y in SU(N), given Z in SU(N).

    The weight of picking up a fully random Q is not taken into account and
    is not returened. Setting `n_max` to a big number, such as 10, reduces
    the effect of not considering the weight.

    Parameters
    ----------
    Z : torch.Tensor
        Special unitary input matrix of shape `(..., N, N)`.

    Returns
    -------
    X, Y : torch.Tensor
        One random pair of special unitary matrices satisfying `Z = X Y X† Y†`.
    """
    # Part 1: Draw sample for Q and D
    Q = rand_sun_group_like(Z)
    D = rand_diagonal_sun_group_like(Z)

    # Part 2: Construct Y
    N = Z.shape[-1]
    assert N in (2, 3), "only 2 and 3 are supported"

    if N == 2:
        Lambda = solve_lambda_su2(Z, Q)
    else:
        Lambda = solve_lambda_su3(Z, Q)

    Lambda, Q = sort_eig(Lambda, Q, sort_by='angle')
    Y = Q @ torch.diag_embed(Lambda) @ Q.adjoint()

    # Part 3: Construct X
    _, B = sort_eig(*torch.linalg.eig(Z @ Y), sort_by='angle')
    X = B @ D @ Q.adjoint()
    X = X / torch.linalg.det(X)[..., None, None] ** (1/N)

    for _ in range(n_mix):
        X = X @ Y
        Y = Y @ X

    # This is the weight if n_mix is 0
    # weight for Y: |1 - Tr(Q.adjoint() @ Z @ Q)|

    return X, Y


# =============================================================================
def solve_lambda_su2(Z, Q, descending_angle: bool = False):
    """
    Solve `Tr (I - Q† Z Q) Λ = 0`, where `Λ` is diagonal and all matrices are
    SU(2).

    Because `Q† Z Q` is in SU(2), m_2 = conj(m_1) and the constraint
    `sum_k m_k u_k = 0` is the equation `Re(m_1 u_1) = 0`.

    The angle of the first item if between 0 to pi if `descending_angle=True`.
    """
    m_1 = 1 - (Q.adjoint() @ Z @ Q)[..., 0, 0]
    fac = 1j if descending_angle else -1j  # rotation factor
    u_1 = fac * m_1.conj() / m_1.abs()
    return torch.stack([u_1, u_1.conj()], dim=-1)


# =============================================================================
def solve_lambda_su3(Z, Q, descending_angle: bool = False, root: int = 0):
    """
    Solve `Tr (I - Q† Z Q) Λ = 0`, where `Λ` is diagonal and all matrices are
    SU(3).

    `descending_angle` is meaningful only for `root = 0`.

    Because `Q† Z Q` is in SU(3), the problem reduces to triangle closure.
    """
    diag = 1 - torch.diagonal(Q.adjoint() @ Z @ Q, dim1=-2, dim2=-1)

    a, b, c = diag.abs().unbind(-1)

    # Let us assume three vectors v1, v2, v3 with absolute value a, b, c
    # we define delta as the angle between v1 and v2
    delta = torch.pi - torch.arccos((a**2 + b**2 - c**2) / (2 * a * b))

    # 0 =< delta =< pi
    if descending_angle:
        v1 = a.to(diag.dtype) * torch.exp(1j * delta)
        v2 = b.to(diag.dtype)
    else:
        v1 = a.to(diag.dtype) * torch.exp(-1j * delta)
        v2 = b.to(diag.dtype)

    u = torch.stack([v1, v2, -v1 - v2], dim=-1) / diag

    # We should impose product(u) = 1 for SU(N)
    alpha_0 = (-torch.angle(u).sum(dim=-1) + 2 * torch.pi * root) / 3

    return u * torch.exp(1j * alpha_0)[..., None]


# =============================================================================
def sort(x: torch.Tensor, dim=-1, descending=False, sort_by='real'):
    """Sort a tensor by real, imaginary, magnitude, or phase."""

    if sort_by in ["real", "imag", "abs", "angle"]:
        key = getattr(torch, sort_by)(x)
    else:
        raise ValueError(f"Invalid sort_by: {sort_by}")

    key = torch.angle(x)
    _, sorted_ind = torch.sort(key, dim=dim, descending=descending)

    return x.gather(-1, sorted_ind), sorted_ind


def sort_eig(eigvals, eigvecs, descending=False, sort_by="real"):
    """Sort a batch of eigenvalues and permute the corresponding eigenvectors.

    Args:
        eigvals (Tensor): (..., n) tensor of eigenvalues.
        eigvecs (Tensor): (..., n, n) tensor of eigenvectors (as columns).
        descending (bool): sorting order. (Default is False, i.e. ascending.)
        sort_by (str): key used for sorting: "real", "imag", "abs", "angle".

    Returns:
        Tensor, Tensor: the sorted eigenvalues and permuted eigenvectors.
    """
    eigvals, sorted_ind = sort(eigvals, descending=descending, sort_by=sort_by)

    n = eigvecs.shape[-1]
    sorted_ind = sorted_ind.unsqueeze(-2).repeat(*[1]*(eigvecs.ndim - 2), n, 1)
    eigvecs = eigvecs.gather(-1, sorted_ind)
    return eigvals, eigvecs


# =============================================================================
def verify_sampler(n=3, n_samples=1024):
    """
    Draw `n_samples` targets in SU(n) and report how well the sampler's (X, Y)
    satisfy `X Y X_dagger Y_dagger = Z` for each one.
    """
    # pylint: disable=import-outside-toplevel

    torch.set_default_dtype(torch.float64)
    from normflow.prior import UniformSUnPrior

    Z = UniformSUnPrior(n).sample(n_samples)
    X, Y = sample_sun_group_commutator(Z)
    Z_hat = X @ Y @ X.adjoint() @ Y.adjoint()

    err = torch.linalg.matrix_norm(Z_hat - Z)

    print(f"worst error over {len(err)} samples: {err.max():.4g}")
    print(f"fraction with error > 1e-10: {(err > 1e-10).float().mean():.4g}")
