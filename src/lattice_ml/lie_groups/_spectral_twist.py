r"""
Jacobian of the spectral-twist map on SU(N).

    U = W Lambda W^dagger  (eigendecomposition, eigenvalues sorted by argument)
    Z_tilde = W Lambda W^dagger Lambda^dagger = U Lambda^dagger

In matched left-invariant frames

    d eta = W^dagger U^dagger dU W ,
    d xi  = W^dagger Z_tilde^dagger dZ_tilde W

the Jacobian is independent of Lambda and equals

    |det(d xi / d eta)| = det(I - B) restricted to  { h : sum_m h_m = 0 },
    B[k, j] = |W[j, k]|^2      (doubly stochastic)

            = prod_{k=2}^{N} (1 - mu_k); mu_1 = 1, mu_2..mu_N = rest of spec(B)
            = sum_{k=1}^{N} det[(I - B) with row k and column k deleted]
            = N * det[(I - B) with row 1 and column 1 deleted].

The eigenangles themselves never appear. What does matter is the ORDERING RULE:
B pairs eigenvector index k with diagonal slot j, so W must be the eigenvector
matrix whose columns are already in the chosen (sorted-by-argument) order.
A different ordering gives a different Z_tilde and a different J.

All functions take W (or U) with an arbitrary leading batch shape (..., N, N)
and return a real tensor of shape (...).


Relation to `_sun_group_commutator`
-----------------------------------
Two parallel problems reach the same matrix by different routes:

    spectral twist (this module):    U  ->  Z_tilde = U Lambda_U^dagger
    group commutator (that module):    (X, Y)  ->  Z = X Y X^dagger Y^dagger

where Z_tilde = Q^dagger Z Q, with Q defined below.

They are different maps with different inputs, and the bridge is conjugation
into Y's eigenframe with Q the eigenframe of Y. The computation of log-Jacobian
is in this module.

The word "twist" belongs to this module only: there is no twisting in the
group-commutator problem, and `_sun_group_commutator` speaks of conjugation.
"""

# pylint: disable=invalid-name  # for matrices like W, U, B and dims like N

import torch

from lattice_ml.functions._matrix_func import enforce_zero_sum

__all__ = [
    "log_jacobian_sun_spectral_twist",
    "log_jacobian_su2_spectral_twist",
    "log_jacobian_su3_spectral_twist",
    "log_jacobian_su2_spectral_twist_from_diag",
    "log_jacobian_su3_spectral_twist_from_diag",
    "solve_lambda_su2_spectral_twist_from_diag",
    "solve_lambda_su3_spectral_twist_from_diag",
]


# ----------------------------------------------------------------------
# B matrix
# ----------------------------------------------------------------------
def bistochastic_from_W(W: torch.Tensor) -> torch.Tensor:
    """B[..., k, j] = |W[..., j, k]|^2.

    In probability and combinatorics, a doubly stochastic matrix (also called
    bistochastic matrix) is a square matrix with nonnegative entries whose rows
    and columns each sum to 1.
    """
    # For consistency with docstring, we use W^T; it does not change det(I - B)
    Q = torch.transpose(W, -1, -2)
    return (Q.real**2 + Q.imag**2) if Q.is_complex() else Q**2


# ----------------------------------------------------------------------
# General SU(N)
# ----------------------------------------------------------------------
def jacobian_sun_spectral_twist(W: torch.Tensor) -> torch.Tensor:
    """|det dZ_tilde/dU| for SU(N), any N >= 2. W: (..., N, N) complex."""
    B = bistochastic_from_W(W)
    N = B.shape[-1]
    V = _helmert(N, B.dtype, B.device)
    eye = torch.eye(N, dtype=B.dtype, device=B.device)
    M = V.transpose(-1, -2) @ (eye - B) @ V  # (..., N-1, N-1)
    return torch.linalg.det(M)


def log_jacobian_sun_spectral_twist(W: torch.Tensor) -> torch.Tensor:
    """log |det dZ_tilde/dU|, via slogdet (stable for near-singular W)."""
    B = bistochastic_from_W(W)
    N = B.shape[-1]
    V = _helmert(N, B.dtype, B.device)
    eye = torch.eye(N, dtype=B.dtype, device=B.device)
    M = V.transpose(-1, -2) @ (eye - B) @ V  # (..., N-1, N-1)
    _, logabsdet = torch.linalg.slogdet(M)
    return logabsdet


def _helmert(N: int, dtype: torch.dtype, device) -> torch.Tensor:
    """N x (N-1) matrix with orthonormal columns spanning {h : sum h = 0}."""
    V = torch.zeros(N, N - 1, dtype=dtype, device=device)
    for k in range(1, N):
        V[:k, k - 1] = 1.0 / (k * (k + 1.0)) ** 0.5
        V[k, k - 1] = -(k / (k + 1.0)) ** 0.5
    return V


# ----------------------------------------------------------------------
# Closed forms for N = 2, 3
# ----------------------------------------------------------------------
def jacobian_su2_spectral_twist(W: torch.Tensor) -> torch.Tensor:
    """SU(2): |det dZ_tilde/dU| = 2 |W_12|^2 = 2 (1 - |W_11|^2)."""
    w01 = W[..., 0, 1]
    return 2.0 * (w01.real**2 + w01.imag**2)


def log_jacobian_su2_spectral_twist(W: torch.Tensor) -> torch.Tensor:
    """SU(2): log |det dZ_tilde/dU| = log(2 |W_12|^2) = log(2 - 2|W_11|^2)."""
    w01 = W[..., 0, 1]
    return torch.log(2.0 * (w01.real**2 + w01.imag**2))


def jacobian_su3_spectral_twist(W: torch.Tensor) -> torch.Tensor:
    """SU(3): |det dZ_tilde/dU| = 2 - sum_k |W_kk|^2 + det[|W_ij|^2]."""
    P = W.real**2 + W.imag**2  # P[..., i, j] = |W_ij|^2
    e1 = torch.einsum('...ii -> ...', P)
    e3 = torch.linalg.det(P)
    return 2.0 - e1 + e3


def log_jacobian_su3_spectral_twist(W: torch.Tensor) -> torch.Tensor:
    """SU(3): log |det dZ_tilde/dU| = log(2 - sum|W_kk|^2 + det[|W_ij|^2])."""
    P = W.real**2 + W.imag**2  # P[..., i, j] = |W_ij|^2
    e1 = torch.einsum('...ii -> ...', P)
    e3 = torch.linalg.det(P)
    return torch.log(2.0 - e1 + e3)


# ----------------------------------------------------------------------
# Triangle geometry, for the equivalent "from the twisted matrix" form
# ----------------------------------------------------------------------
# The same Jacobian has a second closed form that never mentions W. Write the
# twisted matrix in Lambda's own eigenbasis and keep only its diagonal,
#
#     diag = diag(I - Z_tilde) ,   d_k = diag[k] ,
#
# then, with v_k = d_k lambda_k (so sum_k v_k = 0 -- the closure that
# `solve_lambda_su3_spectral_twist_from_diag` inverts),
#
#     N = 2 :   |det dZ_tilde/dU| = 2 d_1 d_2 / (d_1 + d_2)
#                                   (real, since d_2 = conj d_1)
#     N = 3 :   |det dZ_tilde/dU| = 3 A_v / A_lambda
#
# with A_v the area of the closed v-triangle -- side lengths |d_1|, |d_2|,
# |d_3| -- and A_lambda the area of the triangle whose VERTICES are lambda_1,
# lambda_2, lambda_3 on the unit circle. A triangle inscribed in the unit
# circle has area abc/4, so A_lambda = |Vandermonde(lambda)| / 4.
#
# This is the SAME Jacobian in a different chart: `log_jacobian_su*` evaluates
# det(I - B) in the B chart, `log_jacobian_su*_from_diag` below evaluates it in
# the (lambda, diag) chart. The N = 3 one has to call the branch rule, because
# the two triangle chiralities give different lambda -- hence different
# A_lambda -- so the closure on its own does not pin it down.
def triangle_area_from_sides(a, b, c):
    """Heron's area of the triangle with side lengths a, b, c.

    Returns 0 rather than NaN when the side lengths violate the triangle
    inequality, so a degenerate configuration reads as zero area.
    """
    disc = 2*(a*b)**2 + 2*(b*c)**2 + 2*(c*a)**2 - a**4 - b**4 - c**4
    return 0.25 * torch.sqrt(disc.clamp_min(0))


def inscribed_triangle_area(lam: torch.Tensor) -> torch.Tensor:
    """Area of the triangle with vertices lam_1, lam_2, lam_3 in the plane.

    `lam` is complex of shape (..., 3). For points on the unit circle this is
    |Vandermonde(lam)| / 4, at most 3 sqrt(3) / 4 (equilateral).
    """
    l1, l2, l3 = lam.unbind(-1)
    return 0.5 * ((l2 - l1).conj() * (l3 - l1)).imag.abs()


# ----------------------------------------------------------------------
# The Jacobian in the (lambda, diag) chart, and the closure it rests on
# ----------------------------------------------------------------------
def log_jacobian_su2_spectral_twist_from_diag(
    diag: torch.Tensor
) -> torch.Tensor:
    r"""SU(2): log |det dZ_tilde/dU| from `diag = diag(I - Z_tilde)`.

        |det dZ_tilde/dU| = 2 d_1 d_2 / (d_1 + d_2)

    the harmonic mean of the two entries. Real, because Z_tilde is in SU(2),
    so `d_2 = conj(d_1)` and the value is `|d_1|^2 / Re(d_1)`, whose
    denominator is `det(I - Z_tilde) = 2 - Tr Z_tilde`.

    This returns exactly the same number as
    `log_jacobian_su2_spectral_twist`, from the other side of the map: that
    one takes W, U's eigenframe, while this one takes the diagonal of the
    twisted matrix. Same sign, same value -- the two are interchangeable.

    Use this one when UNTWISTING. Going that way Z_tilde is what is in hand
    and W is not, so the W form would have to reconstruct it first; this form
    reads the answer straight off what the untwist direction already has.

    No Lambda is needed: at N = 2 the closure `sum_k d_k lambda_k = 0` has
    only two terms, so Lambda cancels between numerator and denominator
    instead of entering through a triangle.
    """
    d_1, d_2 = diag.unbind(-1)
    return torch.log((2 * d_1 * d_2 / (d_1 + d_2)).real)


def log_jacobian_su3_spectral_twist_from_diag(
    diag: torch.Tensor, descending_angle: bool = False
) -> torch.Tensor:
    r"""SU(3): log |det dZ_tilde/dU| from `diag = diag(I - Z_tilde)`.

        |det dZ_tilde/dU| = 3 A_v / A_lambda = 12 A_v / |Vandermonde(lambda)|

    with `A_v` the area of the closed triangle whose edges are
    `v_k = diag_k lambda_k` -- side lengths `|diag_k|` -- and `A_lambda` the
    area of the triangle with vertices `lambda_1, lambda_2, lambda_3`.

    This returns exactly the same number as
    `log_jacobian_su3_spectral_twist`, from the other side of the map: that
    one takes W, U's eigenframe, while this one takes the diagonal of the
    twisted matrix. Same sign, same value -- the two are interchangeable.

    Use this one when UNTWISTING. Going that way Z_tilde is what is in hand
    and W is not, so the W form would have to reconstruct it first; and the
    branch solve this needs is work the untwist direction is doing anyway.

    The W-form value is `log(2 - Tr B + det B)`. The route between the charts:
    `B` is doubly stochastic, so row `i` of `B` is the unique barycentric
    coordinate vector of `(1 - diag_i) lambda_i` in the triangle
    `(lambda_1, lambda_2, lambda_3)`; then `A = I - B` has zero row and column
    sums, `|det dZ_tilde/dU| = e_2(A)`, and `A^T` factors as a ratio of two
    2x2 determinants that are twice the two areas.
    """
    lam = solve_lambda_su3_spectral_twist_from_diag(
        diag, descending_angle=descending_angle
    )
    A_v = triangle_area_from_sides(*diag.abs().unbind(-1))
    return torch.log(3 * A_v / inscribed_triangle_area(lam))


# =============================================================================
# Solve lambda for spectral untwist
# =============================================================================
def solve_lambda_su2_spectral_twist_from_diag(diag, descending_angle=False):
    """Invert the twist for SU(2): recover Lambda from `diag(I - Z_tilde)`.

    Solves the closure `sum_k d_k lambda_k = 0` with `prod_k lambda_k = 1`.
    Z_tilde is in SU(2), so `d_2 = conj(d_1)` and the closure collapses to
    `Re(d_1 lambda_1) = 0`, leaving `lambda_1 = +- i conj(d_1) / |d_1|`.

    The two roots are the SU(2) centre, Lambda and -Lambda -- the whole
    spectrum flips sign together. `descending_angle` picks the branch; False
    (default) puts the first eigenangle in (-pi, 0).

    `solve_lambda_su2(Z, Q)` in `_sun_group_commutator` is the entry point
    that builds `diag` from a commutator's (Z, Q) and calls this.
    """
    d_1 = diag[..., 0]
    fac = 1j if descending_angle else -1j  # rotation factor
    lam_1 = fac * d_1.conj() / d_1.abs()
    return torch.stack([lam_1, lam_1.conj()], dim=-1)


def solve_lambda_su3_spectral_twist_from_diag(diag, descending_angle=False):
    """Invert the twist for SU(3): recover Lambda from `diag(I - Z_tilde)`.

    Solves the closure `sum_k diag_k lambda_k = 0` with `prod_k lambda_k = 1`,
    i.e. finds the spectrum Lambda for which the given Z_tilde equals
    `U Lambda^dagger` with `spec(U) = Lambda`.

    Geometrically it is triangle closure: the side lengths `|diag_k|` fix the
    triangle up to 6 choices -- 3 SU(3)-centre rotations times 2 chiralities --
    and the zero-sum-angle sort picks one. That sort is not cosmetic: the
    centre rotations leave `A_lambda` alone, but the two chiralities give
    genuinely different Lambda and therefore a different Jacobian, so the
    branch has to be pinned. Plain angle order would be ambiguous whenever
    Lambda's eigenangles are nearly degenerate, since a centre rotation can
    then leave the angle order unchanged.

    `solve_lambda_su3(Z, Q)` in `_sun_group_commutator` is the entry point that
    builds `diag` from a commutator's (Z, Q) and calls this.
    """
    a, b, c = diag.abs().unbind(-1)

    # Angle between the first two triangle sides. Clamp against a nearly
    # degenerate triangle (two of a, b, c nearly equal), which can push
    # the ratio just outside [-1, 1] by rounding error.
    cos_delta = (a**2 + b**2 - c**2) / (2 * a * b)
    delta = torch.pi - torch.arccos(cos_delta.clamp(-1, 1))

    # Both triangle chiralities and three possible SU(3) roots.
    kwargs = {'dtype': diag.dtype, 'device': diag.device}
    signs = torch.tensor((1, -1), **kwargs)
    roots = 2 * torch.pi / 3 * torch.tensor([0, 1, 2], **kwargs)

    v1 = a[..., None] * torch.exp(1j * signs * delta[..., None])
    v2 = b[..., None].expand_as(v1)

    u = torch.stack([v1, v2, -v1 - v2], dim=-1) / diag[..., None, :]
    # u shape here: (..., 2[chirality], 3[eigenvalue])

    # Enforce det(u) = 1: three possible SU(3) roots.
    phase = -torch.angle(u).mean(dim=-1)[..., :, None] + roots
    # phase shape: (..., 2[chirality], 3[root])

    u = u[..., :, None, :] * torch.exp(1j * phase[..., :, :, None])

    # u shape: (..., 2 chirality, 3 roots, 3 eigenvalues)
    u = u.flatten(-3, -2)

    angles = enforce_zero_sum(torch.angle(u), dim=-1)

    if descending_angle:
        sorted_ = (angles[..., :-1] >= angles[..., 1:]).all(dim=-1)
    else:
        sorted_ = (angles[..., :-1] <= angles[..., 1:]).all(dim=-1)

    # argmax gives the first valid candidate, or 0 if none are valid.
    best = sorted_.to(torch.uint8).argmax(dim=-1)

    return u.gather(
        -2, best[..., None, None].expand(*best.shape, 1, 3)
    ).squeeze(-2)


# =============================================================================
# From U directly
# =============================================================================
def eigendecompose_sorted(U: torch.Tensor):
    """
    Eigendecomposition of unitary U with eigenvalues sorted by ascending
    argument.

    Returns (lam, W) with U = W diag(lam) W^dagger.
    """
    lam, W = torch.linalg.eig(U)
    idx = torch.argsort(torch.angle(lam), dim=-1)
    lam = torch.gather(lam, -1, idx)
    W = torch.gather(W, -1, idx.unsqueeze(-2).expand_as(W))
    return lam, W


def twist_eigenvalues(U: torch.Tensor) -> torch.Tensor:
    """Z_tilde = U Lambda^dagger."""
    lam, _ = eigendecompose_sorted(U)
    return U @ torch.diag_embed(lam.conj())


def jacobian_spectral_twist_from_U(U: torch.Tensor) -> torch.Tensor:
    """|det dZ_tilde/dU| computed straight from U."""
    _, W = eigendecompose_sorted(U)
    return jacobian_sun_spectral_twist(W)
