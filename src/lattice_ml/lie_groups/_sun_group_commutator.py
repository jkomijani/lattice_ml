# Copyright (c) 2026 Javad Komijani

r"""SU(N) group-commutator encode/decode via zero-sum-angle ordering.

    (X, Y) -> Z = X Y X^dagger Y^dagger

with the reparametrization (X, Y) <-> (D, Q, Z), where Q is Y's eigenframe.


Relation to `_spectral_twist`
-----------------------------
Two parallel problems reach the same matrix by different routes:

    group commutator (this module):    (X, Y)  ->  Z = X Y X^dagger Y^dagger
    spectral twist (that module):    U  ->  Z_tilde = U Lambda_U^dagger

where Z_tilde = Q^dagger Z Q, with Q defined below.

They are different maps with different inputs, and the bridge is conjugation
into Y's eigenframe with Q the eigenframe of Y. The computation of log-Jacobian
is in the other module.
"""

# pylint: disable=invalid-name

from typing import Callable
import torch

from lattice_ml.functions._matrix_func import enforce_zero_sum
from ._spectral_twist import (
    log_jacobian_sun_spectral_twist,
    log_jacobian_su2_spectral_twist,
    log_jacobian_su3_spectral_twist,
    solve_lambda_su2_spectral_twist_from_diag,
    solve_lambda_su3_spectral_twist_from_diag,
)


__all__ = [
    'encode_sun_group_commutator',
    'decode_sun_group_commutator',
]


# =============================================================================
def encode_sun_group_commutator(
    X, Y, return_logj=False, randomize_vector_phase=True
):
    """
    Compute the SU(N) group commutator `Z = X Y X^dagger Y^dagger`
    and the data needed to invert this map back to (X, Y).

    Conceptually (not literally followed in code -- Z is computed directly):
    0. (X, Y)
    1. (X, lam, Q) by eigendecomposition of Y as Y = Q diag(lam) Q^dagger.
    2. (X_tilde, lam, Q) by conjugation of X as X_tilde = Q^dagger X Q.
    3. (X_tilde_0, D, lam, Q) by collecting the phase and permutation.
    4. (U, D, Q) by eigen composition of lam & X_tilde_0.
    5. (Z_tilde, D, Q) by the spectral twist of U.
    6. (Z, D, Q) by conjugation of Z_tilde as Z = Q Z_tilde Q^dagger.

    Args:
        X: SU(N) matrices, shape (..., N, N); N in {2, 3}.
        Y: SU(N) matrices, shape (..., N, N), same batch shape as X.
        return_logj: Also return the log-Jacobian of (X, Y) -> (D, Q, Z).
        randomize_vector_phase: Q's column-phase convention;
            an independent random phase per column per call if True (default),
            else the eigensolver's own deterministic phase.

    Returns:
        D: Diagonal SU(N) matrix.
        Q: SU(N) matrix, Y's eigenframe.
        Z: SU(N) matrix, the group commutator.
        logj: (only if return_logj) log-Jacobian of (X, Y) -> (D, Q, Z),
            returned alongside (D, Q, Z) as a nested tuple.
    """
    Z = X @ Y @ X.adjoint() @ Y.adjoint()

    eig_options = {'sort_by': 'zero-sum-angle', 'project_to_sun': True}
    _, Q = eig_with_options(
        Y, randomize_vector_phase=randomize_vector_phase, **eig_options
    )
    _, P = eig_with_options(Z @ Y, **eig_options)

    D = P.adjoint() @ X @ Q

    if not return_logj:
        return D, Q, Z

    X_tilde = Q.adjoint() @ X @ Q
    logj_encoder = compute_logj_sun_group_commutator_encoder(X_tilde)

    return (D, Q, Z), logj_encoder


# =============================================================================
def decode_sun_group_commutator(D, Q, Z, return_logj=False):
    """
    Construct (X, Y) from (D, Q, Z), the exact inverse of
    `encode_sun_group_commutator`, subject to Z having no unit eigenvalues.

    Y's eigenvalues are solved from Z using Q (Y's eigenframe); X is then
    rebuilt from D and other quantities.

    Args:
        D: Diagonal SU(N) matrix, as returned by `encode_sun_group_commutator`.
        Q: SU(N) matrix, Y's eigenframe.
        Z: SU(N) matrix, the group commutator.
        return_logj: Also return the log-Jacobian of (D, Q, Z) -> (X, Y).

    Returns:
        X: SU(N) matrix.
        Y: SU(N) matrix.
        logj: (only if return_logj) log-Jacobian of (D, Q, Z) -> (X, Y),
            returned alongside (X, Y) as a nested tuple.
    """
    # Part 1: Construct Y
    lam = solve_lambda_constraint(Z, Q)  # Y's eigenvalues, in Q's order
    Y = Q @ (lam[..., None] * Q.adjoint())

    # Part 2: Construct X
    eig_options = {'sort_by': 'zero-sum-angle', 'project_to_sun': True}
    _, P = eig_with_options(Z @ Y, **eig_options)
    X = P @ D @ Q.adjoint()

    if not return_logj:
        return X, Y

    X_tilde = Q.adjoint() @ X @ Q
    logj_encoder = compute_logj_sun_group_commutator_encoder(X_tilde)
    logj_decoder = - logj_encoder

    return (X, Y), logj_decoder


# =============================================================================
def compute_logj_sun_group_commutator_encoder(W):
    """
    Dispatch to the N-specific closed form spectral-twist log-Jacobian,
    given X conjugated into Y's eigenframe, `W = X_tilde = Q^dagger X Q`.
    """
    n = W.shape[-1]
    if n == 2:
        compute_logj = log_jacobian_su2_spectral_twist
    elif n == 3:
        compute_logj = log_jacobian_su3_spectral_twist
    else:
        compute_logj = log_jacobian_sun_spectral_twist

    return compute_logj(W)


# =============================================================================
def solve_lambda_constraint(Z, Q):
    """Dispatch to the N-specific solver for Y's eigenvalues."""
    n = Z.shape[-1]
    if n == 2:
        solve_lambda = solve_lambda_su2
    elif n == 3:
        solve_lambda = solve_lambda_su3
    else:
        raise ValueError(f"N = {n} is not supported")
    return solve_lambda(Z, Q)


# =============================================================================
def solve_lambda_su2(Z, Q, descending_angle: bool = False):
    """
    Solve `Tr[(I - Q† Z Q) Λ] = 0` for diagonal Λ in SU(2); input
    matrices Z and Q are SU(2).

    Because `Q† Z Q` is in SU(2), and with with `m = diag(I - Q† Z Q)`,
    we have m_2 = conj(m_1), reducing the constraint to `Re(m_1 u_1) = 0`.
    There are two solutions, the SU(2) center {Y, -Y}.
    The option `descending_angle` specifies the branch: False (default) puts
    the first eigenvalue's angle in (-pi, 0).
    """
    # Near Z = I, conjugating (I - Z) is more precise than (I - Q† Z Q).
    eye = torch.eye(2, dtype=Z.dtype, device=Z.device)
    diag = torch.diagonal(Q.adjoint() @ (eye - Z) @ Q, dim1=-2, dim2=-1)
    return solve_lambda_su2_spectral_twist_from_diag(
        diag, descending_angle=descending_angle
    )


# =============================================================================
def solve_lambda_su3(Z, Q, descending_angle=False):
    """
    Solve `Tr[(I - Q† Z Q) Λ] = 0` for diagonal Λ in SU(3); input
    matrices Z and Q are SU(3).

    Because `Q† Z Q` is in SU(3), the problem reduces to triangle closure
    (sides a, b, c from the diagonal's magnitudes), fixing Λ up to 6 choices:
    3 choices of SU(3)-center rotation (root) times 2 choices of triangle
    chirality (relative sign of a vs b). Searches the 3 center roots times
    2 triangle chiralities and returns the uniquely zero-sum-angle-sorted
    candidate (see `enforce_zero_sum`): plain angle order is ambiguous
    whenever Λ's eigenangles are nearly degenerate, since a center rotation
    can then leave the angle order unchanged.
    """
    # Near Z = I, conjugating (I - Z) is more precise than (I - Q† Z Q).
    eye = torch.eye(3, dtype=Z.dtype, device=Z.device)
    diag = torch.diagonal(Q.adjoint() @ (eye - Z) @ Q, dim1=-2, dim2=-1)
    return solve_lambda_su3_spectral_twist_from_diag(
        diag, descending_angle=descending_angle
    )


# =============================================================================
def eig_with_options(
    X: torch.Tensor,
    sort_by: str | None = None,
    descending: bool = False,
    randomize_vector_phase: bool = False,
    randomize_order: bool = False,
    project_to_sun: bool = False,
    eig_fn: Callable | None = None,
):
    """Eigendecomposition with optional sorting, randomizing, or projecting
    of eigenvectors.

    Args:
        X: Square matrices with shape [..., N, N].
        sort_by: Sort eigenpairs by 'real', 'imag', 'abs', 'angle', or
            'zero-sum-angle'.
        descending: Whether to sort in descending order.
        randomize_vector_phase: Apply random phases/signs to eigenvectors.
        randomize_order: Randomly permute eigenpairs.
        project_to_sun: Project the eigenvector matrix to SU(N).
        eig_fn: Eigendecomposition callable; defaults to `torch.linalg.eig`.

    Notes:
        `randomize_order` and `sort_by` are mutually exclusive.
        `project_to_sun` uses `project_unitary_to_sun(eigvecs, col_ind=0)`.

    Returns eigenvalues [..., N] and eigenvectors [..., N, N] (as columns).
    """
    eigvals, eigvecs = (eig_fn or torch.linalg.eig)(X)

    if randomize_order and sort_by is not None:
        raise ValueError("randomize_order & sort_by are mutually exclusive")

    if randomize_order:
        perm = torch.argsort(torch.rand_like(eigvals.real), dim=-1)
        eigvals = torch.gather(eigvals, -1, perm)
        eigvecs = torch.gather(
            eigvecs, -1, perm.unsqueeze(-2).expand_as(eigvecs)
        )

    if sort_by is not None:
        eigvals, eigvecs = sort_eig(
            eigvals, eigvecs, sort_by=sort_by, descending=descending
        )

    if randomize_vector_phase:
        if eigvecs.is_complex():
            phases = torch.exp(1j * 2*torch.pi * torch.rand_like(eigvals.real))
        else:
            phases = 2 * torch.randint_like(eigvals.real, 0, 2) - 1
        eigvecs = eigvecs * phases.unsqueeze(-2)

    if project_to_sun:
        eigvecs = project_unitary_to_sun(eigvecs, col_ind=0)

    return eigvals, eigvecs


compute_eig = eig_with_options  # alias


# =============================================================================
def sort_eig(eigvals, eigvecs, descending=False, sort_by='real'):
    """Sort a batch of eigenvalues and permute the corresponding eigenvectors.

    Args:
        eigvals (Tensor): (..., n) tensor of eigenvalues.
        eigvecs (Tensor): (..., n, n) tensor of eigenvectors (as columns).
        descending (bool): sorting order. (Default is False, i.e. ascending.)
        sort_by (str): key used for sorting: 'real', 'imag', 'abs', 'angle',
            or 'zero-sum-angle'.

    Returns:
        Tensor, Tensor: the sorted eigenvalues and permuted eigenvectors.
    """
    eigvals, sorted_ind = sort(eigvals, descending=descending, sort_by=sort_by)

    n = eigvecs.shape[-1]
    sorted_ind = sorted_ind.unsqueeze(-2).repeat(*[1]*(eigvecs.ndim - 2), n, 1)
    eigvecs = eigvecs.gather(-1, sorted_ind)
    return eigvals, eigvecs


# =============================================================================
def sort(x: torch.Tensor, dim=-1, descending=False, sort_by='real'):
    """Sort a (possibly complex) tensor by real part, imaginary part,
    magnitude, phase, or zero-sum phase.

    'zero-sum-angle' sorts by phase after enforcing a zero sum along `dim`
    (see `enforce_zero_sum`), which disambiguates SU(N)-center rotations that
    plain phase sorting cannot (N > 2).

    Returns the sorted `x` and the sorting indices along `dim`.
    """

    if sort_by == 'zero-sum-angle' and x.shape[-1] == 2:
        # already zero-sum: SU(2) eigenangles are an exact (-a, a) pair.
        sort_by = 'imag'  # or 'angle'

    if sort_by == 'zero-sum-angle':
        key = enforce_zero_sum(torch.angle(x), dim=dim)
    elif sort_by in ['real', 'imag', 'abs', 'angle']:
        key = getattr(torch, sort_by)(x)
    else:
        raise ValueError(f"Invalid sort_by: {sort_by}")

    _, sorted_ind = torch.sort(key, dim=dim, descending=descending)

    return x.gather(-1, sorted_ind), sorted_ind


# =============================================================================
def project_unitary_to_sun(
    X: torch.Tensor,
    col_ind: int | None = None,
    row_ind: int | None = None
):
    """Rescale a unitary matrix so its determinant is 1 (U(N) -> SU(N)).

    If both indices are None, det(X)^(-1/N) is spread evenly over all entries.
    (An N-th root is multi-valued, so it can jump discontinuously if det(X)
    lands near the branch cut.)

    If `col_ind`/`row_ind` is given, the whole det(X)^(-1) correction goes on
    that single column/row instead -- single-valued.

    Args:
        X: (..., N, N) unitary matrices.
        col_ind, row_ind: column/row to absorb the correction (at most one).

    Returns:
        (..., N, N) unitary matrices with determinant 1.
    """
    if col_ind is not None and row_ind is not None:
        raise ValueError("Pass at most one of col_ind, row_ind.")

    n = X.shape[-1]
    det = torch.linalg.det(X)

    if col_ind is None and row_ind is None:
        return X * det.pow(-1 / n)[..., None, None]

    mask = torch.zeros(n, dtype=X.dtype, device=X.device)
    mask[col_ind if col_ind is not None else row_ind] = 1.0
    scale = 1 + mask * (det.conj()[..., None] - 1)
    scale = scale.unsqueeze(-2 if row_ind is None else -1)

    return scale * X
