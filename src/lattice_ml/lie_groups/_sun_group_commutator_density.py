# Copyright (c) 2026 Claude, under supervision of Javad Komijani

r"""
Group commutator density.

Let X, Y in SU(N) be independently distributed according to the normalized
Haar measure dX and dY. Define the group commutator map

    Phi: SU(N) x SU(N) -> SU(N),   Phi(X, Y) = X Y X† Y†.

The corresponding random matrix is Z = Phi(X, Y) = X Y X† Y†.

The pushforward of the product Haar measure under Phi is a measure on SU(N).
We define the *group commutator density* J(Z) by

    Phi_*(dX dY) = J(Z) dZ,

dZ the normalized Haar measure on SU(N). Equivalently,

    J(Z) = integral dX dY delta_{SU(N)}(Z† X Y X† Y†)
         = sum_R chi_R(Z†) / d_R

(sum over irreducible representations R, dimension d_R, character chi_R),
via the character expansion of the group delta function and two applications
of Schur orthogonality.

J(Z) is the probability density of the group commutator X Y X† Y† when
X, Y are independently Haar distributed: for any integrable f,

    integral dX dY f(X Y X† Y†) = integral dZ f(Z) J(Z).

Since the product Haar measure is invariant under simultaneous conjugation,
J(Z) is a *class function*: J(g Z g†) = J(Z) for all g, Z in SU(N).

Only SU(2) and SU(3) are implemented, both in exact closed form. Writing
phi_1, ..., phi_N in [0, 2pi) for the eigen-angles of Z, both read

    J(Z) = (1/N) prod_{i<j} [ (phi_i - phi_j)/2 ] / sin[ (phi_i - phi_j)/2 ]

i.e. one factor x/(2 sin(x/2)) per pair of eigenvalues. (Verified for
N = 2, 3 only; not claimed for N > 3.)
"""

# pylint: disable=invalid-name

import math
import warnings
import torch


__all__ = [
    "compute_sun_group_commutator_log_density",
    "compute_sun_group_commutator_log_density_from_eigvals"
]


# =============================================================================
def compute_sun_group_commutator_log_density(Z: torch.Tensor) -> torch.Tensor:
    r"""
    log J(Z), the log of the group commutator density (see module docstring):

        J(Z) = integral dX dY delta_{SU(N)}(Z† X Y X† Y†)

    for Haar-random, independent X, Y in SU(N). Exact for both N = 2 and N = 3.

    Args:
        Z: Special unitary input matrix of shape `(..., N, N)`.

    Returns:
        torch.Tensor: log J(Z), shape `Z.shape[:-2]`.
    """
    return compute_sun_group_commutator_log_density_from_eigvals(
        torch.linalg.eigvals(Z)
    )


# =============================================================================
def compute_sun_group_commutator_log_density_from_eigvals(
    eigvals: torch.Tensor
) -> torch.Tensor:
    r"""
    log J(Z) from the eigenvalues of Z (see module docstring):

        J(Z) = (1/N) prod_{i<j} [ (phi_i - phi_j)/2 ] / sin[ (phi_i - phi_j)/2]

    with phi_k in [0, 2pi) the eigen-angles. Exact for N = 2 and N = 3.

    Args:
        eigvals: Eigenvalues of SU(N) matrix of shape `(..., N, N)`.

    Returns:
        torch.Tensor: log J(Z), shape `eigvals.shape[:-1]`.
    """
    N = eigvals.shape[-1]
    if N > 3:
        warnings.warn(
            f"The group commutator density is derived only for SU(2) and "
            f"SU(3); for SU({N}) this expression is only a guess.",
        )
    phi = torch.angle(eigvals) % (2 * math.pi)
    diff = phi.unsqueeze(-1) - phi.unsqueeze(-2)
    i, j = torch.triu_indices(N, N, offset=1)
    log_sinc = torch.log(torch.sinc(diff[..., i, j] / (2 * math.pi)))
    return -math.log(N) - log_sinc.sum(-1)


# =============================================================================
def _su2_log_density(Z: torch.Tensor) -> torch.Tensor:
    r"""
    Exact closed form for SU(2): with theta the class angle of Z
    (eigenvalues e^{+-i theta}, theta in [0, pi]),

        J(theta) = (pi - theta) / (2 sin theta).
    """
    cos_theta = torch.einsum('...ii->...', Z).real / 2
    theta = torch.arccos(cos_theta.clamp(-1, 1))
    return (
        torch.log(torch.pi - theta) - math.log(2) - torch.log(torch.sin(theta))
    )


# =============================================================================
def _su3_log_density(Z: torch.Tensor) -> torch.Tensor:
    r"""
    Exact closed form for SU(3): with phi_1, phi_2, phi_3 in [0, 2pi) the
    eigen-angles of Z,

        J(Z) = (1/3) prod_{i<j} [ (phi_i - phi_j)/2 ] / sin[ (phi_i - phi_j)/2]

    the same one-factor-per-pair shape as `_su2_log_density` (see the module
    docstring). Each factor is even in its argument, so neither the ordering
    of the eigenvalues nor -- because `det Z = 1` -- the 0 vs 2pi ambiguity of
    an eigenvalue at 1 affects the result.

    Derivation:
    -----------
    The Weyl character formula turns `sum_R chi_R(Z) / d_R` into a single
    lattice sum over weights, since `d_R = prod_{i<j}(l_i - l_j) / 2` cancels
    against the Vandermonde `Delta(x) = prod_{i<j}(x_i - x_j)`:

        J(Z) Delta = 2 sum_n x^n / [(n_1-n_2)(n_1-n_3)(n_2-n_3)]

    summed over integer triples `n` with distinct entries, modulo an overall
    shift. Writing `J Delta = 2i H` with `n = (p, q, 0)` and differentiating,
    the substitution `r = p - q` factorizes each derivative into two
    sawtooth series `S(x) = sum_{m!=0} e^{imx}/m` plus the excluded diagonal
    `C(x) = sum_{m!=0} e^{imx}/m^2`:

        d_1 H = sigma(theta_1) sigma(theta_3) + C(theta_2),   S = i sigma

    Both are *piecewise quadratic polynomials* (no Clausen functions survive),
    so H is piecewise cubic; the integration constant is fixed by evaluating
    the lattice sum on the wall `theta_2 = 0`, where the inner sum telescopes
    to `-2/p^3`. The pieces then assemble into the product above.

    Notes
    -----
    `Z = 1` is a genuine singularity, `J = inf`; there this expression instead
    returns the finite value 1/3, as all `phi_i` collapse to 0 and the lift
    that produces the divergence is lost.

    Parameters
    ----------
    Z : torch.Tensor
        Special unitary input matrix of shape `(..., 3, 3)`.

    Returns
    -------
    torch.Tensor
        log J(Z), shape `Z.shape[:-2]`.
    """
    phi = torch.angle(torch.linalg.eigvals(Z)) % (2 * math.pi)
    diff = phi.unsqueeze(-1) - phi.unsqueeze(-2)
    i, j = torch.triu_indices(3, 3, offset=1)
    # x / (2 sin(x/2)) = 1 / sinc(x / 2pi), with torch's normalized sinc.
    log_sinc = torch.log(torch.sinc(diff[..., i, j] / (2 * math.pi)))
    return -math.log(3) - log_sinc.sum(-1)
