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
import torch


__all__ = ["compute_sun_group_commutator_log_density"]


# =============================================================================
def compute_sun_group_commutator_log_density(Z: torch.Tensor) -> torch.Tensor:
    r"""
    log J(Z), the log of the group commutator density (see module docstring):

        J(Z) = integral dX dY delta_{SU(N)}(Z† X Y X† Y†)

    for Haar-random, independent X, Y in SU(N). Exact for both N = 2 and N = 3.

    Parameters
    ----------
    Z : torch.Tensor
        Special unitary input matrix of shape `(..., N, N)`, `N in {2, 3}`.

    Returns
    -------
    torch.Tensor
        log J(Z), shape `Z.shape[:-2]`.
    """
    N = Z.shape[-1]
    if N == 2:
        return _su2_log_density(Z)
    if N == 3:
        return _su3_log_density(Z)
    raise NotImplementedError("Implemented only for SU(2) and SU(3).")


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


# =============================================================================
def su3_log_density_heat_kernel(
    Z: torch.Tensor,
    t: float = 0.05,
    p_max: int = 20,
) -> torch.Tensor:
    r"""
    Heat-kernel-regularized, truncated evaluation of SU(3)'s group commutator
    density,

        J(Z) = 2 sum_{p,q=0}^inf chi_{p,q}(Z) / [(p+1)(q+1)(p+q+2)]

    (using chi_{p,q}(Z) rather than chi_{p,q}(Z†) -- summing symmetrically
    over all (p, q) makes these equal, since the representation set is closed
    under conjugation, (p,q) <-> (q,p)).

    **Superseded by `_su3_log_density`, which evaluates the same quantity in
    exact closed form.** Kept for cross-checking: this is an independent
    route to J(Z), and the two agree in the `t -> 0` limit.

    The series is only *conditionally* convergent, so it is regularized by
    damping each term with `exp(-t * C_2(p, q))`, `C_2` the quadratic Casimir
    -- the standard heat-kernel regularization of the group delta function.
    The sum is then absolutely and rapidly convergent for any `t > 0`, and is
    truncated at `p, q <= p_max`.

    The normalization `int J dZ = 1` holds exactly for every `t`, but the
    *shape* carries an O(t) regulator bias that vanishes only linearly: on a
    Haar sample, J comes out 9.39 / 10.14 / 10.67 / 11.06 / 11.33 at
    `t = 0.02 / 0.01 / 0.005 / 0.0025 / 0.00125` (with `p_max` raised to keep
    `t * C_2(p_max, p_max) >~ 20`) against the exact 11.89. Reaching a few
    digits therefore costs a large `p_max`, which is what makes the closed
    form worth having.

    Parameters
    ----------
    Z : torch.Tensor
        Special unitary input matrix of shape `(..., 3, 3)`.
    t : float, default=0.05
        Heat-kernel regulator; must be > 0.
    p_max : int, default=20
        Truncation of the (p, q) sum (inclusive).

    Returns
    -------
    torch.Tensor
        log J(Z) (regularized, truncated), shape `Z.shape[:-2]`.
    """
    eigvals = torch.linalg.eigvals(Z)

    total = torch.zeros(Z.shape[:-2], dtype=eigvals.real.dtype)
    for p in range(p_max + 1):
        for q in range(p_max + 1):
            chi = su3_character(p, q, eigvals).real
            weight = math.exp(-t * su3_casimir(p, q)) / su3_dim(p, q)
            total = total + chi * weight

    # NOTE: `su3_dim(p, q) = (p+1)(q+1)(p+q+2)/2`, so `chi / su3_dim` already
    # carries the factor of 2 written in the series above; `total` is thus
    # `sum_R chi_R / d_R = J(Z)`. Multiplying by 2 again here made J(Z) twice
    # too large.
    return torch.log(total)


# =============================================================================
def su3_character(p: int, q: int, eigvals: torch.Tensor) -> torch.Tensor:
    r"""
    Weyl character formula for the SU(3) irrep with Dynkin indices (p, q).

    Parameters
    ----------
    p, q : int
        Dynkin indices, `p, q >= 0`. `(1, 0)` is the fundamental, `(0, 1)`
        the antifundamental, `(1, 1)` the adjoint.
    eigvals : torch.Tensor
        Eigenvalues of an SU(3) matrix, shape `(..., 3)`, product 1.

    Returns
    -------
    torch.Tensor
        chi_{p,q}(Z), shape `eigvals.shape[:-1]`.
    """
    x, y, z = eigvals.unbind(-1)
    a, b = p + q + 2, q + 1

    def det3(row0, row1, row2):
        (a0, a1, a2), (b0, b1, b2), (c0, c1, c2) = row0, row1, row2
        return (
            a0 * (b1 * c2 - b2 * c1)
            - a1 * (b0 * c2 - b2 * c0)
            + a2 * (b0 * c1 - b1 * c0)
        )

    ones = torch.ones_like(x)
    num = det3(
        (x ** a, x ** b, ones), (y ** a, y ** b, ones), (z ** a, z ** b, ones)
    )
    den = det3((x ** 2, x, ones), (y ** 2, y, ones), (z ** 2, z, ones))
    return num / den


# =============================================================================
def su3_dim(p: int, q: int) -> float:
    """Dimension of the SU(3) irrep (p, q)."""
    return (p + 1) * (q + 1) * (p + q + 2) / 2


# =============================================================================
def su3_casimir(p: int, q: int) -> float:
    """
    Quadratic Casimir of the SU(3) irrep (p, q), normalized so

        C_2(fundamental) = 4/3, C_2(adjoint) = 3

    with the standard Tr(T^aT^b) = delta^{ab}/2 convention.
    """
    return (p ** 2 + q ** 2 + p * q) / 3 + p + q
