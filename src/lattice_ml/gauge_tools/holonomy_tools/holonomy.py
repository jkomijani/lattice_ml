# Copyright (c) 2026 Javad Komijani
#
# Created by Javad Komijani, Jun 2026
# Modifed to take care of corners, Aug 2026

r"""
Prelink Holonomy
================

Given prelinks V_mu(x) in group G on the extended lattice (see `prelinks.py`),
the *prelink holonomy* is defined as:

    h_{mu, nu}(x) = V_mu(x) V_nu(x)^\dagger

Geometric interpretation
------------------------

In the gauge-fixed prelink construction, V_mu(x) represents the path-ordered
product of links (from a global or a semi-global origin) to x, arriving via
the mu-direction. Thus h_{mu, nu}(x) is the holonomy of the closed loop

    origin  --(mu)-->  x  --(nu, reversed)-->  origin

a gauge-invariant Wilson-loop-like quantity probing the geometry between two
integration paths.

Symmetries
----------

**Ordinary gauge symmetry** — Under V_mu(x) -> V_mu(x) Omega(x)†,
the factors Omega cancel:

    h_{mu, nu}(x)  ->  V_mu(x) Omega†(x) Omega(x) V_nu(x)†
                    =  h_{mu, nu}(x)

so h is *invariant*.

**Semi-global symmetry** — Under V_mu(x) -> Q(x, mu) V_mu(x), where
Q is constant along the equivalence class [x, mu], the holonomy
transforms covariantly:

    h_{mu, nu}(x)  ->  Q(x, mu) h_{mu, nu}(x) Q(x, nu)†

This reduces to a global covariance once the semi-global symmetry is
reduced (see `reduce_prelink_symmetry_to_global` in `prelinks.py`).

**Hermitian property** — Exchanging indices gives the adjoint:

    h_{mu, nu}(x) = h_{nu, mu}(x)†

**Factorization property** — As a matrix in (mu, nu) indices, it factorizes as:

    h_{mu, nu}(x) = h_{mu, 0}(x) h_{nu, 0}(x)†


Conventions
-----------

- Spatial axes may precede (`sites_before_link=True`) or follow the link axis.
- The first `prefix_dims` axes are batch/channel dimensions.
- All tensors have matrix components of size Nc x Nc.
- V and h live on the extended lattice of size (N_mu + 1) along each direction.
"""

# pylint: disable=invalid-name

import torch

from lattice_ml.functions import solve_sun_commutator

from ..prelink_tools.prelinks import link_to_prelink, prelink_to_link


__all__ = [
    "link_to_holonomy",
    "holonomy_to_link",
    "prelink_to_holonomy",
    "holonomy_to_prelink",
    "compute_h_munu",
]


# =============================================================================
def link_to_holonomy(
    U: torch.Tensor,
    prefix_dims: int = 1,
    sites_before_link: bool = True,
) -> torch.Tensor:
    """
    This function composes the maps

        U → V = link_to_prelink(U)
        V → h = prelink_to_holonomy(V)

    See those functions for more details.
    """
    kws = {'prefix_dims': prefix_dims, 'sites_before_link': sites_before_link}
    V = link_to_prelink(U, transverse_boundary_mode='periodic', **kws)
    return prelink_to_holonomy(V, **kws)


# =============================================================================
def prelink_to_holonomy(
    V: torch.Tensor,
    prefix_dims: int = 1,
    sites_before_link: bool = True,
) -> torch.Tensor:
    r"""
    Compute h_{mu, 0}(x) = V_mu(x) V_0(x)† for all mu != 0.

    Parameters
    ----------
    V : torch.Tensor
        Prelink tensor with shape
        ``(...batch..., n0+1, ..., nd+1, ndim, Nc, Nc)``
        if ``sites_before_link=True``, else
        ``(...batch..., ndim, n0+1, ..., nd+1, Nc, Nc)``.

    prefix_dims : int, default=1
        Number of leading batch/channel dimensions.

    sites_before_link : bool, default=True
        If True, spatial axes precede the link-direction axis.

    Returns
    -------
    torch.Tensor
        Holonomy h with same shape as V but ndim-1 entries on the direction
        axis.
    """
    # Axis corresponding to the link direction
    link_axis = -3 if sites_before_link else prefix_dims
    spatial_ndim = V.shape[link_axis]

    V_0 = torch.narrow(V, link_axis, 0, 1)
    V_mu = torch.narrow(V, link_axis, 1, spatial_ndim - 1)

    return V_mu @ V_0.adjoint()


# =============================================================================
def holonomy_to_link(
    h: torch.Tensor,
    prefix_dims: int = 1,
    sites_before_link: bool = True,
    constrained: bool = False,
) -> torch.Tensor:
    """
    This function composes the maps

        h → V = holonomy_to_prelink(h, constrained=constrained)
        V → U = prelink_to_link(V)

    See :func:`holonomy_to_prelink` for the meaning of ``constrained``.
    """
    kws = {'prefix_dims': prefix_dims, 'sites_before_link': sites_before_link}
    V = holonomy_to_prelink(h, constrained=constrained, **kws)
    return prelink_to_link(V, **kws)


# =============================================================================
def holonomy_to_prelink(
    h: torch.Tensor,
    prefix_dims: int = 1,
    sites_before_link: bool = True,
    constrained: bool = False,
) -> torch.Tensor:
    r"""
    Reconstruct prelinks from the prelink holonomy h_{mu, 0}(x).

    Parameters
    ----------
    h : torch.Tensor
        Holonomy tensor with shape
        ``(...batch..., n0+1, ..., nd+1, ndim-1, Nc, Nc)``
        if ``sites_before_link=True``, else
        ``(...batch..., ndim-1, n0+1, ..., nd+1, Nc, Nc)``.
    prefix_dims : int, default=1
        Number of leading batch/channel dimensions.
    sites_before_link : bool, default=True
        If True, spatial axes precede the link-direction axis.
    constrained : bool, default=False
        If False (default), use :func:`holonomy_to_prelink_no_constraints`:
        no assumptions on the corner holonomies; P and Q are solved directly.
        If True, use :func:`holonomy_to_prelink_with_constraints`: assumes
        constraints A and B hold.

    Returns
    -------
    torch.Tensor
        Prelinks V in temporal gauge, same shape as h but ndim entries on
        the direction axis (mu = 0, ..., ndim-1).
    """
    kws = {'prefix_dims': prefix_dims, 'sites_before_link': sites_before_link}
    if constrained:
        return holonomy_to_prelink_with_constraints(h, **kws)
    return holonomy_to_prelink_no_constraints(h, **kws)


# =============================================================================
def holonomy_to_prelink_with_constraints(
    h: torch.Tensor,
    prefix_dims: int = 1,
    sites_before_link: bool = True,
) -> torch.Tensor:
    r"""
    Reconstruct prelinks from h_{mu, 0}(x) assuming constraints A and B hold.

    A. **Corner constraint**: ``C_00 C_01† C_11 C_10† = I``.
    B. **Boundary consistency**: ``h_{mu, 0}(t=N_0)† @ h_{mu, 0}(t=0)``
       is independent of mu.

    The output of :func:`prelink_to_holonomy` satisfies both constraints,
    so this function is the exact left-inverse of that map (up to gauge).

    The reconstruction uses a temporal gauge (V_0 = I for t < N_0):

    1. **V_mu for t < N_0**: h_{mu, 0}(x) directly (since V_0 = I there).
    2. **V_mu at t = N_0**: periodic copy, V_mu(N_0) = h_{mu, 0}(0).
    3. **V_0 at t = N_0**: V_0(N_0) = h_{1,0}(N_0)† @ h_{1,0}(0).

    Parameters
    ----------
    h : torch.Tensor
        Holonomy tensor with shape
        ``(...batch..., n0+1, ..., nd+1, ndim-1, Nc, Nc)``
        if ``sites_before_link=True``, else
        ``(...batch..., ndim-1, n0+1, ..., nd+1, Nc, Nc)``.
    prefix_dims : int, default=1
        Number of leading batch/channel dimensions.
    sites_before_link : bool, default=True
        If True, spatial axes precede the link-direction axis.

    Returns
    -------
    torch.Tensor
        Prelinks V in temporal gauge, same shape as h but ndim entries on
        the direction axis (mu = 0, ..., ndim-1).
    """
    # Axis corresponding to the link direction
    link_axis = -3 if sites_before_link else prefix_dims
    time_axis = prefix_dims if sites_before_link else prefix_dims + 1

    N_t_ext = h.shape[time_axis]  # N_0 + 1 (extended time dimension)

    # Step 1: V_0 = I for t < N_0; V_0(N_0) = h(N_0)† @ h(0).
    h_10 = torch.narrow(h, link_axis, 0, 1)
    V_0 = identity_like(h_10)

    h_10_unbind = torch.unbind(h_10, dim=time_axis)
    V_0.select(dim=time_axis, index=-1).copy_(
        h_10_unbind[-1].adjoint() @ h_10_unbind[0]
    )

    # Step 2: V_mu = h for t < N_0; periodic extension at t = N_0.
    V_mu = torch.cat(
        [torch.narrow(h, time_axis, 0, N_t_ext - 1),
         torch.narrow(h, time_axis, 0, 1)],
        dim=time_axis,
    )

    # Step 3: Stack V_0 and V_mu.
    return torch.cat([V_0, V_mu], dim=link_axis)


# =============================================================================
def holonomy_to_prelink_no_constraints(
    h: torch.Tensor,
    prefix_dims: int = 1,
    sites_before_link: bool = True,
) -> torch.Tensor:
    r"""
    Reconstruct prelinks from h_{mu, 0}(x), relaxing constraint A only.

    Constraint A (relaxed):
        ``C_00 C_01† C_11 C_10† = I`` need **not** hold.  P and Q are solved
        from the corner holonomies and incorporated into the reconstruction.

    Constraint B (still assumed):
        ``h_{mu, 0}(t=N_0)† @ h_{mu, 0}(t=0)`` independent of mu.  Step 2
        determines V_0(N_0) using only h_{1,0}; for mu >= 2 the formula
        V_mu = h @ V_0 is consistent only when constraint B holds.

    The reconstruction proceeds in three parts:

    1. **P, Q** from :func:`fix_corner_semiglobal_freedom`.

    2. **V_0**: identity for t < N_0 (temporal gauge); at t = N_0 set from

           V_0(N_0, x) = h_{1,0}(N_0, x)† @ Q @ h_{1,0}(0, x).

       Then P is left-multiplied into V_0 at x = N_1 for all t.

    3. **V_mu (mu >= 1)**: recovered from the definition h = V_mu V_0†:

           V_mu = h @ V_0.

       For mu >= 2 this is correct only when constraint B holds.

    Parameters
    ----------
    h : torch.Tensor
        Holonomy tensor with shape
        ``(...batch..., n0+1, ..., nd+1, ndim-1, Nc, Nc)``
        if ``sites_before_link=True``, else
        ``(...batch..., ndim-1, n0+1, ..., nd+1, Nc, Nc)``.
    prefix_dims : int, default=1
        Number of leading batch/channel dimensions.
    sites_before_link : bool, default=True
        If True, spatial axes precede the link-direction axis.

    Returns
    -------
    torch.Tensor
        Prelinks V in temporal gauge with corner fix applied, same shape as h
        but ndim entries on the direction axis.
    """
    link_axis = -3 if sites_before_link else prefix_dims
    time_axis = prefix_dims if sites_before_link else prefix_dims + 1
    spatial_axis = prefix_dims + 1 if sites_before_link else prefix_dims + 2

    # Part 1: corner semi-global freedoms.
    P, Q = fix_corner_semiglobal_freedom(
        h, prefix_dims=prefix_dims, sites_before_link=sites_before_link,
        keepdim=True,
    )

    # Part 2: Build V_0.
    #   V_0 = I for t < N_0; V_0(N_0) = h(N_0)† @ Q @ h(0).
    h_10 = torch.narrow(h, link_axis, 0, 1)
    V_0 = identity_like(h_10)

    # h_10_unbind drops the time axis, so elements are one rank lower than h.
    # `torch.select` drops the time axis; squeeze Q to match rank.
    h_10_unbind = torch.unbind(h_10, dim=time_axis)
    V_0.select(dim=time_axis, index=-1).copy_(
        h_10_unbind[-1].adjoint() @ Q.squeeze(time_axis) @ h_10_unbind[0]
    )

    # Apply P to V_0 at x = N_1 for all t (including t = N_0).
    # `torch.select` drops the spatial axis; squeeze P to match rank.
    V0_xborder = V_0.select(spatial_axis, -1)
    V0_xborder.copy_(P.squeeze(spatial_axis) @ V0_xborder)

    # Part 3: V_mu = h @ V_0 for all mu >= 1.
    # Follows from h = V_mu @ V_0† with unitary V_0: right-multiply by V_0.
    # V_0 has size 1 on the link axis and broadcasts over all mu in h.
    V_mu = h @ V_0

    return torch.cat([V_0, V_mu], dim=link_axis)


# =============================================================================
def fix_corner_semiglobal_freedom(
    h: torch.Tensor,
    prefix_dims: int = 1,
    sites_before_link: bool = True,
    keepdim: bool = False,
) -> tuple:
    r"""Return the semi-global gauge freedoms at the top and right borders.

    For a 2D extended lattice (one time direction, one spatial direction), the
    four corner holonomies

        C_10 = h(t=N_0, x=0),   C_11 = h(t=N_0, x=N_1)
        C_00 = h(t=0,   x=0),   C_01 = h(t=0,   x=N_1)

    need not satisfy the constraint ``C = C_00 C_01† C_11 C_10† = I``.
    When ``C != I``, the standard formula ``V_0(N_0, x) = h(N_0, x)† h(0, x)``
    produces inconsistent prelinks at the spatial boundary.

    This function factors ``Z = X Y X† Y†`` (group commutator) via
    :func:`solve_sun_commutator` and recovers

        P = C_01† Y C_00   (semi-global freedom for the right column, x = N_1)
        Q = C_10 C_00† X   (semi-global freedom for the bottom row, t = N_0)

    Applying ``P`` to all prelinks at ``x = N_1`` and ``Q`` to all prelinks
    at ``t = N_0`` yields a globally consistent prelink configuration without
    requiring the corner constraint.

    Parameters
    ----------
    h : torch.Tensor
        Holonomy ``h_{1,0}`` for the 2D lattice. Shape:
        ``(...batch..., N_0+1, N_1+1, 1, Nc, Nc)``
        if ``sites_before_link=True``.
    prefix_dims : int, default=1
        Number of leading batch/channel dimensions.
    sites_before_link : bool, default=True
        Whether spatial axes precede the link axis.
    keepdim : bool, default=False
        If ``True``, P and Q retain the same number of dimensions as ``h``,
        with size-1 in the collapsed time, spatial, and link axes.
        If ``False`` (default), those axes are dropped.

    Returns
    -------
    P : torch.Tensor
        Semi-global freedom for the right column, ``x = N_1``, acts on ``V_0``.
        Shape ``(...batch..., Nc, Nc)`` (``keepdim=False``) or the same rank
        as ``h`` with size-1 in the collapsed axes (``keepdim=True``).
    Q : torch.Tensor
        Semi-global freedom for the top row, ``t = N_0``, acts on ``V_1``.
        Same shape convention as ``P``.
    """
    link_axis = -3 if sites_before_link else prefix_dims
    time_axis = prefix_dims         # axis for t
    spatial_axis = prefix_dims + 1  # axis for x (mu=1 direction)
    # Note: time and spatial axes are computed excluding the link axis.

    # Select the single link component (mu = 1) and drop the link axis.
    h_mu = h.select(link_axis, 0)
    # h_mu shape: (...batch..., N_0+1, N_1+1, Nc, Nc)
    # After removing link_axis (which sits after time_axis and spatial_axis),
    # time_axis and spatial_axis indices are unchanged.

    # Extract the four corner holonomies. Each select removes one dimension,
    # so the spatial index shifts down by 1 after the time select.
    h_t0 = h_mu.select(time_axis, 0)
    h_tN = h_mu.select(time_axis, -1)
    C_00 = h_t0.select(spatial_axis - 1, 0)   # t=0,   x=0
    C_01 = h_t0.select(spatial_axis - 1, -1)  # t=0,   x=N_1
    C_10 = h_tN.select(spatial_axis - 1, 0)   # t=N_0, x=0
    C_11 = h_tN.select(spatial_axis - 1, -1)  # t=N_0, x=N_1

    # Compute Z = C_00 C_10† C_11 C_01† and solve Z = X Y X† Y†.
    # solve_sun_commutator(Z) returns (X, Y) with Z = [X, Y]
    Z = C_00 @ C_10.adjoint() @ C_11 @ C_01.adjoint()
    X, Y = solve_sun_commutator(Z, random_twist=True)

    # Recover the border semi-global freedoms from the commutator solution.
    P = C_01.adjoint() @ Y @ C_00  # right column (x = N_1), acts on V_0
    Q = C_10 @ C_00.adjoint() @ X  # top row      (t = N_0), acts on V_1

    if keepdim:
        # Restore the three collapsed axes (time, spatial, link) as size-1 so
        # P and Q have the same rank as h and broadcast directly.
        # Note: time and spatial axes were computed excluding the link axis.
        P = P.unsqueeze(time_axis).unsqueeze(spatial_axis).unsqueeze(link_axis)
        Q = Q.unsqueeze(time_axis).unsqueeze(spatial_axis).unsqueeze(link_axis)

    return P, Q


# =============================================================================
def compute_h_munu(
    h: torch.Tensor,
    mu: int,
    nu: int,
    prefix_dims: int = 1,
    sites_before_link: bool = True,
) -> torch.Tensor:
    """
    Compute the holonomy h_{mu,nu} with spectator boundary sites zeroed.

    For nonvanishing mu and nu: h_{mu,nu}(x) = h_{mu,0}(x) h_{nu,0}(x)†, and
    for nu == 0 this reduces to h_{mu,0} itself (no matrix multiply needed).

    The plaquette sum for the (mu, nu) plane only involves sites x_k in
    [0, N_k-1] for spectator directions k ≠ mu, nu.  The last site (x_k = N_k)
    in each such direction is zeroed out so that downstream routines —
    `_planar_action_sum` and the holonomy clover for the force — automatically
    sum over the physical volume without extra logic, while the tensor shape
    remains uniform.

    Parameters
    ----------
    h : torch.Tensor
        Holonomy stack h_{mu,0} for mu = 1, ..., d-1.
        Shape ``(batch, N_0+1, ..., N_{d-1}+1, d-1, Nc, Nc)``
        when ``sites_before_link=True``.
    mu : int
        First direction index (1 <= mu < d).
    nu : int
        Second direction index (0 <= nu < mu).
    prefix_dims : int, default=1
        Number of leading non-spatial axes (typically 1 for the batch axis).
    sites_before_link : bool, default=True
        Whether spatial lattice axes precede the link axis in ``h``.

    Returns
    -------
    torch.Tensor
        h_{mu,nu}, same shape as a single component of ``h``, with the last
        site in each spectator direction set to zero.
    """
    assert mu != nu, f"mu ({mu}) and nu ({nu}) must be different!"

    link_axis = -3 if sites_before_link else prefix_dims
    h_stack = torch.unbind(h, dim=link_axis)

    if nu == 0:
        h_munu = h_stack[mu - 1].clone()
    elif mu == 0:
        h_munu = h_stack[nu - 1].adjoint().clone()
    else:
        h_munu = h_stack[mu - 1] @ h_stack[nu - 1].adjoint()

    # Zero out the last site in each spectator direction k ≠ mu, nu.
    # dim_mu and dim_nu keep their extended sites for the plaquette shifts.
    spatial_ndim = h.ndim - prefix_dims - 3
    for k in range(spatial_ndim):
        if k in [mu, nu]:
            continue
        dim_k = prefix_dims + k
        # index_fill_ requires a 1-D index; we use narrow+zero_ for any shape
        h_munu.narrow(dim_k, h_munu.shape[dim_k] - 1, 1).zero_()

    return h_munu


# =============================================================================
def identity_like(matrix: torch.Tensor):
    """Identity matrices with same batch shape, dtype and device."""
    n, m = matrix.shape[-2:]
    if n != m:
        raise ValueError("Input must contain square matrices.")

    identity = torch.zeros_like(matrix)
    identity.diagonal(dim1=-2, dim2=-1).fill_(1)
    return identity
