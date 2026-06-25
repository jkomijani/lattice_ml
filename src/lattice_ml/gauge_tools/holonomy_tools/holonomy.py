# Created by Javad Komijani, Jun 2026

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
) -> torch.Tensor:
    """
    This function composes the maps

        h → V = holonomy_to_prelink(h)
        V → U = prelink_to_link(h)

    See those functions for more details.
    """
    kws = {'prefix_dims': prefix_dims, 'sites_before_link': sites_before_link}
    return prelink_to_link(holonomy_to_prelink(h, **kws), **kws)


# =============================================================================
def holonomy_to_prelink(
    h: torch.Tensor,
    prefix_dims: int = 1,
    sites_before_link: bool = True,
) -> torch.Tensor:
    r"""
    Reconstruct prelinks from the prelink holonomy h_{mu, 0}(x).

    This is the inverse of `prelink_to_holonomy`, up to a gauge transformation.
    The reconstruction is carried out in a temporal gauge where

        V_0(x) = I   for t = 0, ..., N_0 - 1,

    using the N_0 independent site-local gauge parameters Omega(x) at the
    physical time slices. The remaining degrees of freedom are then fixed by:

    1. **V_mu (mu != 0) for t = 0, ..., N_0 - 1**:
       In the gauge V_0 = I, the holonomy reduces to

           h_{mu, 0}(x) = V_mu(x)

    2. **V_mu (mu != 0) at t = N_0**:
       Set by the periodic boundary condition:

           V_mu(t = N_0) = V_mu(t = 0) = h_{mu, 0}(t = 0).

    3. **V_0 at t = N_0**:
       Using V_mu(t = N_0) from step 2 and the holonomy at t = N_0:

           h_{mu, 0}(t = N_0) = V_mu(t = N_0) V_0(t = N_0)^\dagger

       we solve for

           V_0(t = N_0) = h_{mu, 0}(N_0)^\dagger @ h_{mu, 0}(0)

       This is independent of non-vanishing mu; mu = 1 is used in the code.

    .. warning::
       The independence of the right-hand side from mu is not a coincidence:
       it reveals that ``h_{mu, 0}(t=N_0)`` for mu >= 2 are **not** independent
       degrees of freedom — they are all determined by ``h_{1, 0}(t=N_0)`` and
       the values at t = 0.  Treating them as independent in an HMC update
       leads to inconsistent configurations in 3D+.

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
        Prelinks V in temporal gauge (V_0 = I for t < N_0), with same shape
        as h but ndim entries on the direction axis (mu = 0, ..., ndim-1).

    Notes
    -----
    While ``prelink_to_holonomy(holonomy_to_prelink(h))`` is the identity on h,
    the reverse composition ``holonomy_to_prelink(prelink_to_holonomy(V))``
    returns V in temporal gauge, not the original V.
    """
    # Axis corresponding to the link direction
    link_axis = -3 if sites_before_link else prefix_dims
    time_axis = prefix_dims if sites_before_link else prefix_dims + 1

    N_t_ext = h.shape[time_axis]  # N_0 + 1 (extended time dimension)

    # Step 1: Build V_0 on the extended lattice
    h_10 = torch.narrow(h, link_axis, 0, 1)
    V_0 = identity_like(h_10)

    h_10_unbind = torch.unbind(h_10, dim=time_axis)
    V_0_last = h_10_unbind[-1].adjoint() @ h_10_unbind[0]

    V_0.select(dim=time_axis, index=-1).copy_(V_0_last)

    # Step 2: Build V_mu (mu != 0) from h with periodic extension
    V_mu = torch.cat(
        [torch.narrow(h, time_axis, 0, N_t_ext - 1),
         torch.narrow(h, time_axis, 0, 1)
         ],
        dim=time_axis
    )

    # Step 3: Stack V_0 and V_mu to form V in temporal gauge
    V = torch.cat([V_0, V_mu], dim=link_axis)

    return V


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
