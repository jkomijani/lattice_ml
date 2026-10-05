# Copyright (c) 2026 Javad Komijani

r"""
U(1) Prelink Holonomy
======================

U(1) counterpart of `holonomy.py`. Since U(1) is abelian, prelinks V_mu(x)
are plain complex phases, and the prelink holonomy

    h_{mu, nu}(x) = V_mu(x) V_nu(x)^\dagger

reduces to h_{mu, nu}(x) = V_mu(x) V_nu(x)^*, ordinary complex multiplication.

Unlike the SU(n) case, only the *constrained* reconstruction is implemented
here (the U(1) analogue of `holonomy_to_prelink_with_constraints`).

Conventions
-----------
- Spatial axes may precede (`sites_before_link=True`) or follow the link axis.
- The first `prefix_dims` axes are batch/channel dimensions.
- All tensors are plain complex phases (no Nc x Nc matrix components).
- V and h live on the extended lattice of size (N_mu + 1) along each direction.
"""

# pylint: disable=invalid-name

import torch

from ..prelink_tools.prelinks_u1 import prelink_to_link_u1


__all__ = [
    "holonomy_to_link_u1",
    "holonomy_to_prelink_u1",
]


# =============================================================================
def holonomy_to_link_u1(
    h: torch.Tensor,
    prefix_dims: int = 1,
    sites_before_link: bool = True,
) -> torch.Tensor:
    """
    This function composes the maps

        h → V = holonomy_to_prelink_u1(h)
        V → U = prelink_to_link_u1(V)

    U(1) counterpart of `holonomy_to_link`. There is no `constrained` option:
    for abelian U(1) the corner constraint always holds, so the constrained
    reconstruction is the only one that exists.
    """
    kws = {'prefix_dims': prefix_dims, 'sites_before_link': sites_before_link}
    V = holonomy_to_prelink_u1(h, **kws)
    return prelink_to_link_u1(V, **kws)


# =============================================================================
def holonomy_to_prelink_u1(
    h: torch.Tensor,
    prefix_dims: int = 1,
    sites_before_link: bool = True,
) -> torch.Tensor:
    r"""
    Reconstruct U(1) prelinks from the prelink holonomy h_{mu, 0}(x).

    U(1) counterpart of `holonomy_to_prelink_with_constraints`. Since U(1)
    is abelian, the corner constraint ``C_00 C_01^* C_11 C_10^* = 1`` holds
    automatically, so this reconstruction is exact and no `_no_constraints`
    fallback is needed.

    The reconstruction uses a temporal gauge (V_0 = 1 for t < N_0):

    1. **V_mu for t < N_0**: h_{mu, 0}(x) directly (since V_0 = 1 there).
    2. **V_mu at t = N_0**: periodic copy, V_mu(N_0) = h_{mu, 0}(0).
    3. **V_0 at t = N_0**: V_0(N_0) = h_{1,0}(N_0)^* * h_{1,0}(0).

    Parameters
    ----------
    h : torch.Tensor
        Holonomy tensor with shape
        ``(...batch..., n0+1, ..., nd+1, ndim-1)``
        if ``sites_before_link=True``, else
        ``(...batch..., ndim-1, n0+1, ..., nd+1)``.
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
    link_axis = -1 if sites_before_link else prefix_dims
    time_axis = prefix_dims if sites_before_link else prefix_dims + 1

    N_t_ext = h.shape[time_axis]  # N_0 + 1 (extended time dimension)

    # Step 1: V_0 = 1 for t < N_0; V_0(N_0) = h(N_0)^* * h(0).
    h_10 = torch.narrow(h, link_axis, 0, 1)
    V_0 = torch.ones_like(h_10)

    h_10_unbind = torch.unbind(h_10, dim=time_axis)
    V_0.select(dim=time_axis, index=-1).copy_(
        h_10_unbind[-1].conj() * h_10_unbind[0]
    )

    # Step 2: V_mu = h for t < N_0; periodic extension at t = N_0.
    V_mu = torch.cat(
        [torch.narrow(h, time_axis, 0, N_t_ext - 1),
         torch.narrow(h, time_axis, 0, 1)],
        dim=time_axis,
    )

    # Step 3: Stack V_0 and V_mu.
    return torch.cat([V_0, V_mu], dim=link_axis)
