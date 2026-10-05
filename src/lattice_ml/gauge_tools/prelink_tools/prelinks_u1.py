# Copyright (c) 2026 Javad Komijani

r"""
U(1) Prelink (Prepotential) Operations for Lattice Gauge Theory
=================================================================

U(1) counterpart of `prelinks.py`. Since U(1) is abelian, prelinks V_mu(x)
are plain complex phases (no Nc x Nc matrix components): the group-valued
forward difference becomes ordinary (commutative) complex multiplication,
and adjoints become complex conjugation.

Only `prelink_to_link_u1` is provided here (the reverse direction), since
that is all the U(1) holonomy pipeline needs; the forward direction
(`link_to_prelink`) is not implemented.

Conventions
-----------
- Spatial axes may precede (`sites_before_link=True`) or follow the link axis.
- The first `prefix_dims` axes are batch/channel dimensions.
- All tensors are plain complex phases (no matrix components).
"""

# pylint: disable=invalid-name

from typing import List
import torch


__all__ = ["prelink_to_link_u1"]


# =============================================================================
def prelink_to_link_u1(
    V: torch.Tensor,
    prefix_dims: int = 1,
    sites_before_link: bool = True,
):
    r"""
    Convert U(1) link prepotentials (prelinks) to gauge links.

    U(1) counterpart of `prelink_to_link`. The links are computed as a
    group-valued forward difference:

        U_mu(x) = V_mu(x)^* V_mu(x + mu_hat)

    which, since U(1) is abelian, is ordinary complex multiplication.

    The prelinks V_mu(x) live on an extended lattice of size (N_mu + 1) along
    each spatial direction. The resulting links live on a lattice of size N_mu
    along every direction.

    Parameters
    ----------
    V : torch.Tensor
        Prelink tensor with shape:
        (prefix_dims..., spatial_dims..., mu) if sites_before_link=True
        or (prefix_dims..., mu, spatial_dims...) otherwise.

    prefix_dims : int, default=1
        Number of leading batch/channel dimensions in the tensor.

    sites_before_link : bool, default=True
        If True, the spatial lattice axes precede the link direction axis.

    Returns
    -------
    torch.Tensor
        Tensor of physical U(1) gauge links with reduced lattice sizes.
    """
    # Axis corresponding to link direction mu
    link_axis = -1 if sites_before_link else prefix_dims

    # Number of spatial lattice dimensions (exclude prefix & link axes)
    spatial_ndim = V.ndim - prefix_dims - 1

    # Separate prelinks by direction mu
    prelinks_stack = torch.unbind(V, dim=link_axis)

    # Allocate container for resulting links
    links_stack: List[torch.Tensor] = [None] * spatial_ndim

    for mu, prelink_mu in enumerate(prelinks_stack):

        dim_mu = prefix_dims + mu  # axis corresponding to direction mu
        len_mu = prelink_mu.shape[dim_mu]

        # Group-valued forward difference along direction mu:
        left = torch.narrow(prelink_mu, dim_mu, 0, len_mu - 1)
        right = torch.narrow(prelink_mu, dim_mu, 1, len_mu - 1)

        link_mu = left.conj() * right

        # Clamp all other spatial dimensions to size N_nu
        for nu in range(spatial_ndim):
            if nu == mu:
                continue
            dim_nu = prefix_dims + nu  # axis corresponding to direction nu
            len_nu = prelink_mu.shape[dim_nu]
            link_mu = torch.narrow(link_mu, dim_nu, 0, len_nu - 1)
        links_stack[mu] = link_mu

    # Restore original layout with link-direction axis
    return torch.stack(links_stack, dim=link_axis)
