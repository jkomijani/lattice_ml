# Copyright (c) 2026 Javad Komijani

"""
Gauge-covariant prelink routines.

This module provides tools to construct and manipulate gauge-covariant
objects from prelinks (prepotentials). Key functionality includes:

- `compute_sealed_staples`: Compute *sealed staples* from prelinks, where
  staples are sandwiched (contracted) with neighboring prelinks to ensure
  gauge invariance under local transformations. In the current convention,
  the semi-global prelink symmetry is reduced to a global one.

- `compute_sealed_prelinks`: Map sealed staples to prelink support. These can
  be projected to the Lie algebra to obtain the HMC force.

Conventions:
- Supports arbitrary batch/channel dimensions (`prefix_dims`).
- Spatial axes may precede (`sites_before_link=True`) or follow the link axis.
- All tensors are assumed to have matrix components of size Nc x Nc.
- Prelinks are assumed to be integrated from gauge links with a fixed origin.
"""

# pylint: disable=invalid-name

import torch

from .prelinks import (
    prelink_to_link,
    prelink_to_left_right_pair,
    pad_to_max_shape
)
from ..wilson_staples import compute_staples


__all__ = [
    "compute_sealed_staples",
    "compute_sealed_prelinks"
]


# =============================================================================
def compute_sealed_staples(
    V: torch.Tensor,
    prefix_dims: int = 1,
    sites_before_link: bool = True,
    sum_over_staples: bool = True
):
    r"""Compute sealed staples form gauge prelinks for all directions.

    This function evaluates the staple contributions entering the Wilson gauge
    action and *seals* them by contracting with the corresponding neighboring
    prelinks. For each direction mu,

        sealed_mu(x) = V_mu(x + mu_hat) @ staple_mu(x) @ V_mu(x)^\dagger.

    By *sealing* we mean that the Wilson staples are sandwiched (contracted
    via matrix multiplication) by prelinks such that the resulting object is
    invariant under local gauge transformations. In the present construction,
    where the semi-global symmetry of the prelink formulation is reduced to
    a single global transformation, the sealed staples transform covariantly
    only under that remaining global symmetry.

    For each link direction 'mu', this function computes the sealed-staples in
    all planes spanned by ('mu', 'nu') for every perpendicular direction 'nu'.
    The results are stacked along the link-axis dimension, producing a tensor
    with the same shape as `V`.

    Parameters
    ----------
    V : torch.Tensor
        Prelink tensor with shape:
        (prefix_dims..., spatial_dims..., mu, Nc, Nc) if sites_before_link=True
        or (prefix_dims..., mu, spatial_dims..., Nc, Nc) otherwise.

    prefix_dims : int, default=1
        Number of leading batch/channel dimensions in the tensor.

    sites_before_link : bool, default=True
        If True, the spatial lattice axes precede the link direction axis.

    sum_over_staples : bool, default=True
        If True, sums the upper and lower planar staples over all transverse
        directions direction before sealing.

        If False, returns the individual planar sealed staples (including upper
        and lower contributions) stacked along a new axis inserted.

    Returns
    -------
    torch.Tensor
        If `sum_over_staples=True`:
            A tensor of sealed staples with the same shape as corresponding
            links `U`.

        If `sum_over_staples=False`:
            A tensor containing all planar sealed staples stacked along an
            additional axis at position `prefix_dims`.

    Notes
    -----
    Internally, this function:

    1. Reconstructs gauge links U from the prelinks V.
    2. Computes standard Wilson staples from U.
    3. Forms aligned left/right prelink pairs
           V_mu(x), V_mu(x + mu_hat).
    4. Contracts them as `right @ staples @ left`.

    The sealed staples are the natural gauge-invariant building blocks
    entering the Wilson gauge action in the prelink representation.
    """
    U = prelink_to_link(V)
    staples = compute_staples(
        U,
        prefix_dims=prefix_dims,
        sites_before_link=sites_before_link,
        sum_over_staples=sum_over_staples
    )

    if not sum_over_staples:
        # All staples are stacked along a new axis inserted at `prefix_dims`.
        V = V.unsqueeze(prefix_dims)
        prefix_dims = prefix_dims + 1

    left, right = prelink_to_left_right_pair(
        V,
        prefix_dims=prefix_dims,
        sites_before_link=sites_before_link
    )

    sealed_staples = right @ staples @ left.adjoint()

    return sealed_staples


# =============================================================================
def compute_sealed_prelinks(
    V: torch.Tensor,
    prefix_dims: int = 1,
    sites_before_link: bool = True,
    sum_over_staples: bool = True
):
    """Compute *sealed prelinks* from gauge prelinks.

    This function maps the *sealed staples* to prelink support by applying
    a discrete forward difference along each lattice direction. The output
    inherit the gauge-covariant structure of the sealed staples and respect
    the reduced global gauge symmetry of the prelink construction.

    The forward difference is computed with zero boundary conditions, extending
    the tensor along each direction by one element, and it is padded in the
    transverse directions to the same shape of prelinks V.
    The resulting object corresponds to the raw matrix-valued term appearing in
    the derivative of the Wilson gauge action with respect to the prelinks V.
    The output is NOT yet projected onto the Lie algebra; for HMC applications,
    a projection to the algebra is needed to obtain the algebra-valued force.

    Parameters
    ----------
    V : torch.Tensor
        Prelink tensor with shape:
        (prefix_dims..., spatial_dims..., mu, Nc, Nc) if sites_before_link=True
        or (prefix_dims..., mu, spatial_dims..., Nc, Nc) otherwise.

    prefix_dims : int, default=1
        Number of leading batch/channel dimensions in the tensor.

    sites_before_link : bool, default=True
        If True, the spatial lattice axes precede the link direction axis.

    sum_over_staples : bool, default=True
        If True, sums the planar sealed staples over all perpendicular
        directions before mapping to prelinks. If False, keeps all individual
        planar sealed staples along a new axis.

    Returns
    -------
    torch.Tensor
        If `sum_over_staples=True`:
            Tensor of sealed prelinks with the same layout as `V`.

        If `sum_over_staples=False`:
            Tensor of components of sealed prelinks, each with of the same
            layout as `V`, and the components are stacked along an additional
            axis at position `prefix_dims`.
    """
    # Compute sealed staples, sandwiched by prelinks to ensure gauge covariance
    sealed_staples = compute_sealed_staples(
        V,
        prefix_dims=prefix_dims,
        sites_before_link=sites_before_link,
        sum_over_staples=sum_over_staples
    )
    # If not summing over staples, increment prefix_dims for stacked axis
    if not sum_over_staples:
        prefix_dims = prefix_dims + 1

    # Axis corresponding to the link direction mu
    link_axis = -3 if sites_before_link else prefix_dims

    # Number of spatial lattice dimensions (exclude prefix, link, matrix)
    spatial_ndim = sealed_staples.ndim - prefix_dims - 3

    # Separate sealed staples by direction mu
    sealed_staples_stack = torch.unbind(sealed_staples, dim=link_axis)

    # Allocate container for resulting *sealed* prelinks
    sealed_prelinks_stack: list[torch.Tensor] = [None] * spatial_ndim

    # Apply forward difference with zero boundary along each direction
    for mu, staple_mu in enumerate(sealed_staples_stack):
        dim_mu = prefix_dims + mu  # axis along which to difference
        diff = forward_difference_zero_boundary(staple_mu, dim=dim_mu)
        sealed_prelinks_stack[mu] = diff

    # Stack along the link axis; pad to maximum spatial shape if necessary
    out = torch.stack(pad_to_max_shape(sealed_prelinks_stack), dim=link_axis)
    return out


# =============================================================================
def forward_difference_zero_boundary(x: torch.Tensor, dim: int):
    """
    Forward finite difference with homogeneous (zero) boundary conditions.

    Maps a tensor of size N along `dim` to size N+1 via

        (x_0, ..., x_{N-1})
        → (x_0, x_1 - x_0, ..., x_{N-1} - x_{N-2}, -x_{N-1}).

    Implemented as `torch.diff` with zero prepend and append.
    """
    zero = torch.zeros_like(x.narrow(dim, 0, 1))
    return torch.diff(x, dim=dim, prepend=zero, append=zero)
