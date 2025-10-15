# Copyright (c) 2025 Javad Komijani

"""
Staples calculations for Lattice Gauge Theory.

This module provides functions to compute planar staples and sums of staples
for links in specified directions, as used in Wilson gauge action computations.
"""

from typing import Tuple, List
import torch


__all__ = [
    'compute_all_directional_staples',
    'compute_directional_staples',
    'compute_planar_staples'
]

matmul = torch.matmul


def compute_all_directional_staples(
    x: torch.Tensor,
    prefix_dims: int = 1,
    sites_before_link: bool = True,
):
    """
    Compute the staple sums for all link directions in the lattice.

    For each link direction 'mu', this function sums the staples in all planes
    spanned by ('mu', 'nu') for every perpendicular direction 'nu'.
    The results are stacked along the link-axis dimension, producing a tensor
    with the same shape as `x`.

    Parameters
    ----------
    x : torch.Tensor
        Tensor containing the gauge links. After any batch and channel axes,
        the spatial lattice axes come first (if sites_before_link=True),
        followed by the link direction axis, and then the matrix components.
    prefix_dims : int, default=1
        Number of leading batch and channel dimensions in the tensor.
        For example, if x.shape = (batch, channel, Lx, Ly, Lz, Lt, mu, Nc, Nc),
        then prefix_dims=2. If only a single batch dimension, prefix_dims=1.
    sites_before_link : bool, default=True
        Whether the spatial lattice axes come before the link axis.

    Returns
    -------
    torch.Tensor
         Tensor of staple sums for all link directions. Its shape is identical
         to `x`, with the link-direction axis containing the summed staples
         for each direction `mu`.

    Notes
    -----
    The staples are defined such that the expression `x @ g` is gauge
    covariant, where `g` is the output of this function. Using this definition,
    the Wilson gauge action can be expressed is proportional to

        ReTr(x @ g) + ...

    where the sum over lattice sites and directions is implied.
    """
    # Prepare keyword arguments to pass to compute_directional_staples
    kws = {'prefix_dims': prefix_dims, 'sites_before_link': sites_before_link}

    # Determine the number of spatial dimensions
    ndim = len(x.shape) - 4  # exclude batch & direction & matrix indices

    # Initialize a list to store staples sums for each direction 'mu'
    staples_stack: List[torch.Tensor] = [None] * ndim

    # Loop over each link direction 'mu'
    for mu in range(ndim):
        # Loop over each link direction 'mu'
        nu_list = [nu for nu in range(ndim) if nu != mu]

        # Compute the staples sum for this direction
        staples_stack[mu] = compute_directional_staples(x, mu, nu_list, **kws)

    # Stack the results along the link-axis to recreate the full tensor
    link_axis = -3 if sites_before_link else prefix_dims
    gamma_matrix = torch.stack(staples_stack, dim=link_axis)

    return gamma_matrix


def compute_directional_staples(
    x: torch.Tensor,
    mu: int,
    nu_list: Tuple[int, ...],
    prefix_dims: int = 1,
    sites_before_link: bool = True,
):
    """
    Compute the sum of staples in the mu-nu planes for a given link direction
    `mu` as used in the Wilson gauge action.

    Staples are 3-link paths forming a 'staple' shape adjacent to a central
    link, used in the Wilson gauge action in lattice gauge theory. The upper
    and lower staples on the mu-nu plane can be visualized as:

        >>>     --b--
        >>>    c|   |a
        >>>     @ U @    +   @ U @
        >>>                 f|   |d
        >>>                  --e--

    where `@ U @` represents the central link whose staples are being computed.

    Parameters
    ----------
    x : torch.Tensor
        Tensor containing the gauge links. After any batch and channel axes,
        the spatial lattice axes come first (if sites_before_link=True),
        followed by the link direction axis, and then the matrix components.
    mu : int
        The index of the link direction along which the staples are computed.
    nu_list : list of int
        List of perpendicular directions `nu` over which to sum the staples.
    prefix_dims : int, default=1
        Number of leading batch and channel dimensions in the tensor.
        For example, if x.shape = (batch, channel, Lx, Ly, Lz, Lt, mu, Nc, Nc),
        then prefix_dims=2. If only a single batch dimension, prefix_dims=1.
    sites_before_link : bool, default=True
        Whether the spatial lattice axes come before the link axis.

    Returns
    -------
    torch.Tensor
        Tensor containing the sum of staples for links in direction `mu`.
        Shape is the same as `x`, except the link-direction axis is removed.
    """
    kws = {'prefix_dims': prefix_dims, 'sites_before_link': sites_before_link}
    return sum(compute_planar_staples(x, mu, nu, **kws) for nu in nu_list)


def compute_planar_staples(
    x: torch.Tensor,
    mu: int,
    nu: int,
    prefix_dims: int = 1,
    sites_before_link: bool = True,
    return_sum: bool = True
):
    r"""
    Compute the staples in the mu-nu plane for a given link direction `mu` as
    used in the Wilson gauge action.

    Staples are 3-link paths forming a 'staple' shape adjacent to a central
    link, used in the Wilson gauge action in lattice gauge theory. The upper
    and lower staples on the mu-nu plane can be visualized as:

        >>>     --b--
        >>>    c|   |a
        >>>     @ U @    +   @ U @
        >>>                 f|   |d
        >>>                  --e--

    where `@ U @` represents the central link whose staples are being computed.

    Parameters
    ----------
    x : torch.Tensor
        Tensor containing the gauge links. After any batch and channel axes,
        the spatial lattice axes come first (if sites_before_link=True),
        followed by the link direction axis, and then the matrix components.
    mu : int
        The index of the link direction along which the staples are computed.
    nu : int
        Orthogonal direction in the mu-nu plane.
    prefix_dims : int, default=1
        Number of leading batch and channel dimensions in the tensor.
        For example, if x.shape = (batch, channel, Lx, Ly, Lz, Lt, mu, Nc, Nc),
        then prefix_dims=2. If only a single batch dimension, prefix_dims=1.
    sites_before_link : bool, default=True
        Whether the spatial lattice axes come before the link axis.
    return_sum : bool
        If False, return both the upper and lower staples as a tuple.
        If True, return their sum. Default is True.

    Returns
    -------
    torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
        If `return_sum` is True (default), returns the sum of the upper and
        lower staples. If `return_sum` is False, returns a tuple containing
        the upper and lower staples separately.

        In either case, each staple tensor has the same shape as `x`, except
        that the link-direction axis is removed.
    """

    # Extract the central and nu-direction links using unbind
    link_axis = -3 if sites_before_link else prefix_dims
    links = torch.unbind(x, dim=link_axis)

    u = links[mu]  # Central link (U in the cartoon)
    c = links[nu]  # Link in nu direction from the same site

    # Calculate links needed to form staples
    a = torch.roll(c, -1, dims=prefix_dims + mu)
    b = torch.roll(u, -1, dims=prefix_dims + nu)
    e = torch.roll(u, +1, dims=prefix_dims + nu)
    f = torch.roll(c, +1, dims=prefix_dims + nu)
    d = torch.roll(f, -1, dims=prefix_dims + mu)

    # Upper staple: a b^\dagger c^\dagger
    #   --b--
    #  c|   |a
    #   @ U @
    staple_upper = matmul(a, matmul(c, b).adjoint())

    # Lower staple: d^\dagger e^\dagger f
    #   @ U @
    #  f|   |d
    #   --e--
    staple_lower = matmul(matmul(e, d).adjoint(), f)

    # Return the staples
    if return_sum:
        return staple_upper + staple_lower
    return staple_upper, staple_lower
