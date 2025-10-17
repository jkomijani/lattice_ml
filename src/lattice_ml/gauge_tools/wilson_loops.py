# Copyright (c) 2025 Javad Komijani

"""
Wilson loop and path calculations for Lattice Gauge Theory.

This module provides functions to compute standard 1×1 Wilson loops and their
linear responses to Lie-algebra-valued inputs. It also includes utilities for
parallel transport and the averaged trace of rectangular Wilson loops of size
m×n across all lattice planes.
"""

from typing import Tuple
import torch


__all__ = [
    'compute_avg_trace_wilson_mxn_loop',
    'compute_wilson_1x1_loop',
    'compute_wilson_1x1_loop_response',
    'parallel_transport'
]

matmul = torch.matmul


def compute_avg_trace_wilson_mxn_loop(
    x: torch.Tensor,
    n: int,
    m: int,
    prefix_dims: int = 1,
    sites_before_link: bool = True
):
    """
    Compute rectangular Wilson loops of size m x n in all lattice planes and
    return the mean of their reduced traces (real part).

    Parameters
    ----------
    x : torch.Tensor
        Tensor containing the gauge links. After any batch and channel axes,
        the spatial lattice axes come first (if sites_before_link=True),
        followed by the link direction axis, and then the matrix components.
    m, n : int
        Length and width of the rectangle.
    prefix_dims : int, default=1
        Number of leading batch and channel dimensions in the tensor.
        For example, if x.shape = (batch, channel, Lx, Ly, Lz, Lt, mu, Nc, Nc),
        then prefix_dims=2. If only a single batch dimension, prefix_dims=1.
    sites_before_link : bool, default=True
        Whether the spatial lattice axes come before the link axis.

    Returns
    -------
    torch.Tensor
        Mean of the real part of the reduced trace over all planes.
        Shape = x.shape[:prefix_dims].
    """
    # Determine the number of spatial dimensions
    spatial_ndim = x.ndim - prefix_dims - 3  # exclude batch, direction, matrix
    sum_dims = tuple(range(prefix_dims, prefix_dims + spatial_ndim))

    mean = torch.zeros(
        x.shape[:prefix_dims], device=x.device, dtype=x.real.dtype
    )

    for mu in range(spatial_ndim):
        for nu in range(spatial_ndim):
            if mu == nu:
                continue  # collapsed rectangle
            if m == n and mu < nu:
                continue  # avoid double counting

            if m == 1 and n == 1:
                w_mxn = compute_wilson_1x1_loop(
                    x, mu, nu, prefix_dims, sites_before_link
                )
            else:
                # Use 1-based indexing for parallel_transport
                directions = [mu+1]*m + [nu+1]*n + [-(mu+1)]*m + [-(nu+1)]*n
                w_mxn = parallel_transport(
                    x, directions, prefix_dims, sites_before_link,
                )

            mean += torch.mean(compute_reduced_trace(w_mxn).real, dim=sum_dims)

    # Normalize by number of planes
    num_planes = spatial_ndim * (spatial_ndim - 1)
    mean /= num_planes
    if m == n:
        mean *= 2  # square loops counted only once in previous loop

    return mean


def compute_wilson_1x1_loop(
    x: torch.Tensor,
    mu: int,
    nu: int,
    prefix_dims: int = 1,
    sites_before_link: bool = True
):
    """
    Compute 1×1 Wilson loops in the specified mu-nu plane for all sites.

    The loop goes from the sites along the mu direction, then nu direction,
    and returns along the opposite mu and nu directions to close the plaquette.

    Parameters
    ----------
    x : torch.Tensor
        Tensor containing the gauge links. After any batch and channel axes,
        the spatial lattice axes come first (if sites_before_link=True),
        followed by the link direction axis, and then the matrix components.
    mu : int
        The index of the first link direction.
    nu : int
        The index of the second link direction.
    prefix_dims : int, default=1
        Number of leading batch and channel dimensions in the tensor.
        For example, if x.shape = (batch, channel, Lx, Ly, Lz, Lt, mu, Nc, Nc),
        then prefix_dims=2. If only a single batch dimension, prefix_dims=1.
    sites_before_link : bool, default=True
        Whether the spatial lattice axes come before the link axis.

    Returns
    -------
    torch.Tensor
        Tensor of 1×1 Wilson loops in the mu–nu plane.
        Shape is the same as `x`, except the link-direction axis is removed.
    """
    # Extract the links using unbind
    link_axis = -3 if sites_before_link else prefix_dims
    links = torch.unbind(x, dim=link_axis)

    x_mu = links[mu]
    x_nu = links[nu]

    y_nu = torch.roll(x_nu, -1, dims=prefix_dims + mu)
    z_mu = torch.roll(x_mu, -1, dims=prefix_dims + nu)

    w_11 = matmul(matmul(x_mu, y_nu), matmul(x_nu, z_mu).adjoint())

    return w_11


def compute_wilson_1x1_loop_response(
    x: torch.Tensor,
    w: torch.Tensor,
    mu: int,
    nu: int,
    prefix_dims: int = 1,
    sites_before_link: bool = True
):
    """
    Compute the response of a 1×1 Wilson loop to a Lie-algebra input `w`.

    The tensor `w` specifies Lie-algebra directions (tangent to the gauge
    group) along which the derivative of the Wilson loop in the (mu, nu)
    plane is evaluated. The output is matrix-valued, representing the linear
    effect of `w` on the Wilson loop.

    Parameters
    ----------
    x : torch.Tensor
        Tensor containing the gauge links. After any batch and channel axes,
        the spatial lattice axes come first (if sites_before_link=True),
        followed by the link direction axis, and then the matrix components.
    w : torch.Tensor
        Lie-algebra-valued tensor specifying the derivative directions of the
        Wilson loop. Each element lies in the tangent space of the gauge group
        (e.g., su(N) for SU(N)).
    mu : int
        Index of the first link direction.
    nu : int
        Index of the second link direction.
    prefix_dims : int, default=1
        Number of leading batch and channel dimensions in the tensor.
        For example, if x.shape = (batch, channel, Lx, Ly, Lz, Lt, mu, Nc, Nc),
        then prefix_dims=2. If only a single batch dimension, prefix_dims=1.
    sites_before_link : bool, default=True
        Whether the spatial lattice axes come before the link axis.

    Returns
    -------
    torch.Tensor
        Tensor of the response of 1×1 Wilson loops in the mu-nu plane.
        Shape is the same as `x`, except the link-direction axis is removed.
    """

    # Determine which axis corresponds to the link directions
    link_axis = -3 if sites_before_link else prefix_dims

    # Separate link matrices and their corresponding weight/insertions
    links = torch.unbind(x, dim=link_axis)
    weights = torch.unbind(w, dim=link_axis)

    # Extract links for the mu and nu directions
    x_mu = links[mu]
    x_nu = links[nu]

    # Shift links to form the closed plaquette
    y_nu = torch.roll(x_nu, -1, dims=prefix_dims + mu)
    z_mu = torch.roll(x_mu, -1, dims=prefix_dims + nu)

    # Extract and roll the corresponding Lie-algebra insertions
    w_x_mu = weights[mu]
    w_x_nu = weights[nu]
    w_y_nu = torch.roll(w_x_nu, -1, dims=prefix_dims + mu)
    w_z_mu = torch.roll(w_x_mu, -1, dims=prefix_dims + nu)

    # Compute four weighted contributions, inserting w at each link:
    # insertion acts as δU = w @ U at that link position
    part1 = matmul(matmul(w_x_mu @ x_mu, y_nu), matmul(x_nu, z_mu).adjoint())
    part2 = matmul(matmul(x_mu, w_y_nu @ y_nu), matmul(x_nu, z_mu).adjoint())
    part3 = matmul(matmul(x_mu, y_nu), matmul(w_x_nu @ x_nu, z_mu).adjoint())
    part4 = matmul(matmul(x_mu, y_nu), matmul(x_nu, w_z_mu @ z_mu).adjoint())

    # Sum the four Lie-algebra-weighted response contributions
    return part1 + part2 + part3 + part4


def parallel_transport(
    x: torch.Tensor,
    directions: Tuple[int],
    prefix_dims: int = 1,
    sites_before_link: bool = True,
    start_tensor: torch.Tensor | None = None,
    restore_origin: bool = False
):
    r"""
    Compute the parallel transporter along a specified path in the lattice.

    The path is given by a sequence of signed integers in ``directions``.
    Indexing of directions starts from **1** rather than 0:

    - `+d` (with `1 ≤ d ≤ ndim`) means moving **forward** along the `d`-th
        lattice direction using the link :math:`U_d(x)`.
    - `-d` means moving **backward** along the `d`-th direction, i.e. applying
        the adjoint link :math:`U_d^\dagger(x - \hat e_d)`.

    Example
    -------
    A standard 1×1 plaquette in the (1,2) plane:
        >>> directions = (1, 2, -1, -2)
        >>> # corresponds to (0,1) plane in 0-based indexing

    Parameters
    ----------
    x : torch.Tensor
        Tensor containing the gauge links. After any batch and channel axes,
        the spatial lattice axes come first (if sites_before_link=True),
        followed by the link direction axis, and then the matrix components.
    directions : Tuple[int]
        Ordered list of signed **1-based** directions describing the path.
        Positive = forward step, Negative = backward step.
    prefix_dims : int, default=1
        Number of leading batch and channel dimensions in the tensor.
        For example, if x.shape = (batch, channel, Lx, Ly, Lz, Lt, mu, Nc, Nc),
        then prefix_dims=2. If only a single batch dimension, prefix_dims=1.
    sites_before_link : bool, default=True
        Whether the spatial lattice axes come before the link axis.
    start_tensor : torch.Tensor, default=None
        Tensor to be transported along the path. If None, the path starts
        with the first link in `directions`, which is equivalent to starting
        from the identity matrix.
    restore_origin : bool, default=False
        If True, shifts the final transporter back to the original site,
        so the output is available at the starting point.

    Returns
    -------
    torch.Tensor
        Parallel transporter along the specified path.
        Shape is the same as `x`, except the link-direction axis is removed.
    """
    # Forward vs backward steps are treated differently:
    #
    # Forward step (d > 0):
    #     Multiply the transporter by U_d(x), then shift it forward to align
    #     with the destination site.
    #
    # Backward step (d < 0):
    #     First shift the transporter backward along |d|, then multiply by
    #     U_d^\dagger(x - e_d).
    #
    # Initialization is equivalent to starting from the identity matrix.
    # Shifting the identity yields the same result, so it can be skipped.

    if len(directions) == 0:
        return None

    # Determine link axis
    link_axis = -3 if sites_before_link else prefix_dims
    links = torch.unbind(x, dim=link_axis)

    # Determine the number of spatial dimensions
    spatial_ndim = x.ndim - prefix_dims - 3  # exclude batch, direction, matrix
    displacements = [0] * spatial_ndim

    # Initialize transporter
    if start_tensor is None:
        d0 = directions[0]
        if d0 > 0:
            mu = d0 - 1
            transporter = links[mu]
            transporter = torch.roll(transporter, +1, dims=prefix_dims + mu)
            displacements[mu] += 1
        elif d0 < 0:
            mu = -d0 - 1
            # Shifting the identity yields the same result; it can be skipped
            transporter = links[mu].adjoint()
            displacements[mu] -= 1
        else:
            raise ValueError("direction cannot be 0")
        directions = directions[1:]
    else:
        transporter = start_tensor

    # Apply subsequent steps
    for d in directions:
        if d > 0:
            mu = d - 1
            transporter = transporter @ links[mu]
            transporter = torch.roll(transporter, +1, dims=prefix_dims + mu)
            displacements[mu] += 1
        elif d < 0:
            mu = -d - 1
            transporter = torch.roll(transporter, -1, dims=prefix_dims + mu)
            transporter = transporter @ links[mu].adjoint()
            displacements[mu] -= 1
        else:
            raise ValueError("direction cannot be 0")

    # Roll back to origin if requested and net displacement is nonzero
    if restore_origin and any(d != 0 for d in displacements):
        shifts = tuple(-d for d in displacements)
        dims = tuple(prefix_dims + i for i in range(spatial_ndim))
        transporter = torch.roll(transporter, shifts, dims=dims)

    return transporter


def compute_reduced_trace(x):  # reduced trace = 1/n trace()
    """Compute the reduced trace of the input matrix x."""
    return torch.mean(torch.diagonal(x, dim1=-2, dim2=-1), dim=-1)
