# Created by Javad Komijani, Feb 2026

r"""
Prelink (Prepotential) Operations for Lattice Gauge Theory
==========================================================

Prelinks
--------

We introduce a set of link prepotentials (or "prelinks") V_mu(x) in group G,
defined on an extended lattice of size (N_mu + 1) along each spatial direction.
The physical gauge links U_mu(x) in group G are obtained via the group-valued
forward difference:

    U_mu(x) = V_mu(x)^\dagger V_mu(x + mu_hat)

Geometrically, U_mu(x) represents the relative group displacement between
neighboring sites along direction mu. Since prelinks live on the extended
lattice, the resulting links live on a lattice of length N_mu along each
direction.

The link lattice can be treated as periodic, whereas the prelink lattice
is not periodic.

Symmetries
----------

The prelink formulation enlarges the configuration space and introduces
local and semi-global redundancies.

Semi-global symmetry
~~~~~~~~~~~~~~~~~~~

The links are invariant under the semi-global transformation:

    V_mu(x) -> Q(x, mu) V_mu(x),   Q(x, mu) in G

where Q(x, mu) must be the same for all sites in the equivalence class

    [x, mu] = { x + n * mu_hat | n = 0, ..., N_mu }

Under this transformation, the physical links remain unchanged:

    U_mu(x) -> U_mu(x)

This symmetry is called semi-global because Q acts along the entire mu-
direction class simultaneously, rather than independently at each site.

Ordinary gauge symmetry
~~~~~~~~~~~~~~~~~~~~~~~

The standard site-local gauge transformation acts as:

    V_mu(x) -> V_mu(x) Omega(x)^\dagger,   Omega(x) in G

Under this transformation, the physical links transform covariantly:

    U_mu(x) -> Omega(x)^\dagger U_mu(x) Omega(x + mu_hat)

Usage
-----

V_mu(x) may be viewed as "prepotentials," whose group-valued forward difference
along direction mu produces the physical gauge link U_mu(x).

Functions Provided
------------------

- `link_to_prelink(U, prefix_dims=1, sites_before_link=True)`
  Constructs the full prelink tensor from gauge links, recursively integrating
  along increasing subspace dimensions (lines → planes → cubes → hypercubes)
  starting from the lattice origin. The value at the lattice origin is fixed
  (using the Polyakov loop in mu=0 direction), reducing the semi-global gauge
  symmetry to a global symmetry.

- `prelink_to_link(V, prefix_dims=1, sites_before_link=True)`
  Reconstructs the original gauge link variables from prelinks.

- `prelinks_to_left_right_pairs(V, prefix_dims=1, sites_before_link=True)`
  Produces left and right sub-tensors from prelinks.

Conventions
-----------

- Spatial axes may precede (`sites_before_link=True`) or follow the link axis.
- The first batch/channel axes are given by `prefix_dims`.
- All tensors are assumed to have matrix components of size Nc x Nc.
"""

# pylint: disable=invalid-name

from typing import List
import torch


__all__ = [
    "link_to_prelink",
    "prelink_to_link",
    "prelink_to_left_right_pair",
    "reduce_prelink_symmetry_to_global"
]


# =============================================================================
def link_to_prelink(
    U: torch.Tensor,
    prefix_dims: int = 1,
    sites_before_link: bool = True
) -> torch.Tensor:
    """
    Build the link prepotentials (prelinks) from gauge links.

    This function integrating gauge links along each lattice direction to
    produce prepotentials (prelinks) `V_mu(x)`:

        V_mu(x + mu_hat) = V_mu(x) @ U_mu(x)

    The prelink at the lattice origin is set to Polyakov loop along mu=0.
    Subsequent integration is performed recursively along increasing subspace
    dimensions (lines → planes → cubes → hypercubes) using the recursive
    ordering scheme:

        - For k=1 (lines): integrate along single axes from origin.
        - For k=2 (planes): integrate in sequences like (1,0) then (0,1).
        - For k=3 (cubes): integrate in sequences like (2,0,1), (0,1,2), etc.
        - The order ensures proper propagation of prelinks from the origin.

    Parameters
    ----------
    U : torch.Tensor
        Tensor of gauge links with shape
        `[batch..., n1, n2, ..., nd, ndim, Nc, Nc]`, where `nd` is the
        number of spatial dimensions, `ndim` is the number of link directions,
        and `Nc` is the size of the gauge group matrices.
    prefix_dims : int, default=1
        Number of leading batch/channel dimensions in the tensor.
    sites_before_link : bool, default=True
        If True, the spatial lattice axes precede the link direction axis.

    Returns
    -------
    torch.Tensor
        Prelink tensor `V` with shape `[batch..., n1+1, ..., nd+1, Nc, Nc]`,
        representing the prepotentials along each lattice link. This tensor
        can be used as input to `prelink_to_link` to recover `U`.

    Notes
    -----
    Gauge symmetry structure
    ~~~~~~~~~~~~~~~~~~~~~~~~~

    In the abstract prelink formulation, the variables V_mu(x) admit
    a semi-global symmetry of the form

        V_mu(x) -> Q(x, mu) V_mu(x),   Q(x, mu) in G

    where Q(x, mu) must be the same for all sites in the equivalence class

        [x, mu] = { x + n * mu_hat | n = 0, ..., N_mu }

    Under this transformation, the physical links remain unchanged:

         U_mu(x) -> U_mu(x)

    In the present construction, this freedom is fixed by:

    1. Choosing the Polyakov loop along mu = 0 at the origin, and
    2. Recursively integrating prelinks outward from that reference point.

    This procedure correlates the different directions and reduces the
    semi-global symmetry to a single global gauge transformation, related to
    the guage symmetry of the Polyakov loop along mu = 0 at the origin.
    """
    # Axis corresponding to the link direction mu
    link_axis = -3 if sites_before_link else prefix_dims

    # Number of spatial lattice dimensions (exclude prefix, link, matrix)
    spatial_ndim = U.ndim - prefix_dims - 3

    # Separate links by direction mu
    links_stack = torch.unbind(U, dim=link_axis)

    # Allocate container for resulting prelinks
    prelinks_stack: list[torch.Tensor] = [None] * spatial_ndim

    # Step 0: Fix gauge freedom using origin Polyakov loop in mu=0
    V0 = calc_origin_polyakov(links_stack[0], prefix_dims)
    prelinks_stack[0] = V0  # initial value for first axis

    # Recursive integration: lines → planes → cubes → hypercubes
    for hyperplane_ndim in range(1, spatial_ndim + 1):
        # Axes involved in this subspace
        varying_axes = list(range(hyperplane_ndim))

        # Initial value for next step
        V0 = prelinks_stack[0]

        # Cut last item along mu=0 if subspace has more than 1 axis
        if hyperplane_ndim > 1:
            # V0 comes from prelink[mu=0] which is extended in mu=0 direction.
            # Removing the last element avoids overshooting the lattice size.
            V0 = torch.narrow(V0, prefix_dims, 0, V0.shape[prefix_dims] - 1)

        # Loop over axes in reversed order (last → first) for integration
        for mu in varying_axes[::-1]:
            # Extract subspace of links for this integration
            links_cut = select_spatial_cut(
                links_stack[mu],
                prefix_dims=prefix_dims,
                varying_axes=varying_axes
            )
            # shape: [batch..., n0, ..., nd, Nc, Nc]

            # Integrate along mu starting from current V0
            prelinks_stack[mu] = integrate_prelink_along_axis(
                links_cut,
                V0,
                dim_mu=prefix_dims + mu
            )
            # shape: [batch..., ..., 1 + n_mu, ..., Nc, Nc]

            # Prepare initial value for next axis in this subspace
            if mu == 0:
                continue  # no preceeding axis

            # Select subspace excluding next axis (mu-1)
            boundary_varying_axes = [k for k in varying_axes if k != (mu - 1)]
            V0 = select_spatial_cut(
                prelinks_stack[mu],
                prefix_dims=prefix_dims,
                varying_axes=boundary_varying_axes
            )

            # V0 comes from prelink[mu] which is extended in mu direction.
            # Removing the last element avoids overshooting the lattice size.
            extended_dim = prefix_dims + mu - 1  # -1 because V0 is already cut
            V0 = torch.narrow(V0, extended_dim, 0, V0.shape[extended_dim] - 1)

    # Stack prelinks along the link direction to form full V tensor
    V = torch.stack(pad_to_max_shape(prelinks_stack), dim=link_axis)
    return V


# =============================================================================
def prelink_to_link(
    V: torch.Tensor,
    prefix_dims: int = 1,
    sites_before_link: bool = True
):
    r"""
    Convert link prepotentials (prelinks) to gauge links.

    The links are computed as a group-valued forward difference:

        U_mu(x) = V_mu(x)^\dagger V_mu(x + mu_hat)

    The prelinks V_mu(x) live on an extended lattice of size (N_mu + 1) along
    each spatial direction. The resulting links live on a lattice of size N_mu
    along every direction.

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

    Returns
    -------
    torch.Tensor
        Tensor of physical gauge links with reduced lattice sizes.
    """
    # Axis corresponding to link direction mu
    link_axis = -3 if sites_before_link else prefix_dims

    # Number of spatial lattice dimensions (exclude prefix, link, matrix)
    spatial_ndim = V.ndim - prefix_dims - 3

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

        link_mu = left.adjoint() @ right

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


# =============================================================================
def reduce_prelink_symmetry_to_global(V: torch.Tensor, **kwargs):
    """
    Reduce the semi-global prelink symmetry to a global one.

    This function composes the maps

        V → U = prelink_to_link(V)
        U → V' = link_to_prelink(U)

    and returns the reconstructed prelinks V'.

    While `prelink_to_link(link_to_prelink(U))` is the identity on links,
    the reverse composition

        link_to_prelink(prelink_to_link(V))

    is generally *not* the identity on prelinks. Instead, it fixes the
    integration convention used to construct prelinks from links (e.g.,
    by choosing a reference origin), thereby removing the residual
    semi-global gauge freedom of the prelink representation.

    As a result, the returned tensor transforms only under a single
    global gauge transformation rather than the original semi-global
    symmetry.
    """
    return link_to_prelink(prelink_to_link(V))


# =============================================================================
def prelink_to_left_right_pair(
    V: torch.Tensor,
    prefix_dims: int = 1,
    sites_before_link: bool = True
):
    r"""
    Split prelinks into left/right shifted pairs along each lattice direction.

    For each direction mu, this function extracts the two neighboring
    prelink tensors

        left  = V_mu(x)
        right = V_mu(x + mu_hat)

    restricted to the lattice region where they are defined.
    These pairs can be used to reconstruct gauge links via the
    group-valued forward difference

        U_mu(x) = left(x)^\dagger @ right(x)

    All spatial dimensions are reduced from size (N + 1) to N so that
    left and right have identical shapes.

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

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        A pair `(left, right)` where each has shape
        `[batch..., n1, ..., nd, ndim, Nc, Nc]`.
        These tensors contain the aligned prelink values needed to
        reconstruct the gauge links.
    """
    # Axis corresponding to link direction mu
    link_axis = -3 if sites_before_link else prefix_dims

    # Number of spatial lattice dimensions (exclude prefix, link, matrix)
    spatial_ndim = V.ndim - prefix_dims - 3

    # Separate prelinks by direction mu
    prelinks_stack = torch.unbind(V, dim=link_axis)

    # Allocate container for resulting left & right prelinks
    left_prelinks_stack: List[torch.Tensor] = [None] * spatial_ndim
    right_prelinks_stack: List[torch.Tensor] = [None] * spatial_ndim

    for mu, prelink_mu in enumerate(prelinks_stack):

        dim_mu = prefix_dims + mu  # axis corresponding to direction mu
        len_mu = prelink_mu.shape[dim_mu]

        # Group-valued forward difference along direction mu:
        left = torch.narrow(prelink_mu, dim_mu, 0, len_mu - 1)
        right = torch.narrow(prelink_mu, dim_mu, 1, len_mu - 1)

        # Clamp all other spatial dimensions to size N_nu
        for nu in range(spatial_ndim):
            if nu == mu:
                continue
            dim_nu = prefix_dims + nu  # axis corresponding to direction nu
            len_nu = prelink_mu.shape[dim_nu]
            left = torch.narrow(left, dim_nu, 0, len_nu - 1)
            right = torch.narrow(right, dim_nu, 0, len_nu - 1)

        left_prelinks_stack[mu] = left
        right_prelinks_stack[mu] = right

    # Restore original layout with link-direction axis
    left_prelinks = torch.stack(left_prelinks_stack, dim=link_axis)
    right_prelinks = torch.stack(right_prelinks_stack, dim=link_axis)

    return left_prelinks, right_prelinks


# =============================================================================
def integrate_prelink_along_axis(
    U: torch.Tensor,
    V0: torch.Tensor,
    dim_mu: int
) -> torch.Tensor:
    """
    Integrate prelinks along a given axis from a specified origin.

    Given a tensor of links U, this function constructs the prelinks V
    by discrete group integration along the axis `dim_mu`:

        V[..., i+1, :, :] = V[..., i, :, :] @ U[..., i, :, :]

    Parameters
    ----------
    U : torch.Tensor
        Tensor of links.

    V0 : torch.Tensor
        Initial prelink value at index 0 along `dim_mu`.
        Shape must match U with the `dim_mu` axis removed.

    dim_mu : int
        Axis along which to integrate.

    Returns
    -------
    torch.Tensor
        Prelink tensor V of shape as U, except the size along `dim_mu` is N+1.
    """

    # Length along integration axis
    N = U.shape[dim_mu]

    # Construct V shape (extend dim_mu by +1)
    V_shape = list(U.shape)
    V_shape[dim_mu] = N + 1

    V = torch.zeros(V_shape, dtype=U.dtype, device=U.device)

    # Set initial value V[..., 0, :, :] = V0
    idx0 = [slice(None)] * V.ndim
    idx0[dim_mu] = 0
    V[tuple(idx0)] = V0

    # Forward group integration
    for i in range(N):
        # Prepare index slices for current, next prelink, & corresponding link
        idx_current = [slice(None)] * V.ndim
        idx_next = [slice(None)] * V.ndim

        idx_current[dim_mu] = i
        idx_next[dim_mu] = i + 1

        # Compute V[..., i+1, :, :] = V[..., i, :, :] @ U[..., i, :, :]
        V[tuple(idx_next)] = V[tuple(idx_current)] @ U[tuple(idx_current)]

    return V


# =============================================================================
def calc_origin_polyakov(
    U_mu0: torch.Tensor,
    prefix_dims: int
) -> torch.Tensor:
    """
    Compute Polyakov loop in mu=0 direction starting from the spatial origin.

    Parameters
    ----------
    U_mu0 : torch.Tensor
        Links in mu=0 direction.
        Shape: [batch..., n0, n1, ..., nd, Nc, Nc]

    prefix_dims : int
        Number of leading batch/channel dimensions in the tensor.

    Returns
    -------
    torch.Tensor
        One matrix per batch element; Shape: [batch..., Nc, Nc].
    """

    # Extract the μ=0 line through the origin
    line = select_spatial_cut(
        U_mu0,
        prefix_dims=prefix_dims,
        varying_axes=[0],  # only μ=0 free
    )
    # shape: [batch..., n0, Nc, Nc]

    mu_axis = prefix_dims  # Spatial mmu=0 axis

    # Ordered product along mu=0
    result = line.select(mu_axis, 0)
    for i in range(1, line.shape[mu_axis]):
        result = result @ line.select(mu_axis, i)

    return result


# =============================================================================
def select_spatial_cut(
    x: torch.Tensor,
    prefix_dims: int,
    varying_axes: list[int],
) -> torch.Tensor:
    """
    Extract a spatial cut from tensor x by keeping only selected spatial axes
    free and fixing the others to zero.

    Parameters
    ----------
    x : torch.Tensor
        Shape: [batch..., n1, ..., nd, Nc, Nc]

    prefix_dims : int
        Number of leading batch/channel dimensions in the tensor.

    varying_axes : list[int]
        Spatial directions (0-based) to keep. Others are fixed to 0.

    Returns
    -------
    torch.Tensor
        Tensor restricted to the specified subspace.
    """

    idx = [slice(None)] * x.ndim

    spatial_ndim = x.ndim - prefix_dims - 2  # exclude prefix & matrix axes

    for k in range(spatial_ndim):
        if k not in varying_axes:
            idx[prefix_dims + k] = 0

    return x[tuple(idx)]


# =============================================================================
def pad_to_max_shape(tensor_list, pad_value=0):
    """
    Pad all tensors in a list to the same shape along each axis
    by concatenating zeros.

    Parameters
    ----------
    tensor_list : list[torch.Tensor]
        List of tensors to pad.

    pad_value : scalar, default=0
        Value to pad with.

    Returns
    -------
    list[torch.Tensor]
        List of tensors all padded to the same shape.
    """
    # Determine max shape along each axis
    max_shape = list(tensor_list[0].shape)
    for t in tensor_list[1:]:
        max_shape = [max(ms, s) for ms, s in zip(max_shape, t.shape)]

    padded_list = []
    for t in tensor_list:
        pad_sizes = []
        # Compute how many zeros to add at the end of each dimension
        for s, ms in zip(t.shape, max_shape):
            pad_sizes.append(ms - s)
        # Prepare padding in PyTorch format (last dimension first)
        pad_flat = []
        for p in reversed(pad_sizes):
            pad_flat.extend([0, p])
        padded_list.append(
            torch.nn.functional.pad(t, pad_flat, value=pad_value)
        )
    return padded_list
