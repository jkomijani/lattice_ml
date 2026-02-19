# Copyright (c) 2025 Javad Komijani

"""
Staples calculations for Lattice Gauge Theory.

This module provides functions to compute planar staples and sums of staples
for links in specified directions, as used in Wilson gauge action computations.
"""

from typing import Tuple, List
import torch


__all__ = [
    'compute_staples',
    'compute_directional_staples',
    'compute_planar_staples'
]

matmul = torch.matmul


def compute_staples(
    x: torch.Tensor,
    prefix_dims: int = 1,
    sites_before_link: bool = True,
    sum_over_staples: bool = True,
    is_string: bool = False
):
    """Compute the staples for all link directions.

    For each link direction 'mu', this function computes the staples in all
    planes spanned by ('mu', 'nu') for every perpendicular direction 'nu'.
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
    sum_over_staples : bool, default=True
        If True, returns the sum over all staples for each link direction.
        If False, returns the individual planar staples, including the upper
        and lower ones, stacked along a new axis.

    Returns
    -------
    torch.Tensor
        If sum_over_staples=True:
            Contains the sum of staples for all link directions. Its shape is
            identical to `x`, with the link-direction axis containing the
            summed staples for each direction `mu`.
        If sum_over_staples=False:
            Contains individual planar staples for each nu direction, including
            both upper and lower contributions. The planar staples are stacked
            along a new axis inserted at `prefix_dims`.

    Notes
    -----
    The staples are defined such that the expression `x @ g` is gauge
    covariant, where `g` is the output of this function. Using this definition,
    the Wilson gauge action can be expressed is proportional to

        ReTr(x @ g) + ...

    where the sum over lattice sites and directions is implied.
    """
    # Prepare keyword arguments to pass to compute_directional_staples
    kws = {
        'prefix_dims': prefix_dims,
        'sites_before_link': sites_before_link,
        'sum_over_staples': sum_over_staples,
        'is_string': is_string
    }

    # Determine the number of spatial dimensions
    spatial_ndim = x.ndim - prefix_dims - 3  # exclude batch, direction, matrix

    # Initialize a list to store staples sums for each direction 'mu'
    staples_stack: List[torch.Tensor] = [None] * spatial_ndim

    # Loop over each link direction 'mu'
    for mu in range(spatial_ndim):
        nu_list = [nu for nu in range(spatial_ndim) if nu != mu]

        # Compute the staples sum for this direction
        staples_stack[mu] = compute_directional_staples(x, mu, nu_list, **kws)

    if not sum_over_staples:
        # All staples are stacked along a new axis inserted at `prefix_dims`.
        prefix_dims += 1

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
    sum_over_staples: bool = True,
    is_string: bool = False
):
    """Compute the staples in the mu-nu planes for a given link direction `mu`
    and all perpendicular directions listed in `nu_list`.

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
        List of perpendicular directions over which to calculate the staples.
    prefix_dims : int, default=1
        Number of leading batch and channel dimensions in the tensor.
        For example, if x.shape = (batch, channel, Lx, Ly, Lz, Lt, mu, Nc, Nc),
        then prefix_dims=2. If only a single batch dimension, prefix_dims=1.
    sites_before_link : bool, default=True
        Whether the spatial lattice axes come before the link axis.
    sum_over_staples : bool, default=True
        If True, returns the sum over all nu in nu_list.
        If False, returns the individual planar staples, including the upper
        and lower ones, stacked along a new axis.

    Returns
    -------
    torch.Tensor
        If sum_over_staples=True:
            Contains the sum of staples for links in direction mu.
            Shape is the same as x, except the link-direction axis is removed.
        If sum_over_staples=False:
            Contains individual planar staples for each nu direction, including
            both upper and lower contributions. The planar staples are stacked
            along a new axis inserted at `prefix_dims`.
    """
    kws = {
        'prefix_dims': prefix_dims,
        'sites_before_link': sites_before_link,
        'sum_over_staples': sum_over_staples
    }

    staples = [compute_planar_staples(x, mu, nu, **kws) for nu in nu_list] if not is_string else [compute_planar_string_staples(x, mu, nu, **kws) for nu in nu_list]

    if sum_over_staples:
        return sum(staples)

    # Flatten the list of (upper, lower) tuples to [U1, L1, U2, L2, ...]
    staples = [z for upper_lower_tuple in staples for z in upper_lower_tuple]
    # stack along a new axis corresponding to nu, and return
    return torch.stack(staples, dim=prefix_dims)

def compute_planar_string_staples(
    x: torch.Tensor,
    mu: int,
    nu: int,
    prefix_dims: int = 1,
    sites_before_link: bool = True,
    sum_over_staples: bool = True
):
    r"""
    Compute the staples in the mu-nu plane for a given link direction `mu` as
    used in the String gauge action.

    Staples are 7 string-link paths forming a nearly closed loop adjacent to a central
    link, used in the String gauge action in lattice gauge theory. The four (font/back,
    upper/lower) staples on the mu-nu plane can be visualized as:

        >>>        c d                 l c 
        >>>       *> >*               *< <*
        >>>      b^   ve             mv   ^b
        >>>      a^   vf        g    nv   ^a    o
        >>>       @<=<*      @<=<*   *>=>@    *>=>@ 
        >>>          g      av   ^f    o      n^   va
        >>>                 hv   ^k           q^   vh
        >>>                  *> >*             *< <*
        >>>                   i j               p i

        >>>     In full:              
        >>>     l c c d  
        >>>    *< <*> >* 
        >>>   mv   ^b  ve
        >>>   nvo  ^a gvf
        >>>    *>=>@<=<* 
        >>>   n^   va  ^f
        >>>   q^   vh  ^k
        >>>    *< <*> >* 
        >>>     p   i j  


    where `=>` represents the central string-link at site @ whose staples are being computed.
    Note the string-links live on the sites and so the side of the site they are shown on is
    not important. For example >* is the same as *<. However, when a string-link is pointing 
    away from its site we take the adjoint, so <* is the adjoint of >*. 

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
    sum_over_staples : bool
        If False, return the four staples as a tuple.
        If True, return their sum. Default is True.

    Returns
    -------
    torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
        If sum_over_staples=True:
            The sum of the upper and lower staples.
        If sum_over_staples=False:
            A tuple containing the upper and lower staples separately.

        In either case, each staple tensor has the same shape as `x`, except
        that the link-direction axis is removed.
    """

    # Extract the central and nu-direction links using unbind
    link_axis = -3 if sites_before_link else prefix_dims
    links = torch.unbind(x, dim=link_axis)

    x_mu = links[mu]  # =>@
    x_nu = links[nu]  # a'

    # Calculate links needed to form staples p: plus t:minus
    xpmu_mu = torch.roll(x_mu, -1, dims=prefix_dims + mu)   # g'  Means x plus mu in mu direction
    xpnu_mu = torch.roll(x_mu, -1, dims=prefix_dims + nu)   # c'
    xpmu_nu = torch.roll(x_nu, -1, dims=prefix_dims + mu)   # f
    xpnu_nu = torch.roll(x_nu, -1, dims=prefix_dims + nu)   # b

    xtmu_mu = torch.roll(x_mu, 1, dims=prefix_dims + mu)    # o'  Means x minus mu in mu direction
    xtnu_mu = torch.roll(x_mu, 1, dims=prefix_dims + nu)    # i'
    xtmu_nu = torch.roll(x_nu, 1, dims=prefix_dims + mu)    # n
    xtnu_nu = torch.roll(x_nu, 1, dims=prefix_dims + nu)    # h

    # Apply boundary conditions
    n_c = x.shape[-1]
    eye = torch.eye(n_c)

    # String boundary conditions: Set the surface x_mu = L_mu to identity.
    idpmu: List = [slice(None)] * x_mu.ndim
    idpmu[prefix_dims + mu] = -1
    xpmu_mu[tuple(idpmu)] = eye

    idpnu: List = [slice(None)] * x_nu.ndim
    idpnu[prefix_dims + nu] = -1
    xpnu_nu[tuple(idpnu)] = eye

    idtmu: List = [slice(None)] * x_mu.ndim
    idtmu[prefix_dims + mu] = 0
    xtmu_mu[tuple(idtmu)] = eye

    idtnu: List = [slice(None)] * x_nu.ndim
    idtnu[prefix_dims + nu] = 0
    xtnu_nu[tuple(idtnu)] = eye

    xpmupnu_mu = torch.roll(xpmu_mu, -1, dims=prefix_dims + nu) # d
    xpmupnu_nu = torch.roll(xpnu_nu, -1, dims=prefix_dims + mu) # e'

    xtmutnu_mu = torch.roll(xtmu_mu, 1, dims=prefix_dims + nu)  # p
    xtmutnu_nu = torch.roll(xtnu_nu, 1, dims=prefix_dims + mu)  # q'

    xpmutnu_mu = torch.roll(xpmu_mu, 1, dims=prefix_dims + nu)  # j
    xtmupnu_nu = torch.roll(xpnu_nu, 1, dims=prefix_dims + mu)  # m'
    
    xtmupnu_mu = torch.roll(xtmu_mu, -1, dims=prefix_dims + nu) # l
    xpmutnu_nu = torch.roll(xtnu_nu, -1, dims=prefix_dims + mu) # k'

    # Can rework to reduce repeated operations
    # abcdefg
    front_upper = matmul(
        matmul(matmul(x_nu.adjoint(), xpnu_nu), matmul(xpnu_mu.adjoint(), xpmupnu_mu)),
        matmul(matmul(xpmupnu_nu.adjoint(), xpmu_nu), xpmu_mu.adjoint())
    )
    # abclmno
    back_upper = matmul(
        matmul(matmul(x_nu.adjoint(), xpnu_nu), matmul(xpnu_mu.adjoint(), xtmupnu_mu)),
        matmul(matmul(xtmupnu_nu.adjoint(), xtmu_nu), xtmu_mu.adjoint())
    )
    # ahipqno
    back_lower = matmul(
        matmul(matmul(x_nu.adjoint(), xtnu_nu), matmul(xtnu_mu.adjoint(), xtmutnu_mu)),
        matmul(matmul(xtmutnu_nu.adjoint(), xtmu_nu), xtmu_mu.adjoint())
    )
    # ahijkfg
    front_lower = matmul(
        matmul(matmul(x_nu.adjoint(), xtnu_nu), matmul(xtnu_mu.adjoint(), xpmutnu_mu)),
        matmul(matmul(xpmutnu_nu.adjoint(), xpmu_nu), xpmu_mu.adjoint())
    )
        # In full:        
    #              
    #     l c c d  
    #    *< <*> >* 
    #   mv   ^b  ve
    #   nvo  ^a gvf
    #    *>=>@<=<* 
    #   n^   va  ^f
    #   q^   vh  ^k
    #    *< <*> >* 
    #     p   i j  
    #                 


    # Return the staples
    if sum_over_staples:
        return front_upper + front_lower + back_upper + back_lower
    return front_upper, front_lower, back_upper, back_lower



def compute_planar_staples(
    x: torch.Tensor,
    mu: int,
    nu: int,
    prefix_dims: int = 1,
    sites_before_link: bool = True,
    sum_over_staples: bool = True
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
    sum_over_staples : bool
        If False, return both the upper and lower staples as a tuple.
        If True, return their sum. Default is True.

    Returns
    -------
    torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
        If sum_over_staples=True:
            The sum of the upper and lower staples.
        If sum_over_staples=False:
            A tuple containing the upper and lower staples separately.

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
    if sum_over_staples:
        return staple_upper + staple_lower
    return staple_upper, staple_lower
