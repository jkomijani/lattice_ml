# Copyright (c) 2026 Javad Komijani

r"""
Holonomy clover sums and HMC force for the Wilson gauge action in the holonomy
parametrization.

.. warning::
    The force (HMC) calculation is not fully correct at the corner sites; see
    the module docstring of ``holonomy_action.py`` for details.

The reference holonomies are

    h_{mu, 0}(x) = V_mu(x) V_0(x)^\dagger,   mu = 1, ..., d-1,

defined on the extended lattice (size N_mu + 1 along each direction).
The four corner holonomies satisfy a constraint in all dimensions; in 3D+,
the values h_{mu, 0}(t=N_0) for mu >= 2 are additionally constrained by
h_{1, 0}(t=N_0) (see the warning above).
All other holonomies are derived via

    h_{mu, nu}(x) = h_{mu, 0}(x) @ h_{nu, 0}(x)^\dagger .

The plaquette in terms of holonomies reads:

    P_{mu,nu}(x)
    = h†_{mu,nu}(x) h_{mu,nu}(x+mu) h†_{mu,nu}(x+mu+nu) h_{mu,nu}(x+nu) .


Why "clover"
------------
In the Wilson action, each holonomy h_{mu,nu}(x) participates in exactly four
plaquettes in the (mu, nu) plane. Using the cyclic property of the trace, all
four can be written so that they end at x — forming a clover-leaf pattern
around x, analogous to the Sheikholeslami-Wohlert discretization of F_{mu,nu}.
"""

# pylint: disable=invalid-name

from typing import List
import torch
import torch.nn.functional as F

from .holonomy import compute_h_munu


__all__ = [
    'compute_holonomy_clover',
    'compute_directional_holonomy_clover',
    'compute_planar_holonomy_clover',
]


# =============================================================================
def compute_holonomy_clover(
    h: torch.Tensor,
    prefix_dims: int = 1,
    sites_before_link: bool = True,
    sum_over_nu: bool = True,
    plus_signature: bool = True,
):
    r"""Compute the holonomy clover sum G_{mu,0}(x) for all mu = 1, ..., d-1.

    Parameters
    ----------
    h : torch.Tensor
        Holonomy tensor h_{mu, 0} for mu = 1, ..., d-1.
        Shape: `(...batch..., N_0+1, ..., N_{d-1}+1, d-1, Nc, Nc)`
        if `sites_before_link=True`, else
        `(...batch..., d-1, N_0+1, ..., N_{d-1}+1, Nc, Nc)`.
    prefix_dims : int, default=1
        Number of leading batch / channel dimensions.
    sites_before_link : bool, default=True
        Whether spatial lattice axes precede the link axis.
    sum_over_nu : bool, default=True
        If True, sum contributions over all nu != mu.
        If False, stack them along a new axis at `prefix_dims`.
    plus_signature : bool, default=True
        Passed to :func:`compute_directional_holonomy_clover`.

    Returns
    -------
    torch.Tensor
        Same shape as `h` (if `sum_over_nu=True`), or with an extra axis
        at `prefix_dims` (if `sum_over_nu=False`).
    """
    # d = total number of lattice directions (including temporal)
    spatial_ndim = h.ndim - prefix_dims - 3

    kws = {
        'prefix_dims': prefix_dims,
        'sites_before_link': sites_before_link,
        'sum_over_nu': sum_over_nu,
        'plus_signature': plus_signature,
    }

    # Independent holonomies are h_{mu, 0} for mu = 1, ..., d-1
    result_stack: List[torch.Tensor] = [None] * (spatial_ndim - 1)

    for mu in range(1, spatial_ndim):
        nu_list = [nu for nu in range(spatial_ndim) if nu != mu]
        result_stack[mu - 1] = compute_directional_holonomy_clover(
            h, mu, nu_list, **kws
        )

    if not sum_over_nu:
        prefix_dims += 1

    # Stack the results along the link-axis to recreate the full tensor
    link_axis = -3 if sites_before_link else prefix_dims
    return torch.stack(result_stack, dim=link_axis)


# =============================================================================
def compute_directional_holonomy_clover(
    h: torch.Tensor,
    mu: int,
    nu_list: List[int],
    prefix_dims: int = 1,
    sites_before_link: bool = True,
    sum_over_nu: bool = True,
    plus_signature: bool = True,
):
    r"""Compute the holonomy clover for direction `mu` summed over `nu_list`.

    Parameters
    ----------
    h : torch.Tensor
        Holonomy tensor (see :func:`compute_holonomy_clover`).
    mu : int
        Direction of the independent holonomy h_{mu,0}. Must satisfy mu >= 1.
    nu_list : list of int
        Transverse directions to accumulate over.
    prefix_dims : int, default=1
        Number of leading batch / channel dimensions.
    sites_before_link : bool, default=True
        Whether spatial lattice axes precede the link axis.
    sum_over_nu : bool, default=True
        If True, sum contributions over all nu != mu.
        If False, stack them along a new axis at `prefix_dims`.
    plus_signature : bool, default=True
        Passed to :func:`compute_planar_holonomy_clover`.

    Returns
    -------
    torch.Tensor
        Summed (or stacked) planar holonomy clover tensors for direction `mu`.
    """
    kws = {
        'prefix_dims': prefix_dims,
        'sites_before_link': sites_before_link,
        'plus_signature': plus_signature
    }

    planar = [compute_planar_holonomy_clover(h, mu, nu, **kws)
              for nu in nu_list]

    if sum_over_nu:
        return sum(planar)

    return torch.stack(planar, dim=prefix_dims)


def compute_planar_holonomy_clover(
    h: torch.Tensor,
    mu: int,
    nu: int,
    prefix_dims: int = 1,
    sites_before_link: bool = True,
    plus_signature: bool = True
):
    r"""Compute the planar holonomy clover sum in the `(mu, nu)` plane.

    For each site x, sums four plaquettes all anchored at x in the (mu, nu)
    plane as described below.

    Writing a = h_{mu,nu}(x), b = h_{mu,nu}(x+mu), etc., the four plaquettes
    (mu = horizontal, nu = vertical) are::

        >>>   upper-left          upper-right
        >>>   i   d†              d   c†
        >>>   g†  a               a†  b
        >>>
        >>>   lower-left          lower-right
        >>>   g   a†              a   b†
        >>>   j†  f               f†  e

    Let Q_ij denote the counterclockwise (CCW) plaquette whose inner corner
    is x: Q01 (upper-right, x at BL), Q10 (lower-left, x at TR), Q00
    (upper-left, x at BR), Q11 (lower-right, x at TL). In the Wilson action,
    both Q_ij and Q†_ij appear as `Re Tr [Q_ij] = Re Tr[Q†_ij])`. Using the
    cyclic property of the trace and the fact that Q and Q† can be swapped,
    all four contributions can be written ending at h†_{mu,nu}(x) = a†:

    ===========  ===========  =================
    Quadrant     Orientation  Expression at x
    ===========  ===========  =================
    upper-right  CCW          Q01  = b c† d a†
    lower-left   CCW          Q10  = g j† f a†
    upper-left   CW           Q00† = g i† d a†
    lower-right  CW           Q11† = b e† f a†
    ===========  ===========  =================

    In computation of the force, all four terms contribute with positive sign
    as `Q01 + Q10 + Q00† + Q11†`, which corresponds to the default choice of
    `plus_signature=True`.

    The option `plus_signature=False` is introduced for flexibility, but it
    might be removed in future.

    See the module docstring for the full force derivation.

    Parameters
    ----------
    h : torch.Tensor
        Holonomy h_{mu,0} for mu > 0. After batch axes, spatial lattice axes
        come first (if sites_before_link=True), then the holonomy mu axis,
        then matrix indices.
    mu : int
        First lattice direction (mu >= 1 for independent holonomies).
    nu : int
        Second lattice direction. Must satisfy `mu != nu`.
    prefix_dims : int, default=1
        Number of leading batch / channel dimensions.
    sites_before_link : bool, default=True
        Whether spatial lattice axes come before the link axis.
    plus_signature : bool, default=True
        If True, all four terms are positive: `Q01 + Q10 + Q00† + Q11†`
        (correct for HMC force). If False, CW terms are negated:
        `Q01 + Q10 - Q00† - Q11†`.

    Returns
    -------
    torch.Tensor
        Planar clover sum with same batch, lattice, and matrix dimensions as
        the holonomy field, with the link-direction axis removed.
    """
    assert mu != nu, f"mu ({mu}) and nu ({nu}) must be different!"

    kwargs = {
        'prefix_dims': prefix_dims,
        'sites_before_link': sites_before_link
    }

    h_munu = compute_h_munu(h, mu, nu, **kwargs)

    dim_mu = prefix_dims + mu  # tensor axis for lattice direction mu
    dim_nu = prefix_dims + nu  # tensor axis for lattice direction nu

    # Step 1: zero-pad one layer at each boundary along mu and nu
    out = pad_zero_boundary_2d(h_munu, dim_mu, dim_nu)

    # Step 2: h(i) @ h†(i+1) along nu  →  length reduces by 1
    out = forward_neighbor_product(out, dim_nu)

    # Step 3: f†(j+1) @ (f(j + 2) ± f(j)) along mu  →  length reduces by 2
    if plus_signature:
        out_right = neighbor_sum_product(out, dim_mu)
        out_left = neighbor_sum_product(out.adjoint(), dim_mu)
    else:
        out_right = neighbor_diff_product(out, dim_mu)
        out_left = neighbor_diff_product(out.adjoint(), dim_mu)

    # Step 4: forward difference with zero BCs along nu → length increases by 1
    n = out.shape[dim_nu] - 1
    if plus_signature:
        out = out_right.narrow(dim_nu, 1, n) + out_left.narrow(dim_nu, 0, n)
    else:
        out = out_right.narrow(dim_nu, 1, n) - out_left.narrow(dim_nu, 0, n)

    return out


# =============================================================================
def pad_zero_boundary_2d(x: torch.Tensor, dim0: int, dim1: int):
    """
    Add zero padding of width 1 on both sides of dimensions `dim0` and `dim1`.

    Pad `x` with one layer of zeros on both sides of lattice directions
    `dim0` and `dim1`.

    The lengths of axes `dim0` and `dim1` increase by 2; all other axes are
    unchanged.
    """
    # Padding in PyTorch's convention: last dimension is fastest-changing.
    pad = [0] * (2 * x.ndim)
    for dim in (dim0, dim1):
        idx = x.ndim - dim - 1
        pad[2 * idx] = 1  # start
        pad[2 * idx + 1] = 1  # end
    return F.pad(x, pad, value=0)


def forward_neighbor_product(x: torch.Tensor, dim: int):
    """
    Multiply neighboring matrices along direction `dim`.

    Returns

        x[i] @ x†[i+1]

    The output length along `dim` is reduced by 1.
    """
    n = x.shape[dim]
    a = torch.narrow(x, dim, 0, n - 1)
    b = torch.narrow(x, dim, 1, n - 1)
    return a @ b.adjoint()


def neighbor_sum_product(x: torch.Tensor, dim: int):
    """
    Nearest-neighbor sum contraction along direction `dim`.

    Returns

        (x[i+2] + x[i]) @ x†[i+1]

    for interior sites. The output length along `dim` is reduced by 2.
    """
    n = x.shape[dim]
    lower = torch.narrow(x, dim, 0, n - 2)
    middle = torch.narrow(x, dim, 1, n - 2)
    upper = torch.narrow(x, dim, 2, n - 2)
    return (upper + lower) @ middle.adjoint()


def neighbor_diff_product(x: torch.Tensor, dim: int):
    """
    Nearest-neighbor sum contraction along direction `dim`.

    Returns

        (x[i+2] - x[i]) @ x†[i+1]

    for interior sites. The output length along `dim` is reduced by 2.
    """
    n = x.shape[dim]
    lower = torch.narrow(x, dim, 0, n - 2)
    middle = torch.narrow(x, dim, 1, n - 2)
    upper = torch.narrow(x, dim, 2, n - 2)
    return (upper - lower) @ middle.adjoint()
