# Copyright (c) 2026 Javad Komijani

r"""
Distribution of SU(3) eigenangles induced by a class function.

For a class function `J` on SU(3) -- by default the group commutator density
of `_sun_group_commutator_density` -- the Weyl integration formula turns the
measure `J(Z) dZ` into a measure on the eigenangles alone,

    p(phi_1, phi_2, phi_3) ~ |Delta|^2 J(phi),
    Delta = prod_{k<l} (e^{i phi_k} - e^{i phi_l}),

with `phi_1 + phi_2 + phi_3 = 0 (mod 2pi)` from `det Z = 1`. Two angles are
therefore free, and the distribution lives on a 2D plane: this module grids
that plane, evaluates the joint density on it, and integrates one axis out
to get the marginal of a single eigenangle.

This mirrors `normflow.lib.matrix_handles.su3_eigenphase`, which does the
same for `exp(-action)` in place of `J`; passing `log_density` recovers that
case exactly (`log_density = lambda eigvals: -action(diag_embed(eigvals))`).

Note that `J` diverges at `Z = 1` while `|Delta|^2` vanishes there to a
higher order, so the joint density is well behaved on the whole plane; the
grid is nevertheless kept marginally off the boundary of the cell.
"""

# pylint: disable=invalid-name

import math
import torch

from lattice_ml.linalg import log_unitary_conjugacy_vol

from ._sun_group_commutator_density import (
    compute_sun_group_commutator_log_density_from_eigvals,
)


__all__ = ["su3_group_commutator_eigangles_marginal_dist"]


# =============================================================================
def su3_eigangles_grid(
    bins: int,
    xlim=(-math.pi * 0.9999, math.pi * 0.9999),
    ylim=(-math.pi, math.pi),
):
    """Create a mesh grid of SU(3) eigenangles.

    The third angle is fixed by `det Z = 1`, i.e. `phi_3 = -(phi_1 + phi_2)`.

    Args:
        bins: Number of points along each axis.
        xlim: Limits for the first angle.
        ylim: Limits for the second angle.

    Returns:
        phi_1, phi_2, phi_3: Mesh grids of the three eigenangles, each of
        shape `(bins, bins)`.
    """
    x = torch.linspace(*xlim, bins)
    y = torch.linspace(*ylim, bins)
    phi_1, phi_2 = torch.meshgrid(x, y, indexing='xy')
    phi_3 = -(phi_1 + phi_2)
    return phi_1, phi_2, phi_3


# =============================================================================
def su3_group_commutator_eigangles_log_joint_dist(
    phi_1, phi_2, phi_3, log_density=None
):
    """Log of the joint eigenangle density, up to an additive constant.

    Args:
        phi_1, phi_2, phi_3: Mesh grids of the three eigenangles.
        log_density: Callable mapping eigenvalues of shape `(..., 3)` to a
            log class function of shape `(...,)`. Defaults to the group
            commutator density, `log J(Z)`.

    Returns:
        `log(|Delta|^2) + log_density`, shape `phi_1.shape`.
    """
    if log_density is None:
        log_density = compute_sun_group_commutator_log_density_from_eigvals

    phi = torch.stack([phi_1, phi_2, phi_3], dim=-1)
    eigvals = torch.exp(1j * phi).reshape(-1, 3)

    log_pdf = log_unitary_conjugacy_vol(eigvals) + log_density(eigvals)

    return log_pdf.reshape(*phi_1.shape)


# =============================================================================
def su3_group_commutator_eigangles_joint_dist(
    phi_1, phi_2, phi_3, log_density=None
):
    """Joint eigenangle density, up to a multiplicative constant.

    Exponential of `su3_group_commutator_eigangles_log_joint_dist`, shifted
    by its maximum so that the exponential cannot overflow; see that
    function for the arguments.

    Returns:
        Tensor of joint density values over the grid, shape `phi_1.shape`.
    """
    log_pdf = su3_group_commutator_eigangles_log_joint_dist(
        phi_1, phi_2, phi_3, log_density=log_density
    )
    return torch.exp(log_pdf - log_pdf.max())


# =============================================================================
def su3_group_commutator_eigangles_marginal_dist(
    log_density=None, bins: int = 200
):
    """Marginal density of a single SU(3) eigenangle.

    Generates the eigenangle grid, evaluates the joint density on it, and
    sums over the second axis. The result is normalized to integrate to one
    over the first axis.

    Args:
        log_density: Callable mapping eigenvalues of shape `(..., 3)` to a
            log class function of shape `(...,)`. Defaults to the group
            commutator density, `log J(Z)`.
        bins: Number of bins along each axis of the grid.

    Returns:
        phi: Eigenangle values along the first axis, shape `(bins,)`.
        marg_pdf: Normalized marginal density at those angles.
    """
    dx = 2 * math.pi / bins

    phi_1, phi_2, phi_3 = su3_eigangles_grid(bins)
    joint_pdf = su3_group_commutator_eigangles_joint_dist(
        phi_1, phi_2, phi_3, log_density=log_density
    )

    marg_pdf = torch.sum(joint_pdf, dim=1)
    marg_pdf = marg_pdf / (torch.sum(marg_pdf) * dx)

    return phi_1[0], marg_pdf
