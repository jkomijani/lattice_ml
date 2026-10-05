# Copyright (c) 2026 Javad Komijani

r"""
Exact sampler for the SU(2) group commutator equation.

Given Z in SU(2), draws (X, Y) in SU(2) x SU(2) satisfying Z = X Y X^dagger
Y^dagger, from the exact conditional law that Haar x Haar induces on (X, Y)
given Z. Plain Haar x Haar is only the *marginal*, recovered by additionally
drawing Z from its own commutator-induced density;
here Z is fixed, so (X, Y) follows a generally non-uniform disintegration.

D and Q, in the (D, Q, Z) encoding are initially drawn Haar; D is already
exactly uniform given Z, and Q is transformed by the closed-form conditional
CDF `sample_for_euler_angle` before being handed to the decoder.
"""

# pylint: disable=invalid-name

import torch

from lattice_ml.lie_groups._euler_angles_su2 import (
    su2_to_euler_angles, euler_angles_to_su2,
)
from lattice_ml.lie_groups._sun_group_commutator import (
    eig_with_options, decode_sun_group_commutator,
)

from ._unitary_group import rand_sun_group_like, rand_diagonal_sun_group_like


__all__ = ["sample_su2_group_commutator_xy_given_z"]


# =============================================================================
def sample_su2_group_commutator_xy_given_z(Z: torch.Tensor):
    r"""
    Exact conditional sampler for X, Y in SU(2) given Z where `Z = X Y X† Y†`.

    Parameters
    ----------
    Z : torch.Tensor
        Special unitary input matrix of shape `(..., 2, 2)`.

    Returns
    -------
    X, Y : torch.Tensor
        One random pair of special unitary matrices satisfying `Z = X Y X† Y†`,
        drawn from the conditional commutator-induced law given Z.
    log_importance_weight : torch.Tensor | float
        The importance weight `log p(X,Y|Z) − log q(X,Y|Z)`, where p is the
        exact PDF and q is proposal PDF. Here, 0 because the sampler is exact.
    """
    eig_options = {
        'sort_by': 'zero-sum-angle',
        'randomize_vector_phase': True,
        'project_to_sun': True,
    }
    lam_z, Omega_z = eig_with_options(Z, **eig_options)
    theta = torch.angle(lam_z[..., 1]).abs()  # Z's class angle, in [0, pi]

    D = rand_diagonal_sun_group_like(Z)
    Q = rand_sun_group_like(Z)

    # b is already Uniform(0, 1) since Q is Haar; reshape it to the exact
    # conditional CDF given theta, leaving the other two channels untouched.
    # By inverse-CDF sampling, b's own density is exp(-log_db_ds) -- exactly
    # logq, with no further Jacobian needed (D and the other two Euler
    # channels of Q stay flat, contributing nothing to the density ratio).
    P = Omega_z.adjoint() @ Q
    a, b, c = su2_to_euler_angles(P, coords='uniform')
    b, _ = sample_for_euler_angle(b.unsqueeze(-1), theta.unsqueeze(-1))
    P = euler_angles_to_su2((a, b.squeeze(-1), c), coords='uniform')
    Q = Omega_z @ P

    X, Y = decode_sun_group_commutator(D, Q, Z)
    log_importance_weight = 0
    return (X, Y), log_importance_weight


# =============================================================================
def sample_for_euler_angle(s: torch.Tensor, theta: torch.Tensor):
    r"""
    Transform s ~ Uniform(0,1) into b with conditional CDF F(b|theta), where

        F(b|theta) = [arctan((2b - 1) cot(theta/2)) + (pi - theta)/2]
                     / (pi - theta)

    via inverse transform sampling: b = F^{-1}(s|theta).

    Args:
        s: Uniform(0,1) draws, any shape.
        theta: Broadcastable with `s`; must lie in (0, pi).

    Returns:
        b: F^{-1}(s|theta), same shape as broadcast(s, theta).
        log_db_ds: log |db/ds|, same shape with the last axis dropped.
    """
    half_span = (torch.pi - theta) / 2.0
    u = s * (torch.pi - theta) - half_span
    tan_half_theta = torch.tan(theta / 2.0)

    b = 0.5 + 0.5 * tan_half_theta * torch.tan(u)
    log_db_ds = (
        torch.log(torch.pi - theta)
        + torch.log(tan_half_theta)
        - torch.log(torch.cos(u) ** 2 * 2.0)
    )
    return b, log_db_ds.squeeze(-1)
