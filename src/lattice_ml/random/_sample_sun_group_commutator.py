# Copyright (c) 2026 Javad Komijani

r"""
Given any Z in SU(N), this module returns a pair (X, Y) in SU(N) satisfying

    Z = [X, Y]

where `[X, Y] = X Y X† Y†`.
"""

# pylint: disable=invalid-name

import torch

from ._sample_su2_group_commutator import sample_su2_group_commutator_xy_given_z
from ._sample_su3_group_commutator import sample_su3_group_commutator_xy_given_z


__all__ = ["sample_sun_group_commutator_xy_given_z"]


# =============================================================================
def sample_sun_group_commutator_xy_given_z(Z: torch.Tensor):
    """
    Sample X, Y in SU(N) given Z, `Z = X Y X† Y†`.

    Supports SU(2) and SU(3):
        For SU(2): the sampling uses an exact, closed-form map.
        For SU(3): the sampling uses a trained flow.

    Parameters
    ----------
    Z : torch.Tensor
        Special unitary input matrix of shape `(..., N, N)`, N in {2, 3}.

    Returns
    -------
    X, Y : torch.Tensor
        One random pair of special unitary matrices satisfying `Z = X Y X† Y†`,
        drawn from the conditional law given Z.
    log_importance_weight : torch.Tensor | float
        The importance weight `log p(X,Y|Z) − log q(X,Y|Z)`, where p is the
        exact PDF and q is the proposal PDF. `0` for N = 2 (exact sampler);
        a computed tensor for N = 3 (trained, not exact).
    """
    N = Z.shape[-1]

    if N == 2:
        return sample_su2_group_commutator_xy_given_z(Z)
    elif N == 3:
        return sample_su3_group_commutator_xy_given_z(Z)
    else:
        raise ValueError(f"N = {N}; only N = 2, 3 are supported.")
