# Copyright (c) 2025 Javad Komijani

"""
Wilson line convolutional layers for lattice gauge theory.

This module provides gauge-equivariant layers that update lattice gauge links
using short Wilson lines starting at the tail of a link and ending at its head.
"""

import torch

from .wilson_staples import compute_all_directional_staples
from .time_embedding import TimeModulatedWeight


__all__ = ["GaugeLinkSmear"]


# =============================================================================
class GaugeLinkSmear(torch.nn.Module):
    """
    Gauge-equivariant link smear layer for lattice gauge fields.

    Parameters
    ----------
    sites_before_link : bool, default=True
        Whether spatial lattice axes come before the link axis.
    """

    def __init__(self, sites_before_link: bool = True):
        super().__init__()

        self.sites_before_link = sites_before_link
        self._project_onto_algebra_space = anti_hermitian_traceless

        # Learnable time-dependent weight tensor
        self.weight_fn = TimeModulatedWeight(weight_shape=(1,))

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
           t (torch.Tensor): A 1D tensor representing time (batch dimension).
           x (torch.Tensor): Tensor containing the gauge links.

        Returns:
            torch.Tensor: Updated gauge field tensor with the same shape as x.
        """
        g = compute_all_directional_staples(
            x, sites_before_link=self.sites_before_link
        )

        algebra_force = self._project_onto_algebra_space(x @ g)
        weight = 0.1 * self.weight_fn(t).reshape(-1, *(1,)*(x.ndim - 1))
        return torch.matrix_exp(weight * algebra_force) @ x


# =============================================================================
def anti_hermitian_traceless(x: torch.Tensor) -> torch.Tensor:
    """
    Project the input onto the space of traceless anti-Hermitian matrices.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor with square matrices in the last two dimensions.

    Returns
    -------
    torch.Tensor
        Tensor of the same shape as `x`, where each matrix is projected to be
        anti-Hermitian and traceless.
    """
    # Anti-Hermitian part
    x = (x - x.adjoint()) / 2

    # Remove trace by subtracting identity * (trace / n_c)
    n_c = x.shape[-1]
    reduced_tr = torch.mean(torch.linalg.diagonal(x), dim=-1, keepdim=True)
    mu = torch.diag_embed(torch.repeat_interleave(reduced_tr, n_c, dim=-1))
    return x - mu
