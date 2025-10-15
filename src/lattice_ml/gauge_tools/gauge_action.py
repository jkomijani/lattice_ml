# Copyright (c) 2025 Javad Komijani

"""
Wilson gauge action and force calculations for lattice gauge theory.
"""

import torch

from .wilson_loops import compute_wilson_1x1_loop
from .wilson_staples import compute_all_directional_staples


__all__ = ['WilsonGaugeAction']


class WilsonGaugeAction:
    r"""
    Wilson gauge action and force calculations for lattice gauge theory.

    Implements the Wilson gauge action for SU(N_c) gauge group, together with
    the corresponding gauge force used in HMC simulations.

    The action is defined as:

    .. math::
        S = - \frac{\beta} {2 N_c} \sum_{\nu \neq \mu} Tr \text{Plaq}_{mu, nu}
          = - \frac{\beta} {N_c} \sum_{\nu < \mu} ReTr \text{Plaq}_{mu, nu} .

    Here:
        - :math:`\beta` is the inverse coupling.
        - :math:`\text{Plaq}_{\mu\nu}` is the plaquette in the (mu, nu) plane.

    Two axis layouts are supported:

    - `sites_before_link=True` (default): spatial lattice axes come before the
      link-direction axis.
    - `sites_before_link=False`: the link axis comes before the lattice sites.

    In both cases, tensors are assumed to have one batch axis as the first
    dimension.
    """

    def __init__(self, beta: float, sites_before_link: bool = True):

        """
        Initialize Wilson gauge action parameters.

        Parameters
        ----------
        beta : float
            Gauge coupling parameter.
        sites_before_link : bool, default=True
            Whether spatial lattice axes come before the link axis.
        """
        self.beta = beta
        self.sites_before_link = sites_before_link
        self._link_axis = -3 if sites_before_link else 1  # 1 for batch axis
        self._project_onto_tangent_space = anti_hermitian_traceless

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the action for a batch of gauge configurations.

        Parameters
        ----------
        x : torch.Tensor
            Tensor containing the gauge links. After one batch axis, spatial
            lattice axes come first (if sites_before_link=True), followed by
            the link direction axis, and then the link matrix components.

        Returns
        -------
        torch.Tensor
            Per-batch action values.
        """
        ndim = len(x.shape) - 4  # exclude batch & direction & matrix indices
        dim = tuple(range(1, 1 + ndim))  # sum over spatial dimensions
        plaq_sum = torch.zeros(len(x), device=x.device, dtype=x.real.dtype)
        for mu in range(1, ndim):
            for nu in range(mu):
                plaq = compute_reduced_trace(compute_wilson_1x1_loop(
                    x, mu, nu, sites_before_link=self.sites_before_link
                )).real
                plaq_sum += torch.sum(plaq, dim=dim)
        # Note: 1 / n_c factor is already included in `compute_reduced_trace`.
        return -self.beta * plaq_sum

    def force(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the gauge force: minus gradient of action w.r.t. gauge link
        variables.

        Parameters
        ----------
        x : torch.Tensor
            Tensor containing the gauge links. After one batch axis, spatial
            lattice axes come first (if sites_before_link=True), followed by
            the link direction axis, and then the link matrix components.

        Returns
        -------
        torch.Tensor
            Force on each link of the same shape of x.
        """
        # The algebra force is multiplied by links to map back to group space
        return self.algebra_force(x) @ x

    def algebra_force(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the minus gradient of the action w.r.t. algebra-valued gauge
        variables.

        Parameters
        ----------
        x : torch.Tensor
            Tensor containing the gauge links. After one batch axis, spatial
            lattice axes come first (if sites_before_link=True), followed by
            the link direction axis, and then the link matrix components.

        Returns
        -------
        torch.Tensor
            Anti-Hermitian traceless force matrices in the Lie algebra of the
            same shape of x.

        Note:
            The magnitude of this force depends on the normalization of
            the SU(N_c) generators T^a. Lattice QCD literature often uses
            Tr(T^a T^b) = -1/2 δ^ab, but this code uses Tr(T^a T^b) = -δ^ab.
        """
        n_c = x.shape[-1]

        g = compute_all_directional_staples(
            x, sites_before_link=self.sites_before_link
        )

        coeff = -self.beta / n_c
        algebra_force = coeff * self._project_onto_tangent_space(x @ g)
        return algebra_force


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


def compute_reduced_trace(x):  # reduced trace = 1/n trace()
    """Compute the reduced trace of the input matrix x."""
    return torch.mean(torch.diagonal(x, dim1=-2, dim2=-1), dim=-1)
