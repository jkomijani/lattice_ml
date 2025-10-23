# Copyright (c) 2025 Javad Komijani

"""
Wilson gauge action and force calculations for lattice gauge theory.
"""

import torch

from .wilson_loops_u1 import compute_u1_wilson_1x1_loop
from .wilson_staples_u1 import compute_all_u1_directional_staples


__all__ = ['WilsonU1GaugeAction']


class WilsonU1GaugeAction:
    r"""
    Wilson gauge action and force calculations for lattice gauge theory.

    Implements the Wilson gauge action for U(1) gauge group, together with
    the corresponding gauge force used in HMC simulations.

    The action is defined as:

    .. math::
        S = - \beta \sum_{\nu \neq \mu} \text{Plaq}_{mu, nu} / 2 ,
          = - \beta \sum_{\nu < \mu} Re \text{Plaq}_{mu, nu} .

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

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the action for a batch of gauge configurations.

        Parameters
        ----------
        x : torch.Tensor
            Tensor containing the gauge links. After one batch axis, spatial
            lattice axes come first (if sites_before_link=True), followed by
            the link direction axis.

        Returns
        -------
        torch.Tensor
            Per-batch action values.
        """
        # Determine the number of spatial dimensions
        spatial_ndim = x.ndim - 2  # exclude batch, direction
        sum_dims = tuple(range(1, 1 + spatial_ndim))  # sum over spatial dims

        plaq_sum = torch.zeros(len(x), device=x.device, dtype=x.real.dtype)

        for mu in range(1, spatial_ndim):
            for nu in range(mu):
                plaq = torch.real(compute_u1_wilson_1x1_loop(
                    x, mu, nu, sites_before_link=self.sites_before_link
                ))
                plaq_sum += torch.sum(plaq, dim=sum_dims)

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
            the link direction axis.

        Returns
        -------
        torch.Tensor
            Force on each link of the same shape of x.
        """
        # The algebra force is multiplied by links to map back to group space
        return self.algebra_force(x) * x

    def algebra_force(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the minus gradient of the action w.r.t. algebra-valued gauge
        variables.

        Parameters
        ----------
        x : torch.Tensor
            Tensor containing the gauge links. After one batch axis, spatial
            lattice axes come first (if sites_before_link=True), followed by
            the link direction axis.

        Returns
        -------
        torch.Tensor
            Pure imaginary force in the Lie algebra of the same shape of x.
        """
        g = compute_all_u1_directional_staples(
            x, sites_before_link=self.sites_before_link
        )

        algebra_force = (-self.beta * 1j) * torch.imag(x * g)
        return algebra_force
