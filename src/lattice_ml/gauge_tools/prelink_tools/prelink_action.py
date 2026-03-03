# Copyright (c) 2026 Javad Komijani

"""
Wilson gauge action and force calculations for lattice gauge theory.
"""

import torch

from .sealed_prelinks import compute_sealed_staples
from .sealed_prelinks import compute_sealed_prelinks


__all__ = ['WilsonPrelinkAction']


class WilsonPrelinkAction:
    r"""
    Wilson gauge-prelink action and force calculations for lattice gauge theory.

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
        self._project_onto_algebra_space = anti_hermitian_traceless

    def __call__(self, V: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the action for a batch of gauge prelink configurations.

        Parameters
        ----------
        V : torch.Tensor
            Tensor containing the gauge prelinks. After one batch axis, spatial
            lattice axes come first (if sites_before_link=True), followed by
            the prelink direction axis, and then the prelink matrix components.

        Returns
        -------
        torch.Tensor
            Per-batch action values.
        """
        bsize = V.shape[0]
        S = compute_sealed_staples(V, sites_before_link=self.sites_before_link)

        trace = compute_normalized_trace(S).real.reshape(bsize, -1).sum(dim=1)
        # Notes:
        # 1) 1/n_c factor is already included in compute_normalized_trace
        # 2) Each plaquette is counted four times in the trace
        return (-self.beta / 4) * trace

    def force(self, V: torch.Tensor) -> torch.Tensor:
        """
        Compute the gauge force: minus gradient of action w.r.t. gauge prelink
        variables.

        Parameters
        ----------
        V : torch.Tensor
            Tensor containing the gauge prelinks. After one batch axis, spatial
            lattice axes come first (if sites_before_link=True), followed by
            the prelink direction axis, and then the prelink matrix components.

        Returns
        -------
        torch.Tensor
            Force on each prelink of the same shape of V.
        """
        # The algebra force is multiplied by prelinks to map to group space
        return self.algebra_force(V) @ V 

    def algebra_force(self, V: torch.Tensor) -> torch.Tensor:
        """
        Compute the minus gradient of the action w.r.t. algebra-valued gauge
        variables.

        Parameters
        ----------
        V : torch.Tensor
            Tensor containing the gauge prelinks. After one batch axis, spatial
            lattice axes come first (if sites_before_link=True), followed by
            the prelink direction axis, and then the prelink matrix components.

        Returns
        -------
        torch.Tensor
            Anti-Hermitian traceless force matrices in the Lie algebra of the
            same shape of V.

        Note:
            The magnitude of this force depends on the normalization of
            the SU(N_c) generators T^a. Lattice QCD literature often uses
            Tr(T^a T^b) = -1/2 δ^ab, but this code uses Tr(T^a T^b) = -δ^ab.
        """
        n_c = V.shape[-1]

        G = compute_sealed_prelinks(
            V, sites_before_link=self.sites_before_link, sum_over_staples=True
        )

        coeff = self.beta / n_c  # (-1 from action) x (-1 for sealed_prelinks)
        algebra_force = coeff * self._project_onto_algebra_space(G)
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

    # Remove trace
    trace = torch.einsum("...ii->...", x)[..., None, None]
    n = x.shape[-1]
    eye = torch.eye(n, device=x.device, dtype=x.dtype)

    return x - (trace / n) * eye


def compute_normalized_trace(x):
    """Compute the normalized trace (trace / n) of the input matrix x."""
    return torch.einsum('...ii->...', x) / x.shape[-1]
