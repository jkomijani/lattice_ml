# Copyright (c) 2026 Javad Komijani

r"""Wilson gauge action and HMC force in the holonomy parametrization."""

# pylint: disable=invalid-name

import math
import torch

from .holonomy import compute_h_munu
from .holonomy_clover import compute_holonomy_clover


__all__ = ['WilsonHolonomyAction']


class WilsonHolonomyAction:
    r"""Wilson gauge action and HMC force in the holonomy parametrization.

    The action is

        S[h] = (beta / N_c / 2) sum_{x, mu != nu} Re Tr[I - P_{mu,nu}(x)],
             = (beta / N_c) sum_{x, mu > nu} Re Tr[I - P_{mu,nu}(x)],

    where P_{mu,nu}(x) = h†(x) h(x+mu) h†(x+mu+nu) h(x+nu) is the plaquette
    expressed in terms of h_{mu,nu}(x) = h_{mu,0}(x) h_{nu,0}(x)†.

    The HMC force on h_{mu,0}(x) is

        F_{mu,0}(x) = (beta / N_c) Pi_su[G_{mu,0}(x)],

    where Pi_su projects onto the traceless anti-Hermitian algebra and
    G is the holonomy clover sum; see :func:`compute_holonomy_clover`.

    Parameters
    ----------
    beta : float
        Inverse gauge coupling.
    sites_before_link : bool, default=True
        Whether spatial lattice axes precede the link axis.
    """

    def __init__(
        self,
        beta: float,
        sites_before_link: bool = True,
    ):
        self.beta = beta
        self.sites_before_link = sites_before_link
        self._project_onto_algebra_space = anti_hermitian_traceless

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        """Evaluate the action for a batch of holonomy configurations.

        Parameters
        ----------
        h : torch.Tensor
            Holonomy h_{mu,0} for mu = 1, ..., d-1. Shape:
            ``(batch, N_0+1, ..., N_{d-1}+1, d-1, Nc, Nc)``
            if ``sites_before_link=True``.

        Returns
        -------
        torch.Tensor
            Per-batch action values, shape ``(batch,)``.
        """
        n_c = h.shape[-1]
        prefix_dims = 1
        spatial_ndim = h.ndim - prefix_dims - 3  # -3 for direction & matrix

        shape = _calc_physical_shape(h, prefix_dims, self.sites_before_link)
        num_plaq = math.prod(shape) * spatial_ndim * (spatial_ndim - 1) / 2

        total = h.new_zeros(h.shape[0], dtype=h.real.dtype)

        kwargs = {
            'prefix_dims': prefix_dims,
            'sites_before_link': self.sites_before_link
        }

        for mu in range(1, spatial_ndim):
            for nu in range(mu):
                h_munu = compute_h_munu(h, mu, nu, **kwargs)
                dim_mu = prefix_dims + mu
                dim_nu = prefix_dims + nu
                total = total + _planar_action_sum(h_munu, dim_mu, dim_nu)

        return self.beta * (num_plaq - total / n_c)

    def force(self, h: torch.Tensor) -> torch.Tensor:
        """Compute the group-valued HMC force: algebra_force(h) @ h.

        Parameters
        ----------
        h : torch.Tensor
            Holonomy h_{mu,0} (see :meth:`__call__`).

        Returns
        -------
        torch.Tensor
            Force on each holonomy, same shape as ``h``.
        """
        return self.algebra_force(h) @ h

    def algebra_force(self, h: torch.Tensor) -> torch.Tensor:
        """Compute the algebra-valued force.

        Parameters
        ----------
        h : torch.Tensor
            Holonomy h_{mu,0} (see :meth:`__call__`).

        Returns
        -------
        torch.Tensor
            Anti-Hermitian traceless force matrices, same shape as ``h``.
        """
        n_c = h.shape[-1]
        kws = {'prefix_dims': 1, 'sites_before_link': self.sites_before_link}

        G = compute_holonomy_clover(h, **kws)

        coeff = self.beta / n_c  # (-1 from action) x (-1 from h†(x) in P)
        algebra_force = coeff * self._project_onto_algebra_space(G)
        return algebra_force


def _calc_physical_shape(h, prefix_dims=1, sites_before_link=True):
    if sites_before_link:
        extended_shape = h.shape[prefix_dims:-3]
    else:
        extended_shape = h.shape[prefix_dims + 1:-2]
    physical_shape = [n - 1 for n in extended_shape]
    return physical_shape


def _planar_action_sum(h_munu: torch.Tensor, dim_mu: int, dim_nu: int):
    """
    Sum Re Tr[P_{mu,nu}(x)] over all sites x, returning a per-batch tensor.

    P(x) = h†(x) h(x+mu) h†(x+mu+nu) h(x+nu).
    """
    N_mu = h_munu.shape[dim_mu] - 1
    N_nu = h_munu.shape[dim_nu] - 1

    def crop(i_mu, i_nu):
        return h_munu.narrow(dim_mu, i_mu, N_mu).narrow(dim_nu, i_nu, N_nu)

    P = crop(0, 0).adjoint() @ crop(1, 0) @ crop(1, 1).adjoint() @ crop(0, 1)

    trace = torch.einsum('...ii->...', P).real
    return trace.reshape(trace.shape[0], -1).sum(dim=1)


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
