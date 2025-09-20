# Created by Javad Komijani, May 2025

"""Predictor-Corrector Sampler for Score-Based Generative Models"""

import functools
from typing import Callable, Tuple

import torch

from ._denoising_sdeint import integrate_denoising_sde
from ._denoising_sdeint import integrate_denoising_sde_


__all__ = ["DenoisingSDEFlow", "DenoisingSDEFlow_"]


Tensor = torch.Tensor


# =============================================================================
class DenoisingSDEFlow(torch.nn.Module):
    """
    A PyTorch module that integrates a denoising stochastic differential
    equation (SDE) using a predictor-corrector method over a time span.

    Applies a predictor-corrector scheme where the predictor evolves the
    system via ODE integration, and the corrector refines it using Langevin
    dynamics based on a score function (derivative of log-probability).
    """

    def __init__(
        self,
        drift_fn: Callable,
        score_fn: Callable,
        t_span: Tuple[float, float],
        **solver_kwargs
    ):
        """
        Initializes the DenoisingSDEFlow module.

        Args:
            drift_fn: Callable(t, y) -> Tensor
                Drift function used by the ODE predictor.
            score_fn: Callable(t, y) -> Tensor
                Score function used by the Langevin corrector.
            t_span: Tuple[float, float]
                Time interval over which to integrate.
            **solver_kwargs:
                Additional keyword arguments passed to integrate_denoising_sde
                function, such as:
                - step_size
                - num_steps
                - langevin_step_size
                - num_langevin_iters
                - ode_step
        """
        super().__init__()
        self.t_span = t_span
        self.flow = functools.partial(
            integrate_denoising_sde,
            drift_fn=drift_fn,
            score_fn=score_fn,
            **solver_kwargs
        )

    def forward(self, y: Tensor) -> Tensor:
        """
        Evolves the system from initial to final time in `t_span`.

        Args:
            y (Tensor): The initial state of the system.

        Returns:
            Tensor: The evolved system state at the final time.
        """
        return self.flow(t_span=self.t_span, y0=y)

    def reverse(self, y: Tensor) -> Tensor:
        """
        Evolves the system in reverse, from final to initial time.

        Args:
            y (Tensor): The state of the system at the final time.

        Returns:
            Tensor: The evolved system state at the initial time.
        """
        return self.flow(t_span=self.t_span[::-1], y0=y)


# =============================================================================
class DenoisingSDEFlow_(torch.nn.Module):  # pylint: disable=invalid-name
    """
    A PyTorch module that integrates a denoising stochastic differential
    equation (SDE) using a predictor-corrector method over a time span.

    Applies a predictor-corrector scheme where the predictor evolves the
    system via ODE integration, and the corrector refines it using Langevin
    dynamics based on a score function (derivative of log-probability).
    """

    def __init__(
        self,
        drift_fn: Callable,
        score_fn: Callable,
        t_span: Tuple[float, float],
        **solver_kwargs
    ):
        """
        Initializes the DenoisingSDEFlow_ module.

        Args:
            drift_fn: Callable(t, y) -> Tensor
                Drift function used by the ODE predictor.
            score_fn: Callable(t, y) -> Tensor
                Score function used by the Langevin corrector.
            t_span: Tuple[float, float]
                Time interval over which to integrate.
            **solver_kwargs:
                Additional keyword arguments passed to integrate_denoising_sde_
                function, such as:
                - step_size
                - num_steps
                - langevin_step_size
                - num_langevin_iters
                - ode_step
                - num_hutchinson_samples
        """
        super().__init__()
        self.t_span = t_span
        self.flow_ = functools.partial(
            integrate_denoising_sde_,
            drift_fn=drift_fn,
            score_fn=score_fn,
            **solver_kwargs
        )

    def forward(self, y: Tensor, logp: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Evolves the system from initial to final time in `t_span`.

        Args:
            y (Tensor): The initial state of the system.

        Returns:
            Tuple[Tensor, Tensor]: Final state tensor and log-probability.
        """
        return self.flow_(t_span=self.t_span, y0=y, logp0=logp)

    def reverse(self, y: Tensor, logp: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Evolves the system in reverse, from final to initial time.

        Args:
            y (Tensor): The state of the system at the final time.

        Returns:
            Tuple[Tensor, Tensor]: State at initial time and log-probability.
        """
        return self.flow_(t_span=self.t_span[::-1], y0=y, logp0=logp)
