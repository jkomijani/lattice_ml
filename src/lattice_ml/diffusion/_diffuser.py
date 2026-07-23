# Created by Javad Komijani, 2024-2026

"""Implements Euclidean (scalar) diffusers for :class:`DiffusionModel`."""

from typing import Callable, Tuple

import torch

from lattice_ml.integrate import odeint

from ._sde_schedule import VPScheduleWithInverseTimeGamma
from ._sde_schedule import SubVPScheduleWithInverseTimeGamma


__all__ = ["VPDiffuser", "SubVPDiffuser"]


# =============================================================================
class VPDiffuser(torch.nn.Module):
    r"""
    Implements a variance preserving diffusion process as

    .. math::
        \frac{d x(t)}{dt} = - \gamma(t) x(t) + \sigma(t) \eta(t)
        \sigma(t) = \sqrt{2 \gamma(t)}

    By default, we use :math:`\gamma(t) = 1 / (1 - t)`.
    """

    integrate = staticmethod(odeint)

    def __init__(self, sde_schedule: Callable | None = None):
        """Initializes the diffuser with an SDE schedule.

        Args:
            sde_schedule (Callable): Defines the time-dependent functions of
            the SDE. (Default is :class:`VPScheduleWithInverseTimeGamma()`.)
        """
        super().__init__()
        if sde_schedule is None:
            sde_schedule = VPScheduleWithInverseTimeGamma()
        self.sde_schedule = sde_schedule

    def forward(self, t_span: Tuple, x_0: torch.Tensor):
        """
        Simulates the forward diffusion process.

        The process starts from the initial state `x_0` at time `t_span[0]`
        and evolves the state until the terminal time `t_span[1]` by adding
        noise to the state.

        Args:
            t_span (Tuple): `(t_0, t)`, the initial and terminal times, each
                a 0d or 1d `torch.Tensor`. These may be batched per-example;
                if 1d, their length must match the batch size of `x_0`. One of
                the two (but not both) may be a plain float instead, if needed.
            x_0 (torch.Tensor): The initial state of the system at time `t_0`.

        In addition to the state at time `t`, this method computes and returns
        other useful quantities. Note that

            [x_t, u_t].T = A [signal, noise].T

        where
                |signal_scale    noise_scale |
            A = |                            |
                |-noise_scale    signal_scale|

        with `det(A) = 1`. Quantities `x_t` and `u_t` are complementary states.
        Unlike the state `x_t`, the complementary state `u_t` mainly contains
        the noise at small diffusion times and mainly the signal at later
        times. Moreover, the complementary state `u_t` is proportional to the
        conditionaly velocity of the state `x_t`.

        Returns
        -------
        torch.Tensor, torch.Tensor, torch.Tensor
            A tuple containing:
            - `x_t`: the final diffused states of the system,
            - `diffusion_context`: dictionary containing:
                - `complementary_state`: complementary component to `x_t`,
                - `noise`: noise samples used in the diffusion,
                - `noise_scale`: weight of the noise in `x_t`,
                - `signal_scale`: weight of the signal in `x_t`.
                - `half_sigma_square`: half of square of `sigma(t)`.
        """
        t_0, t = t_span

        # Expand t_eval dimensions to match x_0
        t = t.view(-1, *[1] * (x_0.ndim - 1))

        # Compute accumulated noise standard deviation and its complementary
        noise_scale = self.sde_schedule.transition_noise_std(t_0, t)
        signal_scale = self.sde_schedule.transition_mean_scale(t_0, t)

        # Sample from normal distribution
        noise = torch.randn_like(x_0)

        # Closed-form solution
        x_t = signal_scale * x_0 + noise_scale * noise
        u_t = -noise_scale * x_0 + signal_scale * noise

        half_sigma_square = self.sde_schedule.half_sigma_square(t)

        diffusion_context = {
            'complementary_state': u_t,
            'noise': noise,
            'noise_scale': noise_scale,
            'signal_scale': signal_scale,
            'half_sigma_square': half_sigma_square,
        }
        return x_t, diffusion_context

    def build_score_fn(self, dynamics_fn: Callable) -> Callable:
        """
        Build the score function from the probability flow ODE dynamics.
        """
        def score_fn(t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
            """Compute the dynamics function of the probability flow ODE."""
            t_ = t.view(-1, *[1] * (x_t.ndim - 1))
            coeff = -1 / self.sde_schedule.half_sigma_square(t_)
            return coeff * dynamics_fn(t, x_t) - x_t

        return score_fn

    def build_ode_dynamics_fn(self, score_plus_x_fn: Callable) -> Callable:
        r"""
        Build the probability flow ODE dynamics of this diffusion process.

        This solves the ODE corresponding to the forward SDE:

        .. math::
            d x(t) = -\frac{1}{2} \sigma(t)^2 x(t) dt + \sigma(t)\,dW_t,

        by instead integrating its probability flow ODE:

        .. math::
            \frac{dx}{dt} = -\frac{1}{2} \sigma(t)^2 (x + \nabla_x \log p_t(x))

        where :math:`\nabla_x \log p_t(x)` is the score function, approximated
        by `score_plus_x_fn(t, x_t) - x_t`.

        Args:
            score_plus_x_fn (Callable): Function approximating `score + x`.

        Returns:
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
                Function `f(t, x_t)` computing the ODE dynamics at `(t, x_t)`.
        """
        def dynamics_fn(t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
            """Compute the dynamics function of the probability flow ODE."""
            t_ = t.view(-1, *[1] * (x_t.ndim - 1))
            coeff = - self.sde_schedule.half_sigma_square(t_)
            return coeff * score_plus_x_fn(t, x_t)

        return dynamics_fn

    @staticmethod
    def euler_step(dynamics_fn, x_t, t, dt):
        """Perform a single Euler step."""
        return x_t + dt * dynamics_fn(t, x_t)


# =============================================================================
class SubVPDiffuser(torch.nn.Module):
    r"""
    Implements a sub variance preserving diffusion process as

    .. math::
        \frac{d x(t)}{dt} = - \gamma(t) x(t) + \sigma(t) \eta(t)
        \sigma(t) = \sqrt{2 \gamma(t) (1 - e^{-\int \gamma(s) ds})}

    By default, we use :math:`\gamma(t) = 1 / (1 - t)`.
    """

    _complementary_for_score_plus_x = False
    integrate = staticmethod(odeint)

    def __init__(self, sde_schedule: Callable | None = None):
        """Initializes the diffuser with an SDE schedule.

        Args:
            sde_schedule (Callable): Defines the time-dependent functions of
            the SDE. (Default is :class:`SubVPScheduleWithInverseTimeGamma()`.)
        """
        super().__init__()
        if sde_schedule is None:
            sde_schedule = SubVPScheduleWithInverseTimeGamma()
        self.sde_schedule = sde_schedule

    def forward(self, t_span: Tuple, x_0: torch.Tensor):
        """
        Simulates the forward diffusion process.

        The process starts from the initial state `x_0` at time `t_span[0]`
        and evolves the state until the terminal time `t_span[1]` by adding
        noise to the state.

        Args:
            t_span (Tuple): `(t_0, t)`, the initial and terminal times, each
                a 0d or 1d `torch.Tensor`. These may be batched per-example;
                if 1d, their length must match the batch size of `x_0`. One of
                the two (but not both) may be a plain float instead, if needed.
            x_0 (torch.Tensor): The initial state of the system at time `t_0`.

        In addition to the state at time `t`, this method computes and returns
        other useful quantities. Note that

            [x_t, u_t].T = A [signal, noise].T

        where
                |signal_scale    noise_scale|
            A = |                           |
                |-1              1          |

        with `det(A) = 1`. Quantities `x_t` and `u_t` are complementary states.
        Unlike the state `x_t`, the complementary state `u_t` mainly contains
        the noise at small diffusion times and mainly the signal at later
        times. Moreover, the complementary state `u_t` is proportional to the
        conditionaly velocity of the state `x_t`.

        Returns
        -------
        torch.Tensor, torch.Tensor, torch.Tensor
            A tuple containing:
            - `x_t`: the final diffused states of the system,
            - `diffusion_context`: dictionary containing:
                - `complementary_state`: complementary component to `x_t`,
                - `noise`: noise samples used in the diffusion,
                - `noise_scale`: weight of the noise in `x_t`,
                - `signal_scale`: weight of the signal in `x_t`.
        """
        t_0, t = t_span

        # Expand t_eval dimensions to match x_0
        t = t.view(-1, *[1] * (x_0.ndim - 1))

        # Compute accumulated noise standard deviation and its complementary
        noise_scale = self.sde_schedule.transition_noise_std(t_0, t)
        signal_scale = self.sde_schedule.transition_mean_scale(t_0, t)

        # Sample from normal distribution
        noise = torch.randn_like(x_0)

        # Closed-form solution
        x_t = signal_scale * x_0 + noise_scale * noise

        if self._complementary_for_score_plus_x:
            u_t = -noise_scale * x_0 + (1 + noise_scale) * noise
        else:
            u_t = noise - x_0

        half_sigma_square = self.sde_schedule.half_sigma_square(t)

        diffusion_context = {
            'complementary_state': u_t,
            'noise': noise,
            'noise_scale': noise_scale,
            'signal_scale': signal_scale,
            'half_sigma_square': half_sigma_square,
        }
        return x_t, diffusion_context

    def build_score_fn(self, dynamics_fn: Callable) -> Callable:
        """
        Build the score function from the probability flow ODE dynamics.
        """
        def score_fn(t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
            """Compute the dynamics function of the probability flow ODE."""
            t_ = t.view(-1, *[1] * (x_t.ndim - 1))
            gamma = self.sde_schedule.gamma(t_)
            coeff = -1 / self.sde_schedule.half_sigma_square(t_)
            return coeff * (dynamics_fn(t, x_t) + gamma * x_t)

        return score_fn

    def build_ode_dynamics_fn(self, score_plus_x_fn: Callable) -> Callable:
        """
        Build the probability flow ODE dynamics of this diffusion process.

        See `VPDiffuser.dynamics_fn` for the general idea; the drift here
        additionally includes the `-x_t` term of the sub-VP schedule.

        Args:
            score_plus_x_fn (Callable): Function approximating `score + x`.

        Returns:
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
                Function `f(t, x_t)` computing the ODE dynamics at `(t, x_t)`.
        """
        def dynamics_fn(t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
            """Compute the dynamics function of the probability flow ODE."""
            t_ = t.view(-1, *[1] * (x_t.ndim - 1))
            coeff = - self.sde_schedule.half_sigma_square(t_)
            return -x_t + coeff * score_plus_x_fn(t, x_t)

        return dynamics_fn

    @staticmethod
    def euler_step(dynamics_fn, x_t, t, dt):
        """Perform a single Euler step."""
        return x_t + dt * dynamics_fn(t, x_t)
