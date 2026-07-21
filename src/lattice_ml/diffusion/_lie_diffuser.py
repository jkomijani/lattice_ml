# Created by Javad Komijani, 2024-2026

"""Implements diffusion models."""

# pylint: disable=too-many-arguments, too-many-positional-arguments

from typing import Callable, Tuple
from abc import ABC, abstractmethod

import torch

from lattice_ml.integrate import lie_odeint

from ._sde_schedule import VEScheduleWithInverseTimeSigmaSquared
from .gauge._randn_xxx_like import randn_traceless_antihermitian_like
from .gauge._lie_sdeint import integrate_sde as _lie_sde_integrate


__all__ = ["LieDiffuser", "SUnDiffuser"]


# =============================================================================
class LieDiffuser(torch.nn.Module, ABC):
    r"""
    Implements a Brownian-motion-on-the-group diffusion process:

    .. math::
        dU(t) = \sigma(t) dW_t U(t)

    where `dW_t` is white noise valued in the Lie algebra.

    By default, the noise scale sigma(t) follows an inverse-time variance law,
    implemented by :class:`VEScheduleWithInverseTimeSigmaSquared`.
    The contraction toward the group's Haar measure emerges from the stochastic
    evolution itself. See: "Noise scheduling and linear dynamics in diffusion
    models on Lie groups," arXiv:2605.17326 (2026).

    This is an abstract class: it is generic over the choice of Lie group, and
    a concrete subclass must define `randn_algebra_like`, sampling Gaussian
    noise valued in that group's Lie algebra with the same shape as its input
    (see `SUnDiffuser` for the SU(N) case).

    Note: Throughout this class `score_fn` and `dynamics_fn` always refer
        to algebra-valued functions.
    """

    def __init__(
        self,
        sde_schedule: Callable | None = None,
        n_random_walk_steps: int = 4,
    ):
        """Initializes the diffuser with an SDE schedule.

        Args:
            sde_schedule (Callable): Defines the time-dependent functions of
                the SDE (default is `VEScheduleWithInverseTimeSigmaSquared()`).
            n_random_walk_steps (int): Number of multiplicative sub-steps
                used to build the heat-kernel-like increment. (Default is 4.)
        """
        super().__init__()
        if sde_schedule is None:
            sde_schedule = VEScheduleWithInverseTimeSigmaSquared()
        self.sde_schedule = sde_schedule
        self.n_random_walk_steps = n_random_walk_steps

    @abstractmethod
    def randn_algebra_like(self, x: torch.Tensor):
        """Sample algebra-valued Gaussian noise with the same shape as `x`."""

    def forward(self, x_0: torch.Tensor, t_0: torch.Tensor, t: torch.Tensor):
        """
        Simulates the forward diffusion process on the group.

        The process starts from the initial group-valued state `x_0` at
        time `t_0` and evolves it to time `t` via a discretized random walk
        in the Lie algebra, mapped to the group through the matrix exponential
        at each sub-step.

        Args:
            x_0 (torch.Tensor): The initial group-valued state at time `t_0`.
            t_0 (torch.Tensor): A 0d (or float) of the initial time.
            t (torch.Tensor): A 0d or 1d tensor of the terminal times. If 1d,
                its length must match the batch size of `x_0`.

        Returns:
            A tuple containing:
            - `x_t` (torch.Tensor): the final diffused (group-valued) states.
            - `diffusion_context`: dictionary containing:
                - `complementary_state` / `noise`: the (identical)
                    algebra-valued conjugate-velocity quantity -- see Note.
                - `noise_scale`: the cumulative noise std over `[t_0, t]`.
                - `signal_scale`: always 1 (no signal decay; VE scheme).
                - `half_sigma_square`: half the square of `sigma(t)`.

        Note:
            Both `complementary_state` and `noise` hold the same value:
            the sum of all algebra-valued noise increments, normalized by its
            cumulative std. This is the Lie-algebra analogue of `VPDiffuser`'s
            `u_t` (proportional to the conditional velocity of `x_t`), and the
            two keys coincide exactly here because this schedule is driftless
            (`gamma = 0`, Variance Exploding).
        """
        assert (t >= t_0).all(), "`t` must be >= `t_0`."

        # Expand t_eval dimensions to match x_0
        t = t.view(-1, *[1] * (x_0.ndim - 1))

        # Time step for discretized diffusion
        h = (t - t_0) / self.n_random_walk_steps

        transition_noise_std = self.sde_schedule.transition_noise_std

        cum_randn_alg = 0
        x_t = x_0

        # Loop over discretized random walk in the Lie algebra
        for m in range(self.n_random_walk_steps):
            std = transition_noise_std(t_0 + h * m, t_0 + h * (m + 1))
            randn_alg = std * self.randn_algebra_like(x_t)
            x_t = torch.matrix_exp(randn_alg) @ x_t
            cum_randn_alg = cum_randn_alg + randn_alg

        noise_scale = transition_noise_std(t_0, t)

        u_t = cum_randn_alg / noise_scale

        signal_scale = 1  # analogous to the VE scheme for Euclidean scalar
        half_sigma_square = self.sde_schedule.half_sigma_square(t)

        diffusion_context = {
            'complementary_state': u_t,
            'noise': u_t,
            'noise_scale': noise_scale,
            'signal_scale': signal_scale,
            'half_sigma_square': half_sigma_square,
        }
        return x_t, diffusion_context

    def build_score_fn(self, dynamics_fn: Callable) -> Callable:
        """
        Build the (algebra-valued) score function from the probability flow
        ODE dynamics.
        """
        def score_fn(t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
            t_ = t.view(-1, *[1] * (x_t.ndim - 1))
            half_sigma_square = self.sde_schedule.half_sigma_square(t_)
            return -dynamics_fn(t, x_t) / half_sigma_square

        return score_fn

    def build_ode_dynamics_fn(self, score_fn: Callable) -> Callable:
        """
        Build the algebra-valued probability flow ODE dynamics of this
        diffusion process, for use with `integrate` (see class docstring's
        note on the algebra-valued vs. group-valued `dynamics_fn`).
        """
        def dynamics_fn(t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
            t_ = t.view(-1, *[1] * (x_t.ndim - 1))
            half_sigma_square = self.sde_schedule.half_sigma_square(t_)
            return -half_sigma_square * score_fn(t, x_t)

        return dynamics_fn

    def integrate(
        self,
        dynamics_fn: Callable,
        t_span: Tuple[float, float],
        x_0: torch.Tensor,
        method: str = 'Euler:g',
        **kwargs,
    ) -> torch.Tensor:
        """
        Integrate algebra-valued ODE dynamics on the group.

        Delegates to `lattice_ml.integrate.lie_odeint`, which by default
        advances the state via the matrix exponential at each step as:

            torch.matrix_exp(dynamics_fn(t, x_t) * dt) @ x_t

        Args:
            dynamics_fn (Callable): Algebra-valued ODE dynamics.
            t_span (Tuple[float, float]): `(t0, t1)`, the integration interval.
            x_0 (torch.Tensor): Initial group-valued state.
            method (str): The integration method (default is 'Euler:g').
            **kwargs: Additional keyword arguments forwarded to `lie_odeint`,
                e.g. `step_size`, `num_steps`, `t_eval`.

        Returns:
            torch.Tensor: Final (or evaluated) group-valued state(s).
        """
        kwargs['method'] = method
        return lie_odeint(dynamics_fn, t_span, x_0, **kwargs)

    @staticmethod
    def euler_step(dynamics_fn, x_t, t, dt):
        """Perform a single Euler step on the group."""
        return torch.matrix_exp(dynamics_fn(t, x_t) * dt) @ x_t

    def integrate_sde(
        self,
        dynamics_fn: Callable,
        t_span: Tuple[float, float],
        x_0: torch.Tensor,
        noise_ratio: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Integrate the reverse-time SDE on the group (stochastic sampling),
        as opposed to `integrate`'s deterministic probability-flow ODE.

        Args:
            dynamics_fn (Callable): Algebra-valued ODE dynamics.
            t_span (Tuple[float, float]): `(t0, t1)`, the integration interval.
            x_0 (torch.Tensor): Initial group-valued state.
            noise_ratio (float): Ratio of reverse to forward noise, controlling
                the stochasticity of the sampler. Default is 1.
            **kwargs: Additional keyword arguments forwarded to
                to `_lie_sdeint.integrate_sde`, e.g. `step_size`, `num_steps`.

        Returns:
            torch.Tensor: Final group-valued state.
        """
        def scaled_dynamics_fn(t, x_t):
            return (1 + noise_ratio ** 2) * dynamics_fn(t, x_t)

        def noise_scale(t):
            return noise_ratio * self.sde_schedule.sigma(t)

        def sde_step(func, t, y, dt, noise_scale_value):
            drift = dt * func(t, y)
            scale = abs(dt) ** 0.5 * noise_scale_value
            diffusion = scale * self.randn_algebra_like(y)
            return torch.matrix_exp(drift + diffusion) @ y

        return _lie_sde_integrate(
            scaled_dynamics_fn, t_span, x_0,
            noise_scale=noise_scale, sde_step=sde_step, **kwargs,
        )


# =============================================================================
class SUnDiffuser(LieDiffuser):
    """
    Concretizes `LieDiffuser` for SU(N): algebra-valued noise is sampled as
    traceless anti-Hermitian matrices (the su(n) Lie algebra), the same
    noise generator used by `SUnDiffusionProcess`.
    """

    def randn_algebra_like(self, x: torch.Tensor):
        return randn_traceless_antihermitian_like(x)
