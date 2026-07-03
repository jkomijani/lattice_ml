# Created by Javad Komijani, 2024-2026

"""Implements diffusion models."""

# pylint: disable=too-many-arguments, too-many-positional-arguments

from typing import Callable, Dict, Sequence

import numpy as np
import pydantic
import torch

from lattice_ml.integrate import odeint

from ._trainer import Trainer
from ._sde_schedule import VPScheduleWithInverseTimeGamma
from ._sde_schedule import SubVPScheduleWithInverseTimeGamma


__all__ = ["DiffusionModel", "VPDiffuser", "SubVPDiffuser"]


# =============================================================================
class DiffusionModel(torch.nn.Module):
    """
    Implements a diffusion model.

    Use Cases:
    - Simulate noisy trajectories (`forward`).
    - Train score-based generative models (`training_step`).
    - Generate samples by reversing the diffusion (`reverse`).
    """

    def __init__(
        self,
        network_fn: Callable,
        diffuser: Callable | None = None,
        as_score_plus_x: bool = True,
        training_config: Dict | None = None,
    ):
        """
        Initializes the diffusion process with a network function.

        Args:
            network_fn (Callable): A neural network whose output is interpreted
                according to `as_score_plus_x`.
            diffuser (Callable | None): Defines the diffusion process. If not
                provided, defaults to the default instance of `VPDiffuser`.
            as_score_plus_x (bool): If True (False), `network_fn` is treated as
                predicting `score + x` (`score`). This reparameterization keeps
                the model stable even when the diffuser's schedule diverges,
                e.g., `gamma(t) = 1 / (1 - t)`. Default is True.
            training_config (Dict | None): Optional dict passed to
                :class:`TrainingConfiguration`.
        """
        if diffuser is None:
            diffuser = VPDiffuser()

        super().__init__()
        self.network_fn = network_fn
        self.diffuser = diffuser
        self.as_score_plus_x = as_score_plus_x
        self.trainer = Trainer(self)
        self.training_config = TrainingConfiguration(**(training_config or {}))

    @property
    def score_fn(self):
        """The raw score function, regardless of `as_score_plus_x`."""
        if self.as_score_plus_x:
            def score_fn(t, x_t):
                return self.network_fn(t, x_t) - x_t
        else:
            score_fn = self.network_fn
        return score_fn

    @property
    def score_plus_x_fn(self):
        """The `score + x` function, regardless of `as_score_plus_x`."""
        if self.as_score_plus_x:
            score_plus_x_fn = self.network_fn
        else:
            def score_plus_x_fn(t, x_t):
                return self.network_fn(t, x_t) + x_t
        return score_plus_x_fn

    def training_step(self, batch, batch_idx=None):
        """Perform a training step to be used by Trainer."""
        x_0, = batch
        bsize = x_0.shape[0]

        # Choose a random diffusion time per sample, uniformly in [0, 1].
        t = torch.rand((bsize,), device=x_0.device)

        # Run the process to time t & get the context of the diffusion
        x_t, diffusion_context = self.diffuser(x_0, t_0=0, t=t)

        # Compute loss: implicit score matching
        if self.as_score_plus_x:
            score_plus_x = self.network_fn(t, x_t)
            loss = implicit_score_plus_x_matching(
                score_plus_x, diffusion_context
            )
        else:
            score = self.network_fn(t, x_t)
            loss = implicit_score_matching_with_variance_weight(
                score,
                diffusion_context['noise'],
                diffusion_context['noise_scale']
            )

        # Contribution from t = 0 if loss_c0 > 0
        if self.training_config.loss_c0 > 0:
            idx = (slice(None) if self.training_config.all_samples_c0
                   else np.random.randint(0, len(x_0), size=1))
            score0 = self.score_fn(0 * t[idx], x_0[idx])
            force0 = self.training_config.force0_fn(x_0[idx])
            res = score0 - force0
            loss0 = torch.mean(res * res.conj()).real
            loss = loss + self.training_config.loss_c0 * loss0

        return loss

    def forward(
        self,
        x_0: torch.Tensor,
        t_0: float = 0.,
        t_eval: float | Sequence[float] | torch.Tensor = 1.0
    ):
        """
        Simulate the forward diffusion process from an initial state.

        The system is evolved sequentially from the initial time `t_0` to one
        or multiple evaluation times `t_eval`. For each evaluation time, the
        underlying diffusion operator `self.diffuser` is called to propagate
        the state from the current state to the next one.

        Args:
            x_0 (torch.Tensor): The initial state of the system at time `t_0`.
            t_0 (float): The initial time for the simulation. Default is 0.
            t_eval (float | Sequence[float] | torch.Tensor): Target evaluation
               time(s). Times must be monotonically non-decreasing.

        Returns:
            torch.Tensor | List[torch.Tensor]:
            - If `t_eval` is a scalar, returns the state `x_t` at that time.
            - If `t_eval` contains multiple times, returns a list of states
              evaluated at each time in `t_eval`.
        """
        # Convert evaluation times to tensor
        if not isinstance(t_eval, torch.Tensor):
            t_eval = torch.as_tensor(t_eval, device=x_0.device)

        if t_eval.ndim == 0:
            t_eval = t_eval.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        x_eval = [None] * len(t_eval)

        for ind, t in enumerate(t_eval):
            assert t >= t_0, "`t_eval` must monotonically increase."

            # Run the process to time t
            x_eval[ind] = self.diffuser(x_0, t_0=t_0, t=t)[0]

            # Update the state for the next round
            x_0, t_0 = x_eval[ind], t

        return x_eval[0] if squeeze_output else x_eval

    def reverse(
        self,
        x_0: torch.Tensor,
        t_0: float = 1.0,
        t_eval: float | Sequence[float] | torch.Tensor = 0.,
        method: str = 'Euler',
        step_size: float = 0.01,
        **solver_kwargs
    ):
        """Integrate the reverse-time ODE to generate samples.

        Starts from `x_0` at time `t_0` (typically noise) and evolves toward
        smaller times using the learned score function.

        Args:
            x_0: Initial state at time `t_0`.
            t_0: Initial time.
            t_eval: Target time(s) (< t_0).
            method: ODE solver ("Euler" or "RK4").
            step_size: Solver step size.
            **solver_kwargs: Additional arguments for `odeint`.

        Returns:
            Final state or states at `t_eval`.
        """
        # Convert to tensor (on correct device)
        t_eval = torch.as_tensor(t_eval, device=x_0.device)

        # Determine integration interval
        t_end = t_eval if t_eval.ndim == 0 else t_eval.min()
        t_span = (t_0, t_end)

        # Pass solver options
        kwargs = {**solver_kwargs, "method": method, "step_size": step_size}

        # Only pass t_eval if it's not scalar
        if t_eval.ndim > 0:
            kwargs["t_eval"] = t_eval

        return self.diffuser.odeint(
            self.score_plus_x_fn, t_span, x_0, **kwargs
        )


# =============================================================================
class VPDiffuser(torch.nn.Module):
    r"""
    Implements a variance preserving diffusion process as

    .. math::
        \frac{d x(t)}{dt} = - \gamma(t) x(t) + \sigma(t) \eta(t)
        \sigma(t) = \sqrt{2 \gamma(t)}

    By default, we use :math:`\gamma(t) = 1 / (1 - t)`.
    """

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

    def forward(self, x_0: torch.Tensor, t_0: torch.Tensor, t: torch.Tensor):
        """
        Simulates the forward diffusion process.

        The process starts from the initial state `x_0` at time `t_0` and
        evolves the states until the terminal time `t` by adding noise to the
        state.

        Args:
            x_0 (torch.Tensor): The initial state of the system at time `t_0`.
            t_0 (torch.Tensor): A 0d or 1d tensor of the initial times.
            t (torch.Tensor): A 0d or 1d tensor of the terminal times.

        Note:
            At least one of `t_0` or `t` must be an instance of `torch.Tensor`.
            If a 1d tensor, their lengths must match the batch size of `x_0`.

        In addition to the state at time `t`, this method computes and returns
        other useful quantities. Note that

            [x_t, z_t].T = A  [signal, noise].T

        where
                |signal_scale    noise_scale |
            A = |                            |
                |-noise_scale    signal_scale|

        with `det(A) = 1`. Quantities `x_t` and `z_t` are complementary states.
        Unlike the state `x_t`, the complementary state `z_t` mainly contains
        the noise at small diffusion times and mainly the signal at later
        times.

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
        # Expand t_eval dimensions to match x_0
        t = t.view(-1, *[1] * (x_0.ndim - 1))

        # Compute accumulated noise standard deviation and its complementary
        noise_scale = self.sde_schedule.transition_noise_std(t_0, t)
        signal_scale = self.sde_schedule.transition_mean_scale(t_0, t)

        # Sample from normal distribution
        noise = torch.randn_like(x_0)

        # Closed-form solution
        x_t = signal_scale * x_0 + noise_scale * noise
        z_t = -noise_scale * x_0 + signal_scale * noise

        diffusion_context = {
            'complementary_state': z_t,
            'noise': noise,
            'noise_scale': noise_scale,
            'signal_scale': signal_scale
        }
        return x_t, diffusion_context

    def odeint(
        self,
        score_plus_x_fn: Callable,
        t_span: Sequence[float],
        x_0: torch.Tensor,
        **solver_kwargs
    ):
        r"""
        Integrate the ODE associated with the diffusion process.

        This method solves the ODE corresponding to the forward SDE:

        .. math::
            d x(t) = -\frac{1}{2} \sigma(t)^2 x(t) dt + \sigma(t)\,dW_t,

        by instead integrating its probability flow ODE:

        .. math::
            \frac{dx}{dt} = -\frac{1}{2} \sigma(t)^2 (x + \nabla_x \log p_t(x))

        where :math:`\nabla_x \log p_t(x)` is the score function, approximated
        by `score_plus_x_fn(t, x_t) - x_t`.

        Args:
            score_plus_x_fn (Callable): Function approximating `score + x`.
            t_span (Sequence[float, float]): Integration interval.
            x_0 (torch.Tensor): Initial state.

            **solver_kwargs:
                Additional arguments passed to :func:`odeint`, including:
                - `step_size` or `num_steps`: controls discretization
                - `method`: e.g. "RK4" or "Euler"
                - `t_eval`: intermediate evaluation times
                - `corrector`: predictor–corrector refinement step

        Returns:
             torch.Tensor: The final state of the system.
              (see :func:`odeint` for exact behavior).
        """
        def drift_fn(t, x_t):
            """Compute the drift fucntion for the ODE form of the process."""
            coeff = -0.5 * self.sde_schedule.sigma_square(t)
            return coeff * score_plus_x_fn(t, x_t)

        return odeint(drift_fn, t_span, x_0, **solver_kwargs)


# =============================================================================
class SubVPDiffuser(torch.nn.Module):
    r"""
    Implements a sub variance preserving diffusion process as

    .. math::
        \frac{d x(t)}{dt} = - \gamma(t) x(t) + \sigma(t) \eta(t)
        \sigma(t) = \sqrt{2 \gamma(t) (1 - e^{-\int \gamma(s) ds})}

    By default, we use :math:`\gamma(t) = 1 / (1 - t)`.
    """

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

    def forward(self, x_0: torch.Tensor, t_0: torch.Tensor, t: torch.Tensor):
        """
        Simulates the forward diffusion process.

        The process starts from the initial state `x_0` at time `t_0` and
        evolves the states until the terminal time `t` by adding noise to the
        state.

        Args:
            x_0 (torch.Tensor): The initial state of the system at time `t_0`.
            t_0 (torch.Tensor): A 0d or 1d tensor of the initial times.
            t (torch.Tensor): A 0d or 1d tensor of the terminal times.

        Note:
            At least one of `t_0` or `t` must be an instance of `torch.Tensor`.
            If a 1d tensor, their lengths must match the batch size of `x_0`.

        In addition to the state at time `t`, this method computes and returns
        other useful quantities. Note that

            [x_t, z_t].T = A  [signal, noise].T

        where
                |signal_scale    noise_scale    |
            A = |                               |
                |-noise_scale    1 + noise_scale|

        with `det(A) = 1`. Quantities `x_t` and `z_t` are complementary states.
        Unlike the state `x_t`, the complementary state `z_t` mainly contains
        the noise at small diffusion times and mainly the signal at later
        times.

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
        # Expand t_eval dimensions to match x_0
        t = t.view(-1, *[1] * (x_0.ndim - 1))

        # Compute accumulated noise standard deviation and its complementary
        noise_scale = self.sde_schedule.transition_noise_std(t_0, t)
        signal_scale = self.sde_schedule.transition_mean_scale(t_0, t)

        # Sample from normal distribution
        noise = torch.randn_like(x_0)

        # Closed-form solution
        x_t = signal_scale * x_0 + noise_scale * noise
        z_t = -noise_scale * x_0 + (1 + noise_scale) * noise

        diffusion_context = {
            'complementary_state': z_t,
            'noise': noise,
            'noise_scale': noise_scale,
            'signal_scale': signal_scale
        }
        return x_t, diffusion_context

    def odeint(
        self,
        score_plus_x_fn: Callable,
        t_span: Sequence[float],
        x_0: torch.Tensor,
        **solver_kwargs
    ):
        """Integrate the ODE associated with the diffusion process."""

        def drift_fn(t, x_t):
            """Compute the drift fucntion for the ODE form of the process."""
            coeff = -0.5 * self.sde_schedule.sigma_square(t)
            return -x_t + coeff * score_plus_x_fn(t, x_t)

        return odeint(drift_fn, t_span, x_0, **solver_kwargs)


# =============================================================================
class TrainingConfiguration(pydantic.BaseModel):
    """Training configuration for :class:`DiffusionModel`."""

    loss_c0: float = 0
    force0_fn: Callable | None = None
    variance_weight_for_time: bool = True
    all_samples_c0: bool = False  # use a single random sample

    def update(self, **kwargs):
        """Update the attributes."""
        for key, value in kwargs.items():
            setattr(self, key, value)


# =============================================================================
def implicit_score_plus_x_matching(
    score_plus_x: torch.Tensor,
    diffusion_context: dict
) -> torch.Tensor:
    """Compute the implicit score matching loss applied on score_plus_x.

    This computes a weighted mean squared error (MSE) between the predicted and
    the empirical conditional score. The MSE is weighted by the variance of the
    accumulated noise relative to the remaining signal scale.

    Mathematically:
        res = complementary_state + score_plus_x * (noise_scale / signal_scale)
        loss = mean(|res|^2)

    Args:
        score_plus_x (torch.Tensor): Predicted score plus the state `x_t`.
        diffusion_context (dict): Dictionary containing quantities from
            the forward diffusion step:
            - complementary_state (torch.Tensor): Complementary to `x_t`.
            - noise_scale (torch.Tensor): Scale of the cumulative noise.
            - signal_scale (torch.Tensor): Scale of the signal in `x_t`.
            - Optional keys like `noise` can also be included.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    z_t = diffusion_context['complementary_state']
    noise_scale = diffusion_context['noise_scale']
    signal_scale = diffusion_context['signal_scale']

    res = z_t + score_plus_x * (noise_scale / signal_scale)
    loss = torch.mean(res * res.conj()).real

    return loss


# =============================================================================
def implicit_score_matching_with_variance_weight(
    score: torch.Tensor,
    eps: torch.Tensor,
    noise_std: torch.Tensor
) -> torch.Tensor:
    """Compute the implicit score matching loss.

    This computes a weighted mean squared error (MSE) between the predicted and
    the empirical conditional score at a diffusion time. The MSE is weighted by
    the effective (cumulative) noise variance at the diffusion time.

    Args:
        score (torch.Tensor): Predicted score, shape (batch_size, ...).
        eps (torch.Tensor): Gaussian noise scaled by the standard deviation.
        noise_std (torch.Tensor): Standard deviation of the cumulative noise.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Residual between predicted and conditional scores (variance-weighted)
    res = score * noise_std + eps
    loss = torch.mean(res * res.conj()).real

    return loss


# =============================================================================
def implicit_score_matching_with_sdev_weight(
    score: torch.Tensor,
    eps: torch.Tensor,
    noise_std: torch.Tensor
) -> torch.Tensor:
    """Compute the implicit score matching loss.

    This computes a weighted mean squared error (MSE) between the predicted and
    the empirical conditional score at a diffusion time. The MSE is weighted by
    the effective (cumulative) noise standard deviation at the diffusion time.
    The pure noise contribution is excluded from the loss.

    Args:
        score (torch.Tensor): Predicted score, shape (batch_size, ...).
        eps (torch.Tensor): Gaussian noise scaled by the standard deviation.
        noise_std (torch.Tensor): Standard deviation of the cumulative noise.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    loss = torch.mean((score.conj() * (noise_std * score + 2 * eps)).real)
    return loss
