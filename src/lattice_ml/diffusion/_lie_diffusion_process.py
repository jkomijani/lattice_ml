# Created by Javad Komijani, 2024-2025

"""Implements the diffusion process."""

# pylint: disable=too-many-arguments, too-many-positional-arguments

from typing import Callable, Tuple, Dict

import pydantic
import torch
import numpy as np

from ._trainer import Trainer
from ._noise_schedule import InverseTimeNoiseSchedule

from .gauge._randn_xxx_like import randn_special_unitary_like
from .gauge._randn_xxx_like import randn_traceless_antihermitian_like
from .gauge._lie_sdeint import integrate_sde


__all__ = ["SUnDiffusionProcess"]


# =============================================================================
class SUnDiffusionProcess(torch.nn.Module):
    r"""
    Implements a diffusion process as described by the following stochastic
    differential equation (SDE)

    .. math::
        \frac{d U(t)}{dt} = \sigma(t) \eta(t) U(t)

    where
    - :math:`\sigma(t)` is a time-dependent noise scale,
    - :math:`\eta(t)` is standard white Gaussian noise in the algebra space.

    By default, the noise scale :math:`\sigma(t)` follows an inverse-time
    variance law, implemented by :class:`InverseTimeNoiseSchedule`, in which
    the variance grows as :math:`1/(1 - t)`.

    Use Cases:
    - Simulate noisy trajectories (`forward`).
    - Train score-based generative models (`training_step`).
    - Generate clean samples by reversing the diffusion (`reverse`).
    """

    def __init__(
        self, score_fn: Callable,
        sigma_0: float = 1.0,
        sigma_schedule: Callable | None = None,
        n_random_walk_steps: int = 4,
        training_config: Dict | None = None
    ):
        """Initializes the diffusion process with a score function.

        Args:
            score_fn (Callable): A neural network for the score function.
            sigma_0 (float): Scaling parameter controlling the noise intensity.
                (Default is 1. Ignored if `sigma_schedule` is provided.)
            sigma_schedule (Callable): Defines the time-dependent noise scale.
                (Default is :class:`InverseTimeNoiseSchedule(sigma_0)`.)
            n_random_walk_steps (int): Number of multipicative terms to obtain
                the heat kernel. (Default is 4.)
            training_config: Contains optional loss_c0 for configuration.
        """
        super().__init__()

        if sigma_schedule is None:
            self.sigma_schedule = InverseTimeNoiseSchedule(sigma_0)
        else:
            self.sigma_schedule = sigma_schedule

        self.score_fn = score_fn
        self.n_random_walk_steps = n_random_walk_steps

        # Components for training
        self.trainer = Trainer(self)

        if training_config is None:
            training_config = {}
        self.training_config = TrainingConfiguration(**training_config)

    def training_step(self, batch, batch_idx=None):
        """Perform a training step to be used by Trainer."""
        x_0, = batch
        bsize = x_0.shape[0]

        # Choose a random diffusion time per sample, uniformly in [0, 1].
        t = torch.rand((bsize,), device=x_0.device)

        # Run the process to time t & get the injected `noise/std` and also std
        x_t, eps, noise_std = self.run_for_training(x_0, t)

        # Predict the score at (t, x_t).
        score = self.score_fn(t, x_t)

        # Compute loss: implicit score matching weighted by noise standard dev.
        loss = implicit_score_matching_with_sdev_weight(score, eps, noise_std)

        # contribution from t = 0 if loss_c0 > 0
        if self.training_config.loss_c0 > 0:
            ind = np.random.randint(0, len(x_0))  # choose only one sample
            score0 = self.score_fn(0 * t[ind:ind+1], x_0[ind:ind+1])
            force0 = self.training_config.force0_fn(x_0[ind:ind+1])
            res = score0 - force0
            loss0 = torch.mean(res * res.conj()).real
            loss = loss + self.training_config.loss_c0 * loss0

        return loss

    def run_for_training(self, y_0: torch.Tensor, t_eval: torch.Tensor, t_0=0):
        """Simulates the forward diffusion process for training purposes.

        Applies a discretized random walk in the Lie algebra with variance
        given by self.sigma_schedule, mapped to the group via the matrix
        exponential.

        Args:
            y_0 (torch.Tensor): The initial state of the system at time `t_0`.
            t_eval (torch.Tensor): A 1d or 0d tensor containing the evaluation
                   times. If 1d, its length must match the batch size of `y_0`.
            t_0 (float): The initial time for the simulation. Default is 0.

        Returns:
            A tuple containing
            - torch.Tensor: `y_t` (noisy group state): The final noisy states.
            - torch.Tensor: `alg / std` (normalized algebraic state): The state
                evoloved in the algebra, normalized by its standard deviation.
            - Tensor: `std`: The accumulated standard deviation of noise over
                time, tracking how much noise is added to the algebraic state.
        """
        # Expand t_eval dimensions to match y_0
        t_eval = t_eval.view(-1, *[1] * (y_0.ndim - 1))

        # Time step for discretized diffusion
        h = (t_eval - t_0) / self.n_random_walk_steps

        cum_randn_alg = 0
        y_t = y_0

        # Discretized random walk in the Lie algebra
        for m in range(self.n_random_walk_steps):
            std = self.sigma_schedule.cumulative(t_0 + m * h, t_0 + (m+1) * h)
            randn_alg = std * randn_traceless_antihermitian_like(y_t)
            y_t = torch.matrix_exp(randn_alg) @ y_t
            cum_randn_alg = cum_randn_alg + randn_alg

        # Total noise scale over the full interval
        cum_std = self.sigma_schedule.cumulative(t_0, t_eval)

        return y_t, cum_randn_alg / cum_std, cum_std

    def forward(
        self,
        y_0: torch.Tensor,
        t_0: float = 0.,
        t_eval: Tuple[float] | float = 1.0
    ):
        """Simulates the forward diffusion process.

        Args:
            y_0 (torch.Tensor): The initial state of the system at time `t_0`.
            t_0 (float): The initial time for the simulation. Default is 0.
            t_eval (Tuple[float] | float): A time or a monotonically increasing
               sequence of times at which the system state must be evaluated.

        Returns:
            The states `y_t` of the system at times defined in `t_eval`.
        """
        if isinstance(t_eval, (float, int)):
            t_eval = (t_eval,)
            squeeze_output = True
        else:
            squeeze_output = False

        y_eval = [None] * len(t_eval)

        for ind, t in enumerate(t_eval):
            assert t >= t_0, "`t_eval` must monotonically increase."

            std = self.sigma_schedule.cumulative(t_0, t)

            n_steps = 1 if (t - t_0).abs() < 0.05 else self.n_random_walk_steps

            randn_grp, _ = randn_special_unitary_like(y_0, std, n_steps)

            # Simulate the diffusion process
            y_eval[ind] = randn_grp @ y_0

            # Update the state for the next round
            y_0, t_0 = y_eval[ind], t

        return y_eval[0] if squeeze_output else y_eval

    def reverse(
        self,
        y_0: torch.Tensor,
        t_0: float = 1.0,
        t_eval: Tuple[float] | float = 0.,
        method: str = 'Euler-Maruyama:su(n)',
        step_size: float = 0.01,
        rev2fwd_noise_ratio: float = 1.0
    ):
        """Simulates the revese diffusion process (denoising process).

        Args:
            y_0 (torch.Tensor): The initial state of the system at time `t_0`.
            t_0 (float): The initial time for the simulation. Default is 0.
            t_eval (Tuple[float] | float): A time or a monotonically increasing
               sequence of times at which the system state must be evaluated.
            method (str): The solving method. Default 'Euler-Maruyama:su(n)'.
            step_size (float): The discretization step size. Default is 0.001.
            rev2fwd_noise_ratio (float): Controls stochasticity in reverse SDE.
                It is the ratio of the noise strength in the reverse vs forward
                processes. Default is 1.

        Returns:
            torch.Tensor: The denoised state at each time step.
        """
        if isinstance(t_eval, (float, int)):
            t_eval = (t_eval,)
            squeeze_output = True
        else:
            squeeze_output = False

        # Put all key-word args in a dictionary to pass to the solver
        rev2fwd = rev2fwd_noise_ratio
        solver_kwargs = {
            'method': method,
            'step_size': step_size,
            'noise_scale': lambda t: rev2fwd * self.sigma_schedule(t)
        }

        # Build the drift function used in the reverse SDE
        drift = self.build_denoising_drift(rev2fwd_noise_ratio)

        y_eval = [None] * len(t_eval)

        for ind, t in enumerate(t_eval):
            assert t <= t_0, "`t_eval` must monotonically decrease."

            # Simulate the denoising process
            y_eval[ind] = integrate_sde(drift, (t_0, t), y_0, **solver_kwargs)

            # Update the state for the next round
            y_0, t_0 = y_eval[ind], t

        return y_eval[0] if squeeze_output else y_eval

    def build_denoising_drift(
        self,
        rev2fwd_noise_ratio: float,
        algebra_valued: bool = True
    ):
        """Build the drift function for the reverse diffusion process.

        The reverse-time drift is defined as:
            f(t, x) = -½ σ(t)² (1 + r²) ∇ log p_t(x),
        where σ(t) is the noise schedule, r is the reverse/forward noise ratio,
        and ∇ log p_t(x) is approximated by the score function.

        Args:
            rev2fwd_noise_ratio (float):
                Ratio of reverse to forward noise, controlling stochasticity.
            algebra_valued (bool):
                If True, return an algebra-valued drift. Default is True.

        Returns:
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
                Function denoising_drift(t, y_t) computing the drift at time t.
        """
        constant_factor = 0.5 * (1 + rev2fwd_noise_ratio ** 2)

        def alg_denoising_drift(t: torch.Tensor, y_t: torch.Tensor):
            """Compute the algebra-valued drift for the denoising process.

            Args:
                t (torch.Tensor): Time tensor (scalar or [batch]).
                y_t (torch.Tensor): System state at time t.

            Returns:
                torch.Tensor: Drift vector field at `(t, y_t)`.
            """
            score = self.score_fn(t, y_t)
            score_coeff = constant_factor * self.sigma_schedule(t) ** 2
            return - score_coeff * score

        def grp_denoising_drift(t: torch.Tensor, y_t: torch.Tensor):
            """Compute the drift for the denoising process.

            Args:
                t (torch.Tensor): Time tensor (scalar or [batch]).
                y_t (torch.Tensor): System state at time t.

            Returns:
                torch.Tensor: Drift vector field at `(t, y_t)`.
            """
            score = self.score_fn(t, y_t)
            score_coeff = constant_factor * self.sigma_schedule(t) ** 2
            return - score_coeff * score @ y_t

        return alg_denoising_drift if algebra_valued else grp_denoising_drift


# =============================================================================
class TrainingConfiguration(pydantic.BaseModel):
    """Training Configuration."""

    loss_c0: float = 0
    force0_fn: Callable | None = None

    def update(self, **kwargs):
        """Update the attributes."""
        for key, value in kwargs.items():
            setattr(self, key, value)


# =============================================================================
def implicit_score_matching_with_variance_weight(
    score: torch.Tensor,
    eps: torch.Tensor,
    noise_std: torch.Tensor
) -> torch.Tensor:
    """Compute the implicit score matching loss.

    This computes a weighted mean squared error (MSE) between the predicted and
    the empirical conditional score at a diffusion time. The MSE is weighted by
    the effective (cumulative) noise variance at the diffusion time. For matrix
    indices, this computes Tr(x x^†) / N and is intended for SU(n) matrix data.

    Args:
        score (torch.Tensor): Predicted score, shape (batch_size, ...).
        eps (torch.Tensor): Gaussian noise scaled by the standard deviation.
        noise_std (torch.Tensor): Standard deviation of the cumulative noise.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    n_c = score.shape[-1]

    # Residual between predicted and conditional scores (variance-weighted)
    res = score * noise_std + eps
    loss = torch.mean(res * res.conj()).real

    # Correcting for noise fluctuation
    fluctuation = torch.mean(eps * eps.conj()).real - (n_c**2 - 1) / n_c**2

    return (loss - fluctuation) * n_c


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
