# Created by Javad Komijani, 2024-2026

"""Implements diffusion models."""

# pylint: disable=too-many-arguments, too-many-positional-arguments

from typing import Callable, Sequence

import torch

from lattice_ml.integrate import odeint

from ._trainer import Trainer
from ._noise_schedule import VPInverseTimeNoiseSchedule



__all__ = ["DiffusionModel", "VPDiffuser"]


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
        score_fn: Callable,
        diffuser: Callable | None = None,
        as_score_plus_x: bool = False
    ):
        """
        Initializes the diffusion process with a score function.

        Args:
            score_fn (Callable): A neural network for the score function.
            diffuser (Callable | None): Defines the diffusion process. If not
                provided, defaults to an instance of `VPDiffuser`.
            as_score_plus_x (bool): If True, treats the provided score function
                as `score + x`.
        """
        if diffuser is None:
            diffuser = VPDiffuser()

        super().__init__()
        self.score_fn = score_fn
        self.diffuser = diffuser
        self.as_score_plus_x = as_score_plus_x
        self.trainer = Trainer(self)

    def training_step(self, batch, batch_idx=None):
        """Perform a training step to be used by Trainer."""
        x_0, = batch
        bsize = x_0.shape[0]

        # Choose a random diffusion time per sample, uniformly in [0, 1].
        t = torch.rand((bsize,), device=x_0.device)

        # Run the process to time t & get the injected `noise/std` and also std
        x_t, noise, noise_scale, signal_scale = self.diffuser(x_0, t_0=0, t=t)

        # Predict the score at (t, x_t).
        score = self.score_fn(t, x_t)

        # Compute loss: implicit score matching
        if self.as_score_plus_x:
            # The evaluated score should be treated as score + x_t
            score_plus_x = score
            loss = implicit_score_plus_x_matching(
                score_plus_x, x_t, noise, noise_scale, signal_scale
            )
        else:
            loss = implicit_score_matching_with_variance_weight(
                score, noise, noise_scale
            )

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
            x_eval[ind], _, _, _ = self.diffuser(x_0, t_0=t_0, t=t)

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
            self.score_fn, t_span, x_0, self.as_score_plus_x, **kwargs
        )


# =============================================================================
class VPDiffuser(torch.nn.Module):
    r"""
    Implements a variance preserving diffusion process as

    .. math::
        \frac{d x(t)}{dt} = - \frac{1}{2}\sigma^2(t) x(t) + \sigma(t) \eta(t)

    By default, we use `\sigma(t) = \sqrt{2 / (1 - t)}`.
    """
    vanishing_drift_term = False

    def __init__(self, sigma_schedule: Callable | None = None):
        """Initializes the diffusion process with a score function.

        Args:
            sigma_schedule (Callable): Defines the time-dependent noise scale.
                (Default is :class:`VPInverseTimeNoiseSchedule()`.)
        """
        super().__init__()
        if sigma_schedule is None:
            sigma_schedule = VPInverseTimeNoiseSchedule()
        self.sigma_schedule = sigma_schedule

    def forward(self, x_0: torch.Tensor, t_0: torch.Tensor, t: torch.Tensor):
        """
        Simulates the forward diffusion process.

        The process starts from the initial state `x_0` at time `t_0` and
        evolves the states until the terminal time `t`.

        Args:
            x_0 (torch.Tensor): The initial state of the system at time `t_0`.
            t_0 (torch.Tensor): A 0d or 1d tensor of the initial times.
            t (torch.Tensor): A 0d or 1d tensor of the terminal times.

        Note:
            At least one of `t_0` or `t` must be an instace of `torch.Tensor`.
            If a 1d tensor, their lengths must match the batch size of `x_0`.

        Returns
        -------
        torch.Tensor, torch.Tensor, torch.Tensor
            A tuple containing:
            - `x_t`: the final diffused states of the system,
            - `eps`: the noise samples used in the diffusion process,
            - `std`: the (exact) standard deviation of the noise samples.
        """
        # Expand t_eval dimensions to match x_0
        t = t.view(-1, *[1] * (x_0.ndim - 1))

        # Compute accumulated noise standard deviation and its complementary
        noise_scale = self.sigma_schedule.noise_scale(t_0, t)
        signal_scale = self.sigma_schedule.signal_scale(t_0, t)

        # Sample from normal distribution
        noise = torch.randn_like(x_0)

        # Closed-form solution
        x_t = signal_scale * x_0 + noise_scale * noise

        return x_t, noise, noise_scale, signal_scale

    def odeint(
        self,
        score_fn: Callable,
        t_span: Sequence[float],
        x_0: torch.Tensor,
        as_score_plus_x: bool = False,
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
        by `score_fn`.

        Args:
            score_fn (Callable): Function approximating the score.
            t_span (Sequence[float, float]): Integration interval.
            x_0 (torch.Tensor): Initial state.
            as_score_plus_x (bool): If True, treats the provided score function
                as `score + x`.

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
        if as_score_plus_x:
            score_plus_x_fn = score_fn
        else:
            def score_plus_x_fn(t, x_t):
                return x_t + score_fn(t, x_t)

        def drift_fn(t, x_t):
            """Compute the drift fucntion for the ODE form of the process."""
            coeff = -0.5 * self.sigma_schedule(t)**2
            return coeff * score_plus_x_fn(t, x_t)

        return odeint(drift_fn, t_span, x_0, **solver_kwargs)


# =============================================================================
def implicit_score_plus_x_matching(
    score_plus_x: torch.Tensor,
    x_t: torch.Tensor,
    noise: torch.Tensor,
    noise_scale: torch.Tensor,
    signal_scale: torch.Tensor
) -> torch.Tensor:
    """Compute the implicit score matching loss applied on score_plus_x.

    This computes a weighted mean squared error (MSE) between the predicted and
    the empirical conditional score at a diffusion time. The MSE is weighted by
    the variance of the accumulated noise at the diffusion time divided by
    sqrt of one minus its power.

    Args:
        score_plus_x (torch.Tensor): Predicted score plus `x_t`.
        x_t (torch.Tensor): The state of the system.
        noise (torch.Tensor): The non-scaled noise injected to `x_t`.
        noise_scale (torch.Tensor): The scale of the cumulative noise.
        signal_scale (torch.Tensor): The scale of the original signal in `x_t`.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Residual between predicted and conditional scores (variance-weighted)
    w = 1 / signal_scale
    res = w * (score_plus_x * noise_scale + (noise - x_t * noise_scale))
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
