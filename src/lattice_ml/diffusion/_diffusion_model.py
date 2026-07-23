# Created by Javad Komijani, 2024-2026

"""Implements diffusion models."""

# pylint: disable=too-many-arguments, too-many-positional-arguments

from typing import Callable, Dict, Sequence

import numpy as np
import pydantic
import torch

from ._trainer import Trainer


__all__ = ["DiffusionModel"]


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
        diffuser: Callable,
        network_fn: Callable,
        network_role: str = "dynamics_fn",
        use_inverse_snr_weight: bool = True,
        training_config: Dict | None = None,
    ):
        """
        Initializes the diffusion process with a network function.

        Args:
            diffuser (Callable): Defines the diffusion process.
            network_fn (Callable): A neural network whose output is interpreted
                according to `network_role`.
            network_role (str): Specifies the role of `network_fn`. The default
                value is 'dynamics_fn'. This reparameterization keeps the model
                stable even when the diffuser's schedule diverges at t=1.
            use_inverse_snr_weight (bool): Specifies the weight for the loss
                function. Default is True.
            training_config (Dict | None): Optional dict passed to
                :class:`TrainingConfiguration`.
        """
        super().__init__()
        self.diffuser = diffuser
        self.network_fn = network_fn
        self.network_role = network_role
        self._as_score = network_role == "score_fn"
        self._as_score_plus_x = network_role == "score_plus_x_fn"
        self._as_ode_dynamics = network_role in ("dynamics_fn", "velocity_fn")
        self.trainer = Trainer(self)
        self.training_config = TrainingConfiguration(**(training_config or {}))
        self._setup_matching_loss_fn(use_inverse_snr_weight)

    def _setup_matching_loss_fn(self, use_inverse_snr_weight: bool):
        """
        Depending on the input and `self.network_role` specifies the loss func.
        """
        if use_inverse_snr_weight:
            if self._as_ode_dynamics:
                func = implicit_dynamics_matching_with_inverse_snr_weight
            elif self._as_score_plus_x:
                func = implicit_score_plus_x_matching_with_inverse_snr_weight
            else:
                raise ValueError("NOT READY")
        else:
            if self._as_ode_dynamics:
                func = implicit_dynamics_matching_with_variance_weight
            elif self._as_score_plus_x:
                func = implicit_score_plus_x_matching_with_variance_weight
            elif self._as_score:
                func = implicit_score_matching_with_variance_weight
            else:
                raise ValueError("NOT READY")
        self._matching_loss_fn = func

    @property
    def score_fn(self):
        """The score function."""

        if self._as_ode_dynamics:
            return self.diffuser.build_score_fn(self.network_fn)
        if self._as_score:
            return self.network_fn
        if self._as_score_plus_x:
            return lambda t, x_t: self.network_fn(t, x_t) - x_t

        raise ValueError(f"{self.network_role} is not known.")

    @property
    def score_plus_x_fn(self):
        """The `score + x` function."""

        if self._as_ode_dynamics:
            score_fn = self.diffuser.build_score_fn(self.network_fn)
            return lambda t, x_t: score_fn(t, x_t) + x_t
        if self._as_score:
            return lambda t, x_t: self.network_fn(t, x_t) + x_t
        if self._as_score_plus_x:
            return self.network_fn

        raise ValueError(f"{self.network_role} is not known.")

    @property
    def dynamics_fn(self):
        """The probability flow ODE dynamics of this diffusion process."""

        if self._as_ode_dynamics:
            return self.network_fn
        return self.diffuser.build_ode_dynamics_fn(self.score_plus_x_fn)

    def training_step(self, batch, batch_idx=None):
        """Perform a training step to be used by Trainer."""
        x_0, = batch
        bsize = x_0.shape[0]

        # Choose a random diffusion time per sample, uniformly in [0, 1].
        t = torch.rand((bsize,), device=x_0.device)

        # Run the process to time t & get the context of the diffusion
        x_t, diffusion_context = self.diffuser((0, t), x_0)

        # Compute loss: implicit score matching
        loss = self._matching_loss_fn(
            self.network_fn(t, x_t), diffusion_context
        )

        # Contribution from t = 0 if loss_c0 > 0
        if self.training_config.loss_c0 > 0:
            loss = loss + self._penalty_term_from_exact_score(0 * t, x_0)

        return loss

    def _penalty_term_from_exact_score(self, t_0, x_0):
        idx = (slice(None) if self.training_config.all_samples_c0
               else np.random.randint(0, len(x_0), size=1)
               )
        score0 = self.score_fn(t_0[idx], x_0[idx])
        force0 = self.training_config.force0_fn(x_0[idx])
        res = score0 - force0
        loss0 = torch.mean(res * res.conj()).real
        return self.training_config.loss_c0 * loss0

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
            x_eval[ind] = self.diffuser((t_0, t), x_0)[0]

            # Update the state for the next round
            x_0, t_0 = x_eval[ind], t

        return x_eval[0] if squeeze_output else x_eval

    def reverse(
        self,
        x_0: torch.Tensor,
        t_0: float = 1.0,
        t_eval: float | Sequence[float] | torch.Tensor = 0.,
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
            step_size: Solver step size.
            **solver_kwargs: Additional keyword arguments forwarded to the
                solver, e.g. `method` and `num_steps`.

        Returns:
            Final state or states at `t_eval`.
        """
        # Convert to tensor (on correct device)
        t_eval = torch.as_tensor(t_eval, device=x_0.device)

        # Determine integration interval
        t_end = t_eval if t_eval.ndim == 0 else t_eval.min()
        t_span = (t_0, t_end)

        # Pass solver options
        kwargs = {**solver_kwargs, "step_size": step_size}

        # Only pass t_eval if it's not scalar
        if t_eval.ndim > 0:
            kwargs["t_eval"] = t_eval

        return self.diffuser.integrate(self.dynamics_fn, t_span, x_0, **kwargs)


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
def implicit_dynamics_matching_with_inverse_snr_weight(
    velocity: torch.Tensor,
    diffusion_context: Dict
) -> torch.Tensor:
    """
    Compute the implicit score matching loss applied on dynamics function.

    The time weight is set to 1/SNR(t) = noise_scale^2 / signal_scale^2.

    Args:
        velocity (torch.Tensor): Predicted dynamics/velocity, dx_t/dt.
        diffusion_context (dict): Dictionary containing quantities from
            the forward diffusion step:
            - complementary_state (torch.Tensor): Complementary to `x_t`.
            - signal_scale (torch.Tensor): Scale of the signal in `x_t`.
            - noise_scale (torch.Tensor): Scale of the cumulative noise.
            - half_sigma_square (torch.Tensor): Half the square of `sigma(t)`.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    u_t = diffusion_context['complementary_state']
    a_t = diffusion_context['signal_scale']
    b_t = diffusion_context['noise_scale']
    half_sigma_square = diffusion_context['half_sigma_square']

    beta_t = b_t / (a_t * half_sigma_square)
    beta_t = torch.nan_to_num(beta_t, nan=1.0)  # in case t might be 0 or 1

    res = u_t - beta_t * velocity
    return torch.mean(res * res.conj()).real


# =============================================================================
def implicit_dynamics_matching_with_variance_weight(
    velocity: torch.Tensor,
    diffusion_context: Dict
) -> torch.Tensor:
    """
    Compute the implicit score matching loss applied on dynamics function.

    The time weight is set to noise_scale^2, equivalent to DDPM's unweighted
    epsilon-prediction loss.

    Args:
        velocity (torch.Tensor): Predicted dynamics/velocity, dx_t/dt.
        diffusion_context (dict): Dictionary containing quantities from
            the forward diffusion step:
            - complementary_state (torch.Tensor): Complementary to `x_t`.
            - signal_scale (torch.Tensor): Scale of the signal in `x_t`.
            - noise_scale (torch.Tensor): Scale of the cumulative noise.
            - half_sigma_square (torch.Tensor): Half the square of `sigma(t)`.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    u_t = diffusion_context['complementary_state']
    a_t = diffusion_context['signal_scale']
    b_t = diffusion_context['noise_scale']
    half_sigma_square = diffusion_context['half_sigma_square']

    alpha_t = b_t / half_sigma_square
    alpha_t = torch.nan_to_num(alpha_t, nan=1.0)  # in case t might be 0 or 1

    res = a_t * u_t - alpha_t * velocity
    return torch.mean(res * res.conj()).real


# =============================================================================
def implicit_score_plus_x_matching_with_inverse_snr_weight(
    score_plus_x: torch.Tensor,
    diffusion_context: Dict
) -> torch.Tensor:
    """Compute the implicit score matching loss applied on score_plus_x.

    This computes a weighted mean squared error (MSE) between the predicted and
    the empirical conditional score. The MSE is weighted by the variance of the
    accumulated noise relative to the remaining signal scale.

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
    u_t = diffusion_context['complementary_state']
    noise_scale = diffusion_context['noise_scale']
    signal_scale = diffusion_context['signal_scale']

    res = u_t + score_plus_x * (noise_scale / signal_scale)
    loss = torch.mean(res * res.conj()).real

    return loss


# =============================================================================
def implicit_score_plus_x_matching_with_variance_weight(
    score_plus_x: torch.Tensor,
    diffusion_context: Dict
) -> torch.Tensor:
    """Compute the implicit score matching loss applied on score_plus_x.

    This computes a weighted mean squared error (MSE) between the predicted and
    the empirical conditional score at a diffusion time. The MSE is weighted by
    the effective (cumulative) noise variance at the diffusion time; equivalent
    to DDPM's unweighted epsilon-prediction loss.

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
    u_t = diffusion_context['complementary_state']
    noise_scale = diffusion_context['noise_scale']
    signal_scale = diffusion_context['signal_scale']

    res = signal_scale * u_t + noise_scale * score_plus_x
    loss = torch.mean(res * res.conj()).real

    return loss


# =============================================================================
def implicit_score_matching_with_variance_weight(
    score: torch.Tensor,
    diffusion_context: Dict
) -> torch.Tensor:
    """Compute the implicit score matching loss.

    This computes a weighted mean squared error (MSE) between the predicted and
    the empirical conditional score at a diffusion time. The MSE is weighted by
    the effective (cumulative) noise variance at the diffusion time; equivalent
    to DDPM's unweighted epsilon-prediction loss.

    Args:
        score (torch.Tensor): Predicted score, shape (batch_size, ...).
        diffusion_context (dict): Dictionary containing quantities from
            the forward diffusion step:
            - complementary_state (torch.Tensor): Complementary to `x_t`.
            - noise_scale (torch.Tensor): Scale of the cumulative noise.
            - signal_scale (torch.Tensor): Scale of the signal in `x_t`.
            - Optional keys like `noise` can also be included.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    noise = diffusion_context['noise']
    noise_scale = diffusion_context['noise_scale']

    res = noise + noise_scale * score
    loss = torch.mean(res * res.conj()).real

    return loss


# =============================================================================
def implicit_score_matching_with_sdev_weight(
    score: torch.Tensor,
    diffusion_context: Dict
) -> torch.Tensor:
    """Compute the implicit score matching loss.

    This computes a weighted mean squared error (MSE) between the predicted and
    the empirical conditional score at a diffusion time. The MSE is weighted by
    the effective (cumulative) noise standard deviation at the diffusion time.
    The pure noise contribution is excluded from the loss.

    Args:
        score (torch.Tensor): Predicted score, shape (batch_size, ...).
        diffusion_context (dict): Dictionary containing quantities from
            the forward diffusion step:
            - complementary_state (torch.Tensor): Complementary to `x_t`.
            - noise_scale (torch.Tensor): Scale of the cumulative noise.
            - signal_scale (torch.Tensor): Scale of the signal in `x_t`.
            - Optional keys like `noise` can also be included.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    noise = diffusion_context['noise']
    noise_scale = diffusion_context['noise_scale']

    loss = torch.mean((score.conj() * (noise_scale * score + 2 * noise)).real)
    return loss
