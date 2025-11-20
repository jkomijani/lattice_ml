# Created by Javad Komijani, 2024-2025

"""Implements the diffusion process."""

# pylint: disable=too-many-arguments, too-many-positional-arguments
# pylint: disable=arguments-differ

from typing import Callable, Tuple, Dict
from datetime import datetime

import pydantic
import torch

from lightning.pytorch import LightningModule
from lightning.pytorch.utilities import rank_zero_only

from ._randn_xxx_like import randn_special_unitary_like
from ._lie_sdeint import integrate_sde


__all__ = ["SUnDiffusionModel", "InverseTimeNoiseScaleSchedule"]


# =============================================================================
class SUnDiffusionModel(LightningModule):
    r"""
    Implements a diffusion process as described by the following stochastic
    differential equation (SDE)

    .. math::
        \frac{d U(t)}{dt} = \sigma(t) \eta(t) U(t)

    where
    - :math:`\sigma(t)` is a time-dependent noise scale,
    - :math:`\eta(t)` is standard white Gaussian noise in the algebra space.

    By default, the noise scale :math:`\sigma(t)` follows an inverse-time
    variance law, implemented by :class:`InverseTimeNoiseScaleSchedule`, in
    which the variance grows as :math:`1/(1 - t)`.

    Use Cases:
    - Simulate noisy trajectories (`forward`).
    - Train score-based generative models (`run_for_training`).
    - Generate clean samples by reversing the diffusion (`reverse`).
    """

    def __init__(
        self, score_fn: Callable,
        sigma_0: float = 1.0,
        sigma_schedule: Callable | None = None,
        n_random_walk_steps: int = 4,
        training_config: Dict | "TrainingConfiguration" | None = None
    ):
        """Initializes the diffusion process with a score function.

        Args:
            score_fn (Callable): A neural network for the score function.
            sigma_0 (float): Scaling parameter controlling the noise intensity.
                (Default is 1. Ignored if `sigma_schedule` is provided.)
            sigma_schedule (Callable): Defines the time-dependent noise scale.
                (Default is :class:`InverseTimeNoiseScaleSchedule(sigma_0)`.)
            n_random_walk_steps (int): Number of multipicative terms to obtain
                the heat kernel. (Default is 4.)
            training_config: Contains training configuration.
        """
        super().__init__()

        if sigma_schedule is None:
            self.sigma_schedule = InverseTimeNoiseScaleSchedule(sigma_0)
        else:
            self.sigma_schedule = sigma_schedule

        self.score_fn = score_fn
        self.n_random_walk_steps = n_random_walk_steps

        if training_config is None:
            training_config = TrainingConfiguration()
        if isinstance(training_config, dict):
            training_config = TrainingConfiguration(**training_config)
        self.training_config = training_config

    def training_step(self, batch, batch_idx):
        """Perform a training step to be used by Lightning Trainer."""

        x_0, = batch
        bsize = x_0.shape[0]

        # Choose a random diffusion time per sample, uniformly in [0, 1].
        t = torch.rand((bsize,), device=x_0.device)

        # Run the process to time t & get the injected noise and its std.
        x_t, eps, noise_std = self.run_for_training(x_0, t)

        # Predict the score at (t, x_t).
        score = self.score_fn(t, x_t)

        # Compute loss: implicit score matching weighted by noise variance.
        loss = implicit_score_matching(score, eps, noise_std)

        # contribution from t = 0 if loss_c0 > 0
        if self.training_config.loss_c0 > 0:
            score0 = self.score_fn(0 * t, x_0)
            force0 = self.training_config.force0_fn(x_0)
            loss0 = implicit_score_matching(score0, -force0, 1)
            loss = loss + self.training_config.loss_c0 * loss0

        self.log(
            "loss", loss,
            on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )

        return loss

    @rank_zero_only
    def on_train_epoch_end(self):
        """To be used by Lightning Trainer."""
        print_every = self.training_config.print_every
        if print_every is None:
            return

        loss = self.trainer.callback_metrics.get("loss")  # avg_loss
        if self.current_epoch % print_every == 0:
            timestamp = datetime.now().strftime("%H:%M:%S")
            epoch = self.current_epoch
            self.print(f"{timestamp} | Epoch: {epoch:d} | Loss: {loss:.4f}")

    def configure_optimizers(self):
        """To be used by Lightning Trainer."""
        return self.training_config.configure_optimizers(self.parameters())

    def run_for_training(self, y_0: torch.Tensor, t_eval: torch.Tensor):
        r"""
        Simulates the forward diffusion process for training purposes.

        This method simulates the forward diffusion process starting from an
        initial state `y_0`, which is a tensor of the Lie group elements, over
        discrete time steps `t_steps`. The method computes noisy states and
        their noise characteristics, which can be used for training denoising
        models.

        Parameters
        ----------
        y_0 : torch.Tensor
            The initial state of the system at time 0.

        t_eval : torch.Tensor
            A 1-dimensional tensor containing the number of evaluations times.
            Its length must match the batch size of `y_0`.

        Returns
        -------
            A tuple containing
            - torch.Tensor: `y_t` (noisy state): The noisy states after
                applying the diffusion process over the specified time steps.

            - torch.Tensor: `alg / std` (normalized algebraic state): The state
                in the tangent space of the Lie group, normalized by its
                standard deviation. This tensor shows how noise evolves in the
                algebraic space over time.

            - Tensor: `std`: The accumulated standard deviation (noise) over
                time. This tensor tracks how much noise has been added to the
                algebraic state during the diffusion process.
        """
        # Expand t_eval dimensions to match y_0
        t_eval = t_eval.view(-1, *[1] * (y_0.ndim - 1))

        # Compute the cumulative noise from 0 to t_intermediate
        std = self.sigma_schedule.cumulative(0, t_eval)
        n_steps = self.n_random_walk_steps
        randn_grp, randn_alg = randn_special_unitary_like(y_0, std, n_steps)

        # Simulate the diffusion process
        y_t = randn_grp @ y_0

        return y_t, randn_alg / std, std

    def forward(
        self,
        y_0: torch.Tensor,
        t_0: float = 0.,
        t_eval: Tuple[float] | float = 1.0
    ):
        """
        Simulates the forward diffusion process, evolving the state `y_0` from
        initial time `t_0` to the time points specified by `t_eval`.

        Parameters
        ----------
        y_0 : torch.Tensor
            The initial state of the system at time `t_0`.

        t_0 : float, optional
            The initial time for the simulation. Default is 0.

        t_eval : Tuple of float | float
            A sequence of times at which the state `y_t` of the system is
            evaluated. The times must be monotonically increasing.

        Returns
        -------
        list of Tensor
            A list containing the states `y_t` of the system at each time
            defined in `t_eval`.
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
        """
        Simulates the denoising process (reverse diffusion process) by applying
        the learned score function to gradually remove the noise and recover
        the clean signal.

        Parameters
        ----------
        y_0 : torch.Tensor
            The state of the system at time `t_0`.

        t_0 : float, optional
            The initial time for the denoising process. Default is 1.

        t_eval : Tuple of float | float
            A sequence of times at which the denoised state `y_t` of the system
            is evaluated. Times must be monotonically decreasing.

        method : str, optional
            The method for solving the SDE. Default is 'Euler-Maruyama:su(n)'.

        step_size : float, optional
            The step size for discretizing the SDE. Default is 0.001.

        rev2fwd_noise_ratio : float, optional
            Scaling factor for the reverse noise relative to the forward
            process. Default is 1. Controls stochasticity in reverse SDE.

        Returns
        -------
        torch.Tensor
            The denoised state at each time step.
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
        """
        Build the drift function for the reverse (denoising) diffusion process.

        The reverse-time drift is defined as:
            f(t, x) = -½ σ(t)² (1 + r²) ∇ log p_t(x),
        where σ(t) is the noise schedule, r is the reverse/forward noise ratio,
        and ∇ log p_t(x) is approximated by the score function.

        Parameters
        ----------
        rev2fwd_noise_ratio : float, optional
            Ratio of reverse to forward noise, controlling stochasticity.

        algebra_valued : bool, opitonla
            If True, return an algebra-valued drift. Default is True.

        Returns
        -------
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
            Function denoising_drift(t, y_t) computing the drift at time t.
        """
        constant_factor = 0.5 * (1 + rev2fwd_noise_ratio ** 2)

        def alg_denoising_drift(t: torch.Tensor, y_t: torch.Tensor):
            """
            Compute the algebra-valued drift for the reverse diffusion process.

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
            """
            Compute the drift for the reverse diffusion process.

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

    loss_c0: int = 0
    force0_fn: Callable = None
    print_every: int | None = None
    optimizer_class: Callable = torch.optim.AdamW
    scheduler: Callable | None = None
    hyperparam: Dict = {}

    def update(self, **kwargs):
        """Update the attributes."""
        for key, value in kwargs.items():
            setattr(self, key, value)

    def configure_optimizers(self, parameters):
        """To be used by Lightning Trainer"""
        optimizer = self.optimizer_class(parameters, **self.hyperparam)
        if self.scheduler is None:
            return optimizer

        scheduler = self.scheduler(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, "interval": "epoch",  "frequency": 1
            },
        }


# =============================================================================
class InverseTimeNoiseScaleSchedule:
    """
    Noise standard deviation scheduler derived from an inverse-time variance
    law: Var(t) ∝ 1 / (1 - t).

    This scheduler provides both the instantaneous noise std as a function of
    time, and its cumulative value between two time points.
    """

    EPS = 1e-4  # Small constant to regulate the divergence at t = 1

    def __init__(self, sigma_0: float = 1.0):
        """
        Initialize the noise standard deviation scheduler.

        Args:
            sigma_0 (float): Scaling factor (default is 1).
        """
        self.sigma_0 = sigma_0

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the instantaneous noise standard deviation at time `t`.

        Args:
            t (torch.Tensor): Time tensor with values in (0, 1).

        Returns:
            torch.Tensor: Standard deviation of noise at time `t`.
        """
        return self.sigma_0 / (1 + self.EPS - t) ** 0.5

    def cumulative(self, t_0: torch.Tensor, t_1: torch.Tensor) -> torch.Tensor:
        """
        Compute the cumulative noise std between two times `t_0` and `t_1`.

        Args:
            t_0 (torch.Tensor): Start time tensor.
            t_1 (torch.Tensor): End time tensor.

        Returns:
            torch.Tensor: Cumulative noise standard deviation.
        """
        t_max = 1 + self.EPS
        return self.sigma_0 * torch.sqrt(
            torch.log((t_max - t_0) / (t_max - t_1)).abs()
        )


# =============================================================================
def implicit_score_matching(
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
        eps (torch.Tensor): Gaussian (cumulative) noise added during diffusion.
        noise_std (torch.Tensor): Standard deviation of the cumulative noise.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    n_c = score.shape[-1]

    # Residual between predicted and conditional scores (variance-weighted)
    res = score * noise_std + eps
    loss = torch.mean(res * res.conj()).real

    # Correcting for noise fluctuation
    fluctuation = torch.mean(eps * eps.conj()).real - (n_c ** 2 - 1) / n_c ** 2

    return (loss - fluctuation) * n_c ** 2
