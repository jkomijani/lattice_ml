# Created by Javad Komijani, 2026

"""Implements consistency diffusion models."""

# pylint: disable=too-many-arguments, too-many-positional-arguments

import copy
from typing import Callable

import torch

from ._trainer import Trainer


__all__ = ["ConsistencyDiffusionModel"]


# =============================================================================
class ConsistencyDiffusionModel(torch.nn.Module):
    """
    Implements a consistency model trained on a diffusion process.

    A consistency function `f(x_t, t)` is trained to be (approximately)
    constant along the trajectories of the probability flow ODE associated
    with `diffuser`, so that a single evaluation at `t=1` maps noise directly
    to an estimate of the data (`t=0`). See `consistency_fn`.

    Based on consistency models as introduced by Song, Dhariwal, Chen, and
    Sutskever, "Consistency Models," arXiv:2303.01469 (2023).

    Two training methods are supported, selected by whether `dynamics_fn`
    is provided:
    - Consistency Training (CT), `dynamics_fn=None`: `(x_t, x_tau)` are built
      from the analytic forward-process marginal, sharing one noise sample
      (`_diffuse_to_pair`).
    - Consistency Distillation (CD), `dynamics_fn` given: `x_t` is drawn from
      the forward process, and `x_tau` is obtained from it by taking one
      Euler step along the known dynamics (see `delta_t`).

    Use Cases:
    - Train a consistency function via self-distillation (`training_step`).
    """

    def __init__(
        self,
        diffuser: Callable,
        network_fn: Callable,
        dynamics_fn: Callable | None = None,
        delta_t: float | None = 0.001,
        target_ema_decay: float | None = None,
    ):
        """
        Initializes the consistency model with a network function.

        Args:
            diffuser (Callable): Defines the diffusion process.
            network_fn (Callable): The consistency function `f(t, x_t)`,
                used directly. It is the caller's responsibility to make
                `network_fn` satisfy the boundary condition `f(0, x) = x`.
            dynamics_fn (Callable | None): If provided, the corresponding ODE
                dynamics `(t, x_t) -> dx/dt`, for training via Consistency
                Distillation (CD); if `None`, via Consistency Training (CT).
            delta_t (float | None): If given, `tau = clip(t - delta_t, 0, 1)`.
                Otherwise (for tests), `tau` is chosen uniformly in `[0, 1]`.
            target_ema_decay (float | None): If provided, must be in [0, 1],
                then exponential moving average (EMA) is used for updating the
                parameters of the target consistency function. Otherwise,
                `target_consistency_fn` becomes `network_fn` itself.
        """
        super().__init__()
        self.diffuser = diffuser
        self.consistency_fn = network_fn
        self.dynamics_fn = dynamics_fn
        self.delta_t = delta_t
        self.target_ema_decay = target_ema_decay

        if target_ema_decay is None:
            self.target_consistency_fn = network_fn
        else:
            self.target_consistency_fn = copy.deepcopy(network_fn)
            for param in self.target_consistency_fn.parameters():
                param.requires_grad_(False)

        self.trainer = Trainer(self)

    def training_step(self, batch, batch_idx=None):
        """Perform a training step to be used by Trainer."""
        x_0, = batch

        t, tau = self._prepare_t_and_tau(bsize=x_0.shape[0], device=x_0.device)

        if self.dynamics_fn is None:
            # Consistency Training (CT): analytic forward-process marginal.
            x_t, x_tau = self._diffuse_to_pair(x_0, t, tau)
        else:
            # Consistency Distillation (CD): an Euler step along known dynamics
            x_t, _ = self.diffuser(x_0, t_0=0, t=t)
            dt = (tau - t).view(-1, *[1] * (x_0.ndim - 1))
            with torch.no_grad():
                x_tau = self.diffuser.euler_step(self.dynamics_fn, x_t, t, dt)

        if self.target_ema_decay is not None:
            # Update the target network using the weights from previous steps
            self.update_target_network()

        f_t = self.consistency_fn(t, x_t)
        f_tau = self.target_consistency_fn(tau, x_tau)

        loss = squared_l2_distance(f_t, f_tau)
        if self.delta_t is not None:
            loss = loss * (1 / self.delta_t**2)

        return loss

    def update_target_network(self):
        """
        EMA-update `target_consistency_fn` toward the current `network_fn`.
        """
        if self.target_ema_decay is None:
            return
        decay = self.target_ema_decay
        with torch.no_grad():
            params = self.consistency_fn.parameters()
            target_params = self.target_consistency_fn.parameters()
            for param, target_param in zip(params, target_params):
                target_param.mul_(decay).add_(param, alpha=1 - decay)

    def _prepare_t_and_tau(self, bsize, device):
        t = torch.rand((bsize,), device=device)
        if self.delta_t is None:
            tau = torch.rand((bsize,), device=device)
        else:
            tau = torch.clip(t - self.delta_t, min=0, max=1)
        return t, tau

    def _diffuse_to_pair(self, x_0, t, tau):
        # NOT applicable to Lie groups
        """
        Diffuses `x_0` to two different times, sharing one noise draw.

        Samples a noise tensor and uses it to build both `x_t` and `x_tau` via
        the closed-form forward-process marginal `a(s) * x_0 + b(s) * noise`,
        evaluated at `s = t` and `s = tau`.

        Args:
            x_0 (torch.Tensor): Clean data state at time 0.
            t (torch.Tensor): First evaluation time(s), shape `(batch,)`.
            tau (torch.Tensor): Second evaluation time(s), shape `(batch,)`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: `(x_t, x_tau)`.
        """
        noise = torch.randn_like(x_0)

        t = t.view(-1, *[1] * (x_0.ndim - 1))
        tau = tau.view(-1, *[1] * (x_0.ndim - 1))

        a = self.diffuser.sde_schedule.transition_mean_scale
        b = self.diffuser.sde_schedule.transition_noise_std

        x_t = a(0, t) * x_0 + b(0, t) * noise
        x_tau = a(0, tau) * x_0 + b(0, tau) * noise

        return x_t, x_tau

    def forward(self, t, x_t):
        """Forward pass."""
        return self.consistency_fn(t, x_t)


# =============================================================================
def squared_l2_distance(a: torch.Tensor, b: torch.Tensor):
    """Default distance function: mean squared error (real part)."""
    res = a - b
    return torch.mean((res * res.conj()).real)
