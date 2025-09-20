# Created by Javad Komijani, 2024

"""Implements the diffusion process."""

# pylint: disable=too-many-arguments, too-many-positional-arguments

from typing import Callable, Tuple

import torch
import numpy as np


from ._trainer import Trainer
from ._sdeint import integrate_sde


__all__ = ["DiffusionProcess"]


# =============================================================================
class DiffusionProcess:
    r"""
    Implements a diffusion process as described by the following stochastic
    differential equation (SDE)

    .. math::
        \frac{d x(t)}{dt} = - \gamma x(t) + \sqrt{2} \sigma \eta(t)

    where
    - :math:`\gamma > 0` controls the drift strength,
    - :math:`\sigma = \sqrt{\gamma}`,
    - :math:`\eta(t)` is standard white Gaussian noise.

    Intuitively:
    - The drift term :math:`-\gamma x(t)` pulls the state back toward
      the origin, shrinking the signal over time.
    - The stochastic term :math:`\sigma \eta(t)` injects Gaussian noise,
      progressively corrupting the signal.

    The reverse process is defined by an SDE that uses a learned score
    function (the gradient of the log-probability density) to iteratively
    denoise the corrupted signal and recover clean samples.

    Use Cases:
    - Simulate noisy trajectories (`forward`).
    - Train score-based generative models (`run_for_training`).
    - Generate clean samples by reversing the diffusion (`reverse`).

    Attributes
    ----------
    score_fn : Callable
        A user-provided function (typically a neural network) that estimates
        the score function. It is used in the reverse diffusion process.

    gamma : float
        Drift coefficient controlling the strength of the deterministic decay.

    sigma : float
        Noise scale parameter, defined as :math:`\sigma = \sqrt{\gamma}`.

    trainer : Trainer
        An associated :class:`Trainer` instance for model training.
        The alias `self.train` maps directly to `trainer.execute`.
    """

    def __init__(self, score_fn: Callable, gamma: float = np.pi):
        r"""
        Initializes the diffusion process with a score function and an optional
        gamma value.

        Parameters
        ----------
        score_fn : Callable
            A neural network for modeling the score function.

        gamma : float, optional
            The parameter controlling the strength of the drift term in the
            diffusion process. Default is :math:`\pi`.
        """
        # Main components of the model
        self.score_fn = score_fn
        self.gamma = gamma
        self.sigma = gamma ** 0.5  # hard-wired constant based on gamma

        # Components for training
        self.trainer = Trainer(self)
        self.train = self.trainer.execute

    def run_for_training(self, x_0: torch.Tensor, t_eval: torch.Tensor):
        """
        Simulates the forward diffusion process for training purposes.

        The process evolves from the initial state `x_0` to the evaluation
        time specified by `t_evals` and returns a tuple containing all needed
        for implicit score matching.

        Parameters
        ----------
        x_0 : torch.Tensor
            The initial state of the system at time 0.

        t_eval : torch.Tensor
            A 1-dimensional tensor containing the number of evaluations times.
            Its length must match the batch size of `x_0`.

        Returns
        -------
        torch.Tensor, torch.Tensor, torch.Tensor
            A tuple containing:
            - `x_t`: the final diffused states of the system,
            - `eps`: the noise samples used in the diffusion process,
            - `std`: the (exact) standard deviation of the noise samples.
        """
        # Expand t_eval dimensions to match x_0
        t_eval = t_eval.view(-1, *[1] * (x_0.ndim - 1))

        # Compute decay and standard deviation factors
        dcy = torch.exp(-self.gamma * t_eval)  # decay factor of signal
        std = torch.sqrt(1 - dcy**2)  # std of integrated noise at time t_eval
        eps = torch.randn_like(x_0)  # normalized noise

        # Simulate the diffusion process
        x_t = dcy * x_0 + std * eps

        return x_t, eps, std

    def forward(
        self,
        x_0: torch.Tensor,
        t_0: float = 0.,
        t_eval: Tuple[float] | float = 1.0
    ):
        """
        Simulates the forward diffusion process, evolving the state `x_0` from
        initial time `t_0` to the time points specified by `t_eval`.

        Parameters
        ----------
        x_0 : torch.Tensor
            The initial state of the system at time `t_0`.

        t_0 : float, optional
            The initial time for the simulation. Default is 0.

        t_eval : Tuple of float | float
            A sequence of times at which the state `x_t` of the system is
            evaluated. The times must be monotonically increasing.

        Returns
        -------
        list of Tensor
            A list containing the states `x_t` of the system at each time
            defined in `t_eval`.
        """
        if isinstance(t_eval, (float, int)):
            t_eval = (t_eval,)
            squeeze_output = True
        else:
            squeeze_output = False

        x_eval = [None] * len(t_eval)

        for ind, t in enumerate(t_eval):
            assert t >= t_0, "`t_eval` must monotonically increase."

            # Compute decay and standard deviation factors for each time step
            dcy = np.exp(-self.gamma * (t - t_0))  # decay factor of the signal
            std = np.sqrt(1 - dcy**2)  # std of integrated noise at time t
            eps = torch.randn_like(x_0)  # generate normalized noise

            # Simulate the diffusion process
            x_eval[ind] = dcy * x_0 + std * eps

            # Update the state for the next round
            x_0, t_0 = x_eval[ind], t

        return x_eval[0] if squeeze_output else x_eval

    def reverse(
        self,
        x_0: torch.Tensor,
        t_0: float = 1.0,
        t_eval: Tuple[float] | float = 0.,
        method: str = 'Euler-Maruyama',
        step_size: float = 0.001,
        sigma_tilde: float = np.pi ** 0.5
    ):
        """
        Simulates the denoising process (reverse diffusion process) by applying
        the learned score function to gradually remove the noise and recover
        the clean signal.

        Parameters
        ----------
        x_0 : torch.Tensor
            The state of the system at time `t_0`.

        t_0 : float, optional
            The initial time for the denoising process. Default is 1.

        t_eval : Tuple of float | float
            A sequence of times at which the denoised state `x_t` of the system
            is evaluated. Times must be monotonically decreasing.

        method : str, optional
            The method for solving the SDE. Default is 'Euler-Maruyama'.

        step_size : float, optional
            The step size for discretizing the SDE. Default is 0.001.

        sigma_tilde : float, optional
            Coefficient of the noise term in the SDE (up to squared root of 2).
            Defaults to `sqrt(pi)`.

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
        solver_kwargs = {
            'method': method,
            'step_size': step_size,
            'noise_scale': np.sqrt(2) * sigma_tilde
        }

        # Build the drift function used in the reverse SDE
        drift = self.build_denoising_drift(sigma_tilde)

        x_eval = [None] * len(t_eval)

        for ind, t in enumerate(t_eval):
            assert t <= t_0, "`t_eval` must monotonically decrease."

            # Simulate the denoising process
            x_eval[ind] = integrate_sde(drift, (t_0, t), x_0, **solver_kwargs)

            # Update the state for the next round
            x_0, t_0 = x_eval[ind], t

        return x_eval[0] if squeeze_output else x_eval

    def build_denoising_drift(self, sigma_tilde: float):
        r"""
        Construct the drift function for the reverse (denoising) diffusion
        process.

        The reverse-time dynamics are governed by an SDE whose drift term
        is given by

        .. math::
            f(t, x) = -\gamma x(t) - (\sigma^2 + \tilde{\sigma}^2)\,
            \nabla_x \log p_t(x),

        where
        - :math:`\gamma` is the drift coefficient,
        - :math:`\sigma^2 = \gamma` is the forward diffusion noise variance,
        - :math:`\tilde{\sigma}` is a user-specified noise scaling factor, and
        - :math:`\nabla_x \log p_t(x)` is approximated by the score function
          :math:`\text{score}(t, x)`.

        This method returns a closure ``denoising_drift(t, x_t)`` that computes
        the drift at time ``t`` for state ``x_t``. It can be passed directly to
        an SDE solver.

        Parameters
        ----------
        sigma_tilde : float
            A scaling factor for the stochastic term in the reverse process.

        Returns
        -------
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            A function ``denoising_drift(t, x_t)`` that computes the drift term
            for the reverse diffusion process at time ``t`` and state ``x_t``.
        """
        gamma = self.gamma
        score_coeff = self.sigma ** 2 + sigma_tilde ** 2

        def denoising_drift(t: torch.Tensor, x_t: torch.Tensor):
            r"""
            Compute the drift term for the reverse diffusion process.

            Parameters
            ----------
            t : torch.Tensor
                Time tensor (scalar or shape ``[batch]``), aligned with the
                batch dimension of ``x_t``.

            x_t : torch.Tensor
                The state of the system at time ``t``. Serves as input to the
                score function and as the variable being denoised.

            Returns
            -------
            torch.Tensor
                The drift vector field at ``(t, x_t)``, to be used in an SDE
                solver.
            """
            score = self.score_fn(t, x_t)
            return -gamma * x_t - score_coeff * score

        return denoising_drift
