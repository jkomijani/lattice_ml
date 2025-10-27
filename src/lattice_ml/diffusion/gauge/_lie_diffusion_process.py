# Created by Javad Komijani, 2024-2025

"""Implements the diffusion process."""

# pylint: disable=too-many-arguments, too-many-positional-arguments

from typing import Callable, Tuple

import torch
import numpy as np


from ._trainer import Trainer
from ._randn_xxx_like import randn_special_unitary_like
from ._lie_sdeint import integrate_sde


__all__ = ["SUnDiffusionProcess"]


# =============================================================================
class SUnDiffusionProcess:
    r"""
    Implements a diffusion process as described by the following stochastic
    differential equation (SDE)

    .. math::
        \frac{d U(t)}{dt} = \sqrt{2} \sigma e^{\gamma t} \eta(t) U(t)

    where
    - :math:`gamma > 0` controls the rate of increament in the noise variance,
    - :math:`\sigma > 0` controls the overal scale of the noise,
    - :math:`\eta(t)` is standard white Gaussian noise in the algebra space.

    The reverse process is defined by an SDE that uses a learned score function
    (the gradient of the log-probability density) to iteratively denoise the
    corrupted signal and recover clean samples.

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
        The parameter controling the rate of increament in the noise variance.

    sigma : float
        Noise scaling parameter.

    trainer : Trainer
        An associated :class:`Trainer` instance for model training.
        The alias `self.train` maps directly to `trainer.execute`.
    """

    n_random_walk_steps = 4
    last_step_h = 0.005

    def __init__(
        self, score_fn: Callable,
        gamma: float = 1,
        sigma: float = None
    ):
        r"""
        Initializes the diffusion process with a score function and an optional
        gamma value.

        Parameters
        ----------
        score_fn : Callable
            A neural network for modeling the score function.

        gamma : float, optional
            The parameter controling the rate of increament in the noise
            variance. It must be a positive number. Default is 1.

        sigma : float, optional
            The scaling factor of noise. If not provided, it will be set to
            the saure root of `gamma`. Default is None.
        """
        assert gamma > 0, "gamma must be positive"

        # Main components of the model
        self.score_fn = score_fn
        self.gamma = gamma
        self.sigma = sigma or gamma ** 0.5
        self.sigma_ratio = self.sigma / self.gamma ** 0.5

        # Components for training
        self.trainer = Trainer(self)
        self.train = self.trainer.execute

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
                More precisely We have :math:`alg_t = \int_0^t d \Gamma_t`.

            - Tensor: `std`: The accumulated standard deviation (noise) over
                time. This tensor tracks how much noise has been added to the
                algebraic state during the diffusion process.
        """
        # Expand t_eval dimensions to match y_0
        t_eval = t_eval.view(-1, *[1] * (y_0.ndim - 1))

        h = self.last_step_h
        t_1 = torch.clamp_min(t_eval - h, 0)  # max(t_eval - h, 0)

        # Comput the std of integrated noise at time t & generate noise terms
        std = self.sigma_ratio * (torch.exp(2 * self.gamma * t_1) - 1)**0.5
        n_steps = self.n_random_walk_steps
        randn_grp, randn_alg = randn_special_unitary_like(y_0, std, n_steps)

        # Simulate the diffusion process
        y_t1 = randn_grp @ y_0

        # Comput the std of integrated noise at time t & generate noise terms
        c_0 = self.sigma_ratio
        c_1 = 2 * self.gamma
        std = c_0 * np.sqrt(np.exp(c_1 * t_eval) - np.exp(c_1 * t_1))
        randn_grp, randn_alg = randn_special_unitary_like(y_0, std, n_steps=1)

        # Simulate the diffusion process
        y_t = randn_grp @ y_t1

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
        n_steps = self.n_random_walk_steps
        c_0 = self.sigma_ratio
        c_1 = 2 * self.gamma

        for ind, t in enumerate(t_eval):
            assert t >= t_0, "`t_eval` must monotonically increase."

            std = c_0 * np.sqrt(np.exp(c_1 * t) - np.exp(c_1 * t_0))
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
        step_size: float = 0.001,
        sigma_tilde: float = 1
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
        factor = np.sqrt(2) * sigma_tilde
        solver_kwargs = {
            'method': method,
            'step_size': step_size,
            'noise_scale': lambda t: factor * torch.exp(self.gamma * t)
        }

        # Build the drift function used in the reverse SDE
        drift = self.build_denoising_drift(sigma_tilde)

        y_eval = [None] * len(t_eval)

        for ind, t in enumerate(t_eval):
            assert t <= t_0, "`t_eval` must monotonically decrease."

            # Simulate the denoising process
            y_eval[ind] = integrate_sde(drift, (t_0, t), y_0, **solver_kwargs)

            # Update the state for the next round
            y_0, t_0 = y_eval[ind], t

        return y_eval[0] if squeeze_output else y_eval

    def build_denoising_drift(self, sigma_tilde: float):
        r"""
        Construct the drift function for the reverse (denoising) diffusion
        process.

        The reverse-time dynamics are governed by an SDE whose drift term
        is given by

        .. math::
            f(t, x) = - (\sigma^2 + \tilde{\sigma}^2)\, \nabla_x \log p_t(x),

        where
        - :math:`\sigma` is the forward diffusion noise scaling factor,
        - :math:`\tilde{\sigma}` is a user-specified noise scaling factor, and
        - :math:`\nabla_x \log p_t(x)` is approximated by the score function
          :math:`\text{score}(t, x)`.

        This method returns a closure ``denoising_drift(t, y_t)`` that computes
        the drift at time ``t`` for state ``y_t``. It can be passed directly to
        an SDE solver.

        Parameters
        ----------
        sigma_tilde : float
            A scaling factor for the stochastic term in the reverse process.

        Returns
        -------
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            A function ``denoising_drift(t, y_t)`` that computes the drift term
            for the reverse diffusion process at time ``t`` and state ``y_t``.
        """
        score_coeff = self.sigma ** 2 + sigma_tilde ** 2
        score_expon = 2 * self.gamma

        def denoising_drift(t: torch.Tensor, y_t: torch.Tensor):
            r"""
            Compute the drift term for the reverse diffusion process.

            Parameters
            ----------
            t : torch.Tensor
                Time tensor (scalar or shape ``[batch]``), aligned with the
                batch dimension of ``y_t``.

            y_t : torch.Tensor
                The state of the system at time ``t``. Serves as input to the
                score function and as the variable being denoised.

            Returns
            -------
            torch.Tensor
                The drift vector field at ``(t, y_t)``, to be used in an SDE
                solver.
            """
            score = self.score_fn(t, y_t)
            return - score_coeff * torch.exp(score_expon * t) * score

        return denoising_drift
