# Copyright (c) 2025 Javad Komijani

"""
This module provides tools for integrating stochastic differential equations
(SDEs) using the Euler–Maruyama method. It includes a callable integrator
function (`integrate_sde`) and a PyTorch module (`SDEIntegrator`) for evolving
systems governed by SDEs within neural network pipelines.

Typical use cases include simulating noisy dynamical systems, modeling
stochastic processes, or embedding SDE integration into differentiable models.
"""

import functools
from typing import Callable, Tuple

import torch


__all__ = ["integrate_sde", "SDEIntegrator", "euler_maruyama_step"]


# =============================================================================
class SDEIntegrator(torch.nn.Module):
    """
    A PyTorch module that integrates a system governed by a stochastic
    differential equation (SDE) from an initial state over a time span.
    """
    def __init__(
        self,
        func: Callable,
        t_span: Tuple[float, float],
        **solver_kwargs
    ):
        """
        Initialize the SDEIntegrator.

        Parameters
        ----------
        func : callable
            The drift function f(t, y, *args) defining the deterministic
            part of the SDE.
        t_span : tuple of float
            A tuple (t0, t1) defining the integration interval.
        **solver_kwargs :
            Additional keyword arguments passed to the integrate_sde function,
            such as step_size, noise_scale, and method.
        """
        super().__init__()
        self.func = func
        self.t_span = t_span
        self.integrate_sde = functools.partial(integrate_sde, **solver_kwargs)

    def forward(self, y0: torch.Tensor, args: Tuple = None) -> torch.Tensor:
        """
        Integrates the SDE from initial to final time in t_span.

        Parameters
        ----------
        y0 : torch.Tensor
            Initial state variable at the start of integration.
        args : tuple, optional
            Additional arguments to pass to the drift function.

        Returns
        -------
        torch.Tensor
            Final state after integration.
        """
        return self.integrate_sde(self.func, self.t_span, y0, args=args)


# =============================================================================
def integrate_sde(
    func: Callable,
    t_span: Tuple[float, float],
    y0: torch.Tensor,
    args: any = None,
    noise_scale: float = 1.0,
    step_size: float = 1e-3,
    num_steps: int = None,
    method: str = "Euler-Maruyama",
    sde_step: Callable | None = None,
) -> torch.Tensor:
    """
    Solves an initial value problem for a system of stochastic differential
    equations (SDEs).

    The SDE system is of the form:
        dy/dt = f(t, y; p) + noise(t)
    where f is the deterministic drift term and noise(t) is Gaussian noise
    scaled by `noise_scale`.

    Parameters
    ----------
    func : Callable(t, y, *args) -> Tensor
        Function computing the deterministic drift f(t, y; p).
    t_span : Tuple[float, float]
        Start and end time of the integration (t0, t1).
    y0 : torch.Tensor
        Initial state at t = t0.
    args : tuple or any or None, optional
        Additional arguments passed to `func`. Can be:
        - A tuple of arguments (e.g., (a, b))
        - A single object (e.g., a tensor or float)
        - None (treated as empty tuple)
        If a single argument is needed, `args=a` or `args=(a,)` are both
        accepted and equivalent.
    noise_scale : float, optional
        Scale of additive Gaussian noise (default: 1.0).
    step_size : float, optional
        Step size for time discretization (default: 1e-3).
    num_steps : int, optional
        If specified, overrides step_size to fix total number of steps.
    method : str, optional
        Name of the integration method (default: "Euler-Maruyama").
    sde_step : callable, optional
        If provided, overrides `method` and is used as the SDE step function.

    Returns
    -------
    torch.Tensor
        Final state after integration.
    """

    # Select integration method
    if sde_step is None:
        sde_step = _get_sde_step_function(method)
    else:
        assert callable(sde_step), "sde_step must be a callable if provided."

    # Normalize args to a tuple
    if args is None:
        args = ()
    elif not isinstance(args, tuple):
        args = (args,)

    t0, t1 = t_span

    # Determine number of steps
    if num_steps is None:
        num_steps = max(1, int(abs((t1 - t0) / step_size)))

    # Generate a uniform time grid
    time_grid = torch.linspace(t0, t1, num_steps + 1, device=y0.device)

    step_size = float(time_grid[1] - time_grid[0])  # Actual step size

    y = y0

    # Integrate over the time grid using the chosen method
    if callable(noise_scale):
        for t in time_grid[:-1]:
            y = sde_step(func, t, y, step_size, noise_scale(t), *args)
    else:
        for t in time_grid[:-1]:
            y = sde_step(func, t, y, step_size, noise_scale, *args)

    return y


# =============================================================================
def _get_sde_step_function(method: str) -> Callable:
    if callable(method):
        sde_step = method
    elif method == 'Euler-Maruyama':
        sde_step = euler_maruyama_step
    else:
        raise ValueError(f"Unsupported integration method: {method}")
    return sde_step


def euler_maruyama_step(
    func: Callable,
    t: float,
    y: torch.Tensor,
    dt: float,
    noise_scale: float,
    *args
) -> torch.Tensor:
    """
    Perform a single Euler–Maruyama integration step.

    Approximates the solution of a stochastic differential equation (SDE)
    over a small time increment dt.

    Parameters
    ----------
    func : callable
        Function computing the deterministic derivative f(t, y, *args).
    t : float
        Current time.
    y : torch.Tensor
        Current state variable.
    dt : float
        Time step for this update.
    noise_scale : float
        Coefficient scaling the stochastic term.
    *args :
        Additional arguments passed to func.

    Returns
    -------
    torch.Tensor
        Updated state after a single Euler–Maruyama step.
    """
    # Deterministic and stochastic increments
    drift = dt * func(t, y, *args)
    diffusion = (abs(dt)**0.5 * noise_scale) * torch.randn_like(y)

    return y + drift + diffusion
