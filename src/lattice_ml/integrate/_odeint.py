# Copyright (c) 2024-2025 Javad Komijani

"""
ODE Solver Utilities

This module provides a flexible implementation of an initial value problem
(IVP) solver for ordinary differential equations (ODEs), including support
for:

- Standard integration methods such as Runge-Kutta 4 (RK4) and Euler
- Custom ODE step functions via `ode_step` override
- Fixed step sizes or a specified number of integration steps
- Loss accumulation over time using Simpson's or trapezoidal rule
- Compatibility with both PyTorch tensors and NumPy arrays

Main Interface
--------------
- `odeint`: Solves dy/dt = f(t, y; args) over a time interval.
  Optionally integrates a loss term over time.

Internal Utilities
------------------
- `_integrate_with_loss`: Helper for time-integrating a loss function
  alongside state evolution.

See the `odeint` function docstring for usage details & parameter definitions.
"""

from typing import Callable, Tuple, Union
import torch
import numpy as np

TensorOrArray = Union[torch.Tensor, np.ndarray]


__all__ = ["odeint"]


# =============================================================================
def odeint(
    func: Callable,
    t_span: Tuple[float, float],
    y0: TensorOrArray,
    args: any = None,
    step_size: float = 1e-3,
    num_steps: int | None = None,
    method: str = "RK4",
    ode_step: Callable | None = None,
    loss_rate: Callable | None = None,
) -> Union[TensorOrArray, Tuple[TensorOrArray, TensorOrArray]]:
    """
    Solve an initial value problem for a system of ODEs.

    The system of ODEs is:

        dy / dt = f(t, y; p)

    where `t` is the flow time of the system, `y` is the state variable
    (a vector describing the state of the system) at time `t`, and `p` is
    a set of fixed parameters that specify the flow (dynamic) of the system.

    Parameters
    ----------
    func : callable
        Function of the form f(t, y, *args) computing dy/dt.
    t_span : tuple of float
        (t0, t1), the interval of integration.
    y0 : torch.Tensor or numpy.ndarray
        Initial state value.
    args : tuple or any or None, optional
        Additional arguments passed to `func` and `loss_rate`. Can be:
        - A tuple of arguments (e.g., (a, b))
        - A single object (e.g., a tensor or float)
        - None (treated as empty tuple)
        If a single argument is needed, `args=a` or `args=(a,)` are both
        accepted and equivalent.
    step_size : float, optional
        Tentative step size for integration (default: 1e-3).
        Ignored if `num_steps` is specified.
    num_steps : int, optional
        If provided, overrides `step_size` to use a fixed number of steps.
    method : str, optional
        Name of the integration method (default: "RK4").
        Must be one of: "RK4", "Euler".
    ode_step : callable, optional
        If provided, overrides `method` and is used as the ODE step function.
    loss_rate : callable, optional
        Optional integrand for loss accumulation. If provided, will compute
        and return the loss using Simpson's rule (if applicable).

    Returns
    -------
    final_state : same type as y0
        Final state after integration.
    loss :  torch.Tensor or numpy.ndarray (if loss_rate is given)
        Accumulated loss over the integration time.
    """
    if ode_step is None:
        ode_step = _get_ode_step_function(method)
    else:
        assert callable(ode_step), "ode_step must be a callable if provided."

    # Normalize args to a tuple
    if args is None:
        args = ()
    elif not isinstance(args, tuple):
        args = (args,)

    # Determine number of steps and time_grid
    t0, t1 = t_span

    if num_steps is None:
        num_steps = max(1, int(abs((t1 - t0) / step_size)))

    if isinstance(y0, np.ndarray):
        time_grid = np.linspace(t0, t1, 1 + num_steps)
    else:
        time_grid = torch.linspace(t0, t1, 1 + num_steps, device=y0.device)

    step_size = float(time_grid[1] - time_grid[0])  # Actual step size

    y = y0

    if loss_rate is not None:
        return _integrate_with_loss(
                ode_step, func, loss_rate, time_grid, y, step_size, args
                )

    for t in time_grid[:-1]:
        y = ode_step(func, t, y, step_size, *args)

    return y


def _integrate_with_loss(
    ode_step: Callable,
    func: Callable,
    loss_rate: Callable,
    time_grid: TensorOrArray,
    y: TensorOrArray,
    step_size: float,
    args: tuple,
) -> Tuple[TensorOrArray, TensorOrArray]:
    """
    Helper for odeint that integrates with a loss_rate term.

    This helper performs ODE integration with loss accumulation using
    the provided `loss_rate` function as the integrand. If the number
    of time points is odd, Simpson's rule is used for accuracy; otherwise,
    the trapezoidal rule is applied.

    See `odeint` for parameter descriptions and integration behavior.
    """
    n_grid = len(time_grid)
    simpsons_rule = (n_grid % 2 == 1)  # Simpson's rule requires odd points

    # Initialize loss with the loss value at the starting time point
    loss = loss_rate(time_grid[0], y, *args)

    # Loop through each time step (except the last point)
    for ind, t in enumerate(time_grid[:-1]):
        # Advance the system state by one time step
        y = ode_step(func, t, y, step_size, *args)

        # Evaluate loss integrand at the next time step
        dloss = loss_rate(time_grid[ind + 1], y, *args)

        # Apply Simpson's rule weighting if applicable
        # Simpson's rule: weights alternate between 4 and 2, starting with 4
        # (except first and last which are 1 and handled outside loop)
        if simpsons_rule and ind % 2 == 0:
            loss += 4 * dloss
        else:
            loss += 2 * dloss

    # Final correction: subtract last added dloss to reweight correctly
    # The endpoint weight will be implicitly handled below
    loss -= dloss

    # Finalize the integral value using Simpson's or trapezoidal rule
    if simpsons_rule:
        loss *= step_size / 3  # Simpson's rule scaling
    else:
        loss *= step_size / 2  # Trapezoidal rule scaling

    return y, loss


# =============================================================================
def _get_ode_step_function(method: str) -> Callable:
    if method == "RK4":
        ode_step = rk4_step
    elif method == "Euler":
        ode_step = euler_step
    else:
        raise ValueError(f"Unsupported method: {method}")
    return ode_step


def euler_step(func, t, y, dt, *args):
    """Perform a single Euler step."""
    return y + func(t, y, *args) * dt


def rk4_step(func, t, y, dt, *args):
    """Perform a single Runge-Kutta-4 step."""
    eps = dt / 2
    k_1 = func(t, y, *args)
    k_2 = func(t + eps, y + eps * k_1, *args)
    k_3 = func(t + eps, y + eps * k_2, *args)
    k_4 = func(t + dt, y + dt * k_3, *args)
    return y + (k_1 + 2 * k_2 + 2 * k_3 + k_4) * (dt / 6)
