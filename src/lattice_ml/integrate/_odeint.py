# Copyright (c) 2024-2025 Javad Komijani

"""
ODE Solver Utilities

This module provides a flexible implementation of an initial value problem
(IVP) solver for ordinary differential equations (ODEs), including support
for:

- Standard integration methods such as Runge-Kutta 4 (RK4) and Euler
- Custom ODE step functions via `ode_step` override
- Fixed step sizes or a specified number of integration steps
- Optional integration of a quantity (e.g., Jacobian) along the ODE trajectory
- Compatibility with both PyTorch tensors and NumPy arrays

Main Interface
--------------
- `odeint`: Solves dy/dt = f(t, y; args) over a time interval.
  Optionally accumulates a loss function along the ODE trajectory.

See the `odeint` function docstring for usage details & parameter definitions.
"""

from typing import Callable, Tuple, Union, Any
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
    t_eval: Tuple[float] | None = None,
    fn_eval: Callable | None = None,
    loss_rate: Callable | None = None,
    corrector: Callable | None = None
) -> Union[TensorOrArray, Tuple[TensorOrArray, Any]]:
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
    t_eval : tuple of float, optional
        Times within `t_span` at which to evaluate the solution. Must be
        strictly monotonic. If omitted, only the final state is returned.
    fn_eval : callable, optional
        Function applied to the system state at each `t_eval`. Should have the
        form `fn_eval(y)`. If None, the raw states `y(t_eval)` are returned.
    loss_rate : callable, optional
        A function that computes an instantaneous quantity to be integrated
        along the ODE trajectory. If provided, it is called at each integration
        step with the signature: `loss_rate(t, y, *args)`. The function should
        return a scalar or a summable object, which is accumulated over time.
        The total loss is then computed using Simpson's rule (if applicable),
        or the trapezoidal rule as a fallback.

        Typical uses include log-Jacobian terms in KL divergence losses or
        gradient accumulation terms in adjoint sensitivity analysis.

    corrector: callable, optional
        Optional corrector function applied after each call to `ode_step`,
        except for the final call, to refine the state without advancing
        time.

        The corrector is intended to reduce discretization error or improve
        sample quality, as in predictor–corrector schemes.
        It uses the updated time and state returned by `ode_step`.
        The function must have signature: `corrector(t, y) -> y_corrected`.

    Returns
    -------
    final_state : same type as y0
        Final state after integration.
    loss : torch.Tensor or numpy.ndarray (if loss_rate is given)
        Accumulated loss over the integration time.
    """
    # Determine ode_step if not explicitly given
    if ode_step is None:
        ode_step = _get_ode_step_function(method)
    else:
        assert callable(ode_step), "ode_step must be a callable if provided."

    # Normalize args to a tuple
    if args is None:
        args = ()
    elif not isinstance(args, tuple):
        args = (args,)

    t0, t1 = t_span

    # Determine number of steps from step size if not explicitly given
    if num_steps is None:
        num_steps = max(1, int(abs((t1 - t0) / step_size)))

    # Use the appropriate time grid depending on tensor type
    if isinstance(y0, np.ndarray):
        time_grid = np.linspace(t0, t1, 1 + num_steps)
    else:
        time_grid = torch.linspace(t0, t1, 1 + num_steps, device=y0.device)

    step_size = float(time_grid[1] - time_grid[0])  # Actual step size

    y = y0

    if corrector is not None:
        return _integrate_with_corrector(
                ode_step, func, corrector, time_grid, y, step_size, args,
                t_eval, fn_eval
                )
    if loss_rate is not None:
        return _integrate_with_loss(
                ode_step, func, loss_rate, time_grid, y, step_size, args
                )
    if t_eval is not None:
        return _integrate_with_eval(
                ode_step, func, time_grid, y, step_size, args, t_eval, fn_eval
                )

    for t in time_grid[:-1]:
        y = ode_step(func, t, y, step_size, *args)

    return y


def _integrate_with_corrector(
    ode_step: Callable,
    func: Callable,
    corrector: Callable,
    time_grid: TensorOrArray,
    y: TensorOrArray,
    step_size: float,
    args: Tuple,
    t_eval: Tuple[float] | None = None,
    fn_eval: Callable | None = None
) -> Union[TensorOrArray, Tuple[TensorOrArray, ...]]:
    """
    Helper for odeint that integrates using a predictor–corrector scheme.

    Each step advances the state using `ode_step` (predictor). After each
    call to `ode_step`, except for the final call, `corrector` is applied to
    refine the state without advancing time.

    If `t_eval` is None, only the final corrected state is returned. Otherwise,
    intermediate states are recorded (t_eval is used only as a flag). When
    recording, both the predicted and corrected states are included in order.
    If `fn_eval` is provided, it is applied before storing.

    See `odeint` for parameter details.
    """

    if t_eval is None:
        # All predictor–corrector steps except the final one
        for t, next_t in zip(time_grid[:-2], time_grid[1:-1]):
            y = ode_step(func, t, y, step_size, *args)
            y = corrector(next_t, y)

        t = time_grid[-2]
        y = ode_step(func, t, y, step_size, *args)
        return y

    # t_eval is irrelevant and used only as a flag
    out_eval = []  # Stores states (or fn_eval outputs)
    out_eval.append(y if fn_eval is None else fn_eval(y))  # initial state

    # All predictor–corrector steps except the final one
    for t, next_t in zip(time_grid[:-2], time_grid[1:-1]):
        y = ode_step(func, t, y, step_size, *args)
        out_eval.append(y if fn_eval is None else fn_eval(y))
        y = corrector(next_t, y)
        out_eval.append(y if fn_eval is None else fn_eval(y))

    t = time_grid[-2]
    y = ode_step(func, t, y, step_size, *args)
    out_eval.append(y if fn_eval is None else fn_eval(y))
    return tuple(out_eval)


def _integrate_with_eval(
    ode_step: Callable,
    func: Callable,
    time_grid: TensorOrArray,
    y: TensorOrArray,
    step_size: float,
    args: Tuple,
    t_eval: Tuple,
    fn_eval: Callable | None,
) -> Tuple[TensorOrArray]:
    """
    Integrate an ODE along a monotonic, uniform time grid and record values at
    specified evaluation times `t_eval`. The time grid is assumed monotonic
    (either increasing or decreasing) with constant increments `step_size`.

    Requirements:
      * `time_grid` has at least two points and is uniformly spaced.
      * `t_eval` is monotonic and lies entirely within `time_grid`; onsecutive
        values in `t_eval` differ by at least `step_size` in magnitude.

    See `odeint` for parameter details.
    """
    if t_eval[0] < torch.min(time_grid) or t_eval[0] > torch.max(time_grid):
        raise ValueError("t_eval must be monotonic and within time_grid.")

    for i in range(len(t_eval) - 1):
        if torch.round((t_eval[i+1] - t_eval[i]) / step_size) < 1:
            raise ValueError("Increament in t_eval is smaller than step size!")

    out_eval = []  # Stores evaluated states or fn_eval outputs
    ind_eval = 0  # Index for current evaluation time

    for t in time_grid:
        # Stop once all evaluation times have been processed
        if len(t_eval) <= ind_eval:
            break

        # Distance from current grid time to next evaluation time
        delta_t = t_eval[ind_eval] - t

        # Detect whether t_eval falls within the current step interval.
        # Condition holds if:
        #   step_size > 0 and delta_t <= step_size, or
        #   step_size < 0 and delta_t >= step_size.
        if step_size * (delta_t - step_size) <= 0:
            # Take a (partial) step to exactly reach t_eval & recoder result
            y_eval_ = ode_step(func, t, y, delta_t, *args)
            out_eval.append(y_eval_ if fn_eval is None else fn_eval(y_eval_))
            ind_eval += 1

        # Advance one full step unless the partial step already did so.
        # If delta_t == step_size, the partial step exactly equals one full
        # grid increment, so reuse y_eval_ instead of integrating again
        if delta_t == step_size:
            y = y_eval_
        else:
            y = ode_step(func, t, y, step_size, *args)

    return tuple(out_eval)


def _integrate_with_loss(
    ode_step: Callable,
    func: Callable,
    loss_rate: Callable,
    time_grid: TensorOrArray,
    y: TensorOrArray,
    step_size: float,
    args: tuple,
) -> Tuple[TensorOrArray, Any]:
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
