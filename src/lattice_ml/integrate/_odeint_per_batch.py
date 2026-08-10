# Copyright (c) 2026 Javad Komijani

"""
ODE integration with per-sample (batched) integration times.

Unlike `odeint`/`lie_odeint`, which integrate a single, shared trajectory
`t0 -> t1` for an entire batch, `odeint_per_batch` allows each sample in the
batch to have its own `t0`/`t1` (e.g. one random diffusion time per sample,
as needed for consistency-model training).

Because `t0`/`t1` may vary per sample, there is no shared time grid to build
via `torch.linspace`; instead, `num_steps` uniform sub-steps of size
`dt = (t1 - t0) / num_steps` are taken per sample. This means `t_eval`,
`loss_rate`, and `corrector` (all of which assume a single, shared time grid)
are not supported here -- use `odeint`/`lie_odeint` for those.
"""

from typing import Callable, Tuple, Union

import torch


TimeOrPerSample = Union[float, torch.Tensor]

__all__ = ["odeint_per_batch"]


# =============================================================================
def odeint_per_batch(
    func: Callable,
    t_span: Tuple[TimeOrPerSample, TimeOrPerSample],
    y0: torch.Tensor,
    num_steps: int = 1000,
    method: str = "RK4",
    args: any = None,
) -> torch.Tensor:
    """
    Integrate `dy/dt = func(t, y)` (or, for Lie-group states,
    `dy/dt = func(t, y) @ y`) over `t_span`, where each item of `t_span` may
    hold one value per sample instead of a single shared time.

    Parameters
    ----------
    func : callable
        `f(t, y, *args)` computing `dy/dt` (scalar `y`), or the algebra-valued
        generator with `dy/dt = f(t, y, *args) @ y` (Lie-group `y`). Note that
        `func` receives `t` already reshaped to broadcast against `y`.
    t_span : tuple of (float or torch.Tensor)
        `(t0, t1)`. Each of `t0`, `t1` is a float, or a 1d tensor with one
        value per sample (matching `y0`'s batch size).
    y0 : torch.Tensor
        Initial state.
    num_steps : int, optional
        Number of uniform sub-steps per sample (default: 100).
    method : str, optional
        "RK4" or "Euler" for a scalar-valued state; "Euler:g" for a Lie-group
        state with algebra-valued `func`. Default is "RK4".
    args : tuple or any or None, optional
        Additional arguments passed to `func`, as in `odeint`.

    Returns
    -------
    torch.Tensor: Final state at `t1`, same shape as `y0`.

    Raises
    ------
    ValueError: If `t0` or `t1` is neither a float nor broadcastable to `y0`.
    """
    ode_step = _get_ode_step_function(method)

    if args is None:
        args = ()
    elif not isinstance(args, tuple):
        args = (args,)

    t0, t1 = t_span
    t0 = _as_broadcastable_time(t0, y0, "t_span[0]")
    t1 = _as_broadcastable_time(t1, y0, "t_span[1]")
    dt = (t1 - t0) / num_steps

    t, y = t0, y0
    for _ in range(num_steps):
        y = ode_step(func, t, y, dt, *args)
        t = t + dt

    return y


def _as_broadcastable_time(
    t: TimeOrPerSample, y0: torch.Tensor, name: str
) -> float | torch.Tensor:
    """
    Return `t` broadcastable to `y0`, requiring it be a float or a 1d
    tensor matching `y0`'s batch size. A float `t` is returned unchanged
    (not converted to a tensor).
    """
    if not isinstance(t, torch.Tensor):
        return float(t)
    if t.ndim == 0:
        return t
    if t.shape[0] == y0.shape[0] and t.ravel().shape[0] == y0.shape[0]:
        return t.view(-1, *[1] * (y0.ndim - 1))
    raise ValueError(f"`{name}` must be a float or broadcastable to `y0`.")


def _get_ode_step_function(method: str) -> Callable:
    if method == "RK4":
        return rk4_step
    if method == "Euler":
        return euler_step
    if method == "Euler:g":
        return lie_euler_algebra_step
    raise ValueError(f"Unsupported method: {method}")


def euler_step(func, t, y, dt, *args):
    """
    Perform a single Euler step. `t`/`dt` are assumed already broadcastable
    to `y` (see `odeint_per_batch`).
    """
    return y + func(t, y, *args) * dt


def rk4_step(func, t, y, dt, *args):
    """
    Perform a single Runge-Kutta-4 step. `t`/`dt` are assumed already
    broadcastable to `y` (see `odeint_per_batch`).
    """
    eps = dt / 2
    k_1 = func(t, y, *args)
    k_2 = func(t + eps, y + eps * k_1, *args)
    k_3 = func(t + eps, y + eps * k_2, *args)
    k_4 = func(t + dt, y + dt * k_3, *args)
    return y + (k_1 + 2 * k_2 + 2 * k_3 + k_4) * (dt / 6)


def lie_euler_algebra_step(algebra_func, t, var, dt, *args):
    """
    Perform a single Euler step on the group. `t`/`dt` are assumed already
    broadcastable to `var` (see `odeint_per_batch`).
    """
    return torch.matrix_exp(algebra_func(t, var, *args) * dt) @ var
