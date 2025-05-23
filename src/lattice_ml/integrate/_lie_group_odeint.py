# Copyright (c) 2024-2025 Javad Komijani

"""
Lie group ODE integration methods.

This module provides functionality for numerically integrating ordinary
differential equations (ODEs) defined on Lie groups, such as SU(n).
It includes the `lie_odeint` function, which acts as a wrapper around
a general-purpose ODE integrator (`odeint`) by selecting Lie-group-aware
integration methods and passing them as custom step functions.

Available integration methods include:
- RK4:SU(n): Standard Runge-Kutta 4 on SU(n)
- RK4:SU(n):aug: Augmented RK4 method
- RK3:auto: Third-order autonomous RK on Lie groups
- Euler: Basic Euler method adapted to Lie groups
"""

from typing import Callable, Tuple, Union
import torch

from ._odeint import odeint
from ._adjoint import TupleVar

__all__ = ["lie_odeint"]


# =============================================================================
def lie_odeint(
    func: Callable,
    t_span: Tuple[float, float],
    y0: torch.Tensor,
    method: str = "RK4:SU(n)",
    **odeint_kwargs
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""
    Integrate a system of ODEs on a Lie group using a Lie-specific step method.

    This function solves differential equations of the form:

        dU / dt = f(t, U; p) = F(t, U; p) @ U

    where `U = U(t)` evolves on a Lie group (e.g., SU(n)) and `F(t, U; p)`
    belongs to its Lie algebra. Here, `t` is time and `p` represents
    fixed parameters defining the system dynamics.

    This is a wrapper around `odeint` that selects a Lie-group-aware method of
    integration and passes it as the `ode_step` argument. All other options are
    forwarded to `odeint`.

    Parameters
    ----------
    func : callable
        Computes the right-hand side of the ODE, i.e., `f(t, U; p)` or
        `F(t, U; p)` depending on the method.
    t_span : tuple of float
        (t0, t1), the time interval to integrate over.
    y0 : torch.Tensor
        Initial state on the Lie group (e.g., an SU(n) matrix).
    method : str, optional
        Name of the Lie group integration method. Determines which ODE step
        function is passed to `odeint` via `ode_step`. Default is "RK4:SU(n)".
    **odeint_kwargs : dict
        Additional keyword arguments passed to `odeint`, such as `args`,
        `step_size`, `num_steps`, or `loss_rate`.

    Returns
    -------
    final_state : same type as `y0`
        Final state after integration.
    loss : torch.Tensor, optional
        Accumulated loss if `loss_rate` is provided.
    """
    ode_step = _get_lie_ode_step(method)
    return odeint(func, t_span, y0, ode_step=ode_step, **odeint_kwargs)


# =============================================================================
def _get_lie_ode_step(method: str) -> Callable:
    """Return the appropriate Lie group ODE step function based on method."""
    if method == 'RK4:SU(n)':
        ode_step = special_unitary_rk4_step
    elif method == 'RK4:SU(n):aug':
        ode_step = augmented_special_unitary_rk4_step
    elif method == 'RK3:auto':
        ode_step = lie_autonomous_rk3_step
    elif method == 'Euler':
        ode_step = lie_euler_step
    else:
        raise ValueError(f"Lie group method '{method}' is not implemented.")

    return ode_step


# =============================================================================
def augmented_special_unitary_rk4_step(func, t, var, dt, *args):
    r"""
    Generalized RK4 step for special unitary matrices with frozen variables.

    This is an extension of `special_unitary_rk4_step` that supports
    augmented systems where the primary variable evolves on the Lie group
    SU(n) and additional auxiliary variables are updated alongside with RK4
    method.

    Specifically, the function integrates systems of the form:

        dU/dt = F(t, {U, z}; p) @ U,    where U ∈ SU(n),
        dz/dt = g(t, {U, z}; p),    where z may be frozen or auxiliary.

    The update for `U` is projected back onto SU(n) using a truncated
    matrix exponential, ensuring group structure is preserved. The update
    for the auxiliary variable is handled additively (no projection).

    This method assumes that `var` is a `TupleVar(U, z)` and that
    `func(t, var, *args)` returns a `TupleVar(dU, dz)`.

    Parameters
    ----------
    func : callable
        The ODE function returning a `TupleVar` of updates `(dU, dz)`.
    t : float
        Current integration time.
    var : TupleVar
        Current state of the system, must be `TupleVar(U, z)` where
        `U` ∈ SU(n) and `z` is auxiliary.
    dt : float
        Time step.
    *args : tuple
        Optional additional arguments passed to `func`.

    Returns
    -------
    TupleVar
        Updated state `TupleVar(U_next, z_next)` after one RK4 step.
        `U_next` remains on SU(n).
    """
    # Compute RK4 deltas for SU(n) and auxiliary components
    delta, d_other = delta_from_rk4_step(func, t, var, dt, *args).tuple

    # Unpack current SU(n) variable and auxiliary variable
    var, other = var.tuple

    # Update SU(n) variable with projection to remain on the manifold
    var = construct_rk4_special_unitary(delta @ var.adjoint()) @ var

    # Update auxiliary variable via standard RK4 increment
    other = other + d_other

    return TupleVar(var, other)


# =============================================================================
def special_unitary_rk4_step(func, t, var, dt, *args):
    r"""
    Perform a single Runge-Kutta-4 step while preserving the special unitary
    property.

    This method evolves `var` in time using the classical RK4 integration
    scheme, then projects the update back onto the special unitary group SU(n).

    The RK4 method gives

    .. math::

         U_{t + dt} = U_t + {shift} + O(h^5) = (I + \delta + O(h^5)) U_t

    We rewrite the coefficient of `U_t` as a special unitary matrix.
    Then, we multiply it to the current value `U(t)` to obtain `U(t + dt)`.

    Parameters
    ----------
    func : callable
        Function of the form `f(t, U, *args)`.
    t : float
        Current time.
    var : torch.Tensor
        The current special unitary matrix `U` of shape (..., n, n).
    dt : float
        Time step for the integration.
    *args : tuple
        Additional arguments passed to `func`.

    Returns
    -------
    torch.Tensor
        Updated special unitary matrix after a single RK4 step.
    """
    eps = delta_from_rk4_step(func, t, var, dt, *args) @ var.adjoint()
    return construct_rk4_special_unitary(eps) @ var


def construct_rk4_special_unitary(eps):
    r"""Project `I + \epsilon + O(\epsilon^5)` to a special unitary matrix."""
    dummy = eps
    exponent = dummy
    for power in range(2, 5):
        # power of 5 and larger powers are not needed as error is O(eps^5)
        dummy = dummy @ (-eps)
        exponent = exponent + dummy / power
    return torch.matrix_exp(anti_hermitian_traceless(exponent))


def delta_from_rk4_step(func, t, var, dt, *args):
    """Calculate the shift in var obtained from a single Runge-Kutta-4 step."""
    half_dt = dt / 2
    k_1 = func(t, var, *args)
    k_2 = func(t + half_dt, var + half_dt * k_1, *args)
    k_3 = func(t + half_dt, var + half_dt * k_2, *args)
    k_4 = func(t + dt, var + dt * k_3, *args)
    return (k_1 + 2 * k_2 + 2 * k_3 + k_4) * (dt / 6)


def anti_hermitian_traceless(mtrx: torch.Tensor) -> torch.Tensor:
    """
    Project a square matrix (or batch of matrices) onto the Lie algebra su(n).

    This function returns an anti-Hermitian, traceless version of the input
    matrix `mtrx`.
    """
    # Make anti-Hermitian
    mtrx = (mtrx - mtrx.adjoint()) / 2.

    # Compute average diagonal value (trace / n) over the last two axes
    reduced_trace = mtrx.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True)

    # Subtract the average from the diagonal to make it traceless
    return mtrx - torch.diag_embed(reduced_trace.expand(mtrx.shape[:-1]))


# =============================================================================
def lie_autonomous_rk3_step(algebra_func, t, var, dt, *args):
    r"""
    Performs one step of a 3-stage exponential Lie group integrator.

    This method is based on the Appendix C of [arXiv:1006.4518], and is
    designed for integrating autonomous systems of the form:

        dU / dt = F(t, U; p) U,

    where `U(t)` evolves on a Lie group (e.g., SU(n)), and `F(t, U; p)` lies in
    the corresponding Lie algebra (e.g., su(n)). The flow is assumed autonomous
    (i.e., time-invariant), so `t` serves as a dummy variable.

    Parameters
    ----------
    algebra_func : callable
        Function of the form `F(t, U, *args)` returning an element of the Lie
        algebra that defines the flow.
    t : float
        Time (a dummy argument in the autonomous case).
    var : torch.Tensor
        Current state matrix `U` in the Lie group, shape (..., n, n).
    dt : float
        Time step.
    *args : tuple
        Additional parameters to pass to `algebra_func`.

    Returns
    -------
    torch.Tensor
        Updated Lie group element.
    """
    for ind in range(3):
        func_value = algebra_func(t, var, *args)  # F(t, U; p)

        # Construct the algebra increment zee for each sub-step
        # Note: zee = eps * Z defined in (C.1) & (1.4) of [arXiv:1006.4518]
        if ind == 0:
            zee = (1 / 4 * dt) * func_value
        elif ind == 1:
            zee = (8 / 9 * dt) * func_value - (17 / 9) * zee
        else:
            zee = (3 / 4 * dt) * func_value - zee

        # Apply exponential map to update the Lie group variable
        var = torch.matrix_exp(zee) @ var

    return var


# =============================================================================
def lie_euler_step(algebra_func, t, var, dt, *args):
    """Perform a single Euler step for unitary matrices."""
    return torch.matrix_exp(algebra_func(t, var, *args) * dt) @ var
