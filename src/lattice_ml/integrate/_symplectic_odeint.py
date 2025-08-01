# Copyright (c) 2025 Javad Komijani

"""
Symplectic ODE Solver Utilities

This module provides a symplectic initial value problem (IVP) solver for
Hamiltonian systems. The integration is performed using symplectic methods,
which are well-suited for conservative dynamical systems due to their
structure-preserving properties.

Currently, only the Leapfrog (Störmer-Verlet) integrator is implemented.
Leapfrog is commonly used in Hamiltonian Monte Carlo and physical simulations
because it is time-reversible and preserves the symplectic geometry of the
system. This leads to stable long-term behavior and approximate energy
conservation.

Features:
- Symplectic integration with fixed step size
- Supports PyTorch tensors and NumPy arrays

Main Interface
--------------
- `symplectic_odeint`: Integrates Hamiltonian system from (p0, q0).
- `symplectic_odeint_safe`: non-in-place, autograd-safe implementation.
"""

from typing import Callable, Tuple, Union
import torch
import numpy as np

TensorOrArray = Union[torch.Tensor, np.ndarray]


__all__ = ["symplectic_odeint", "symplectic_odeint_safe"]


# =============================================================================
def symplectic_odeint(
    force_fn: Callable,
    t_span: Tuple[float, float],
    p0: TensorOrArray,
    q0: TensorOrArray,
    args: any = None,
    step_size: float = 1e-3,
    num_steps: int | None = None,
    method: str = "leapfrog"
) -> Tuple[TensorOrArray, TensorOrArray]:
    """
    Solve an initial value problem for a system of symplectic ODEs.

    The system state is represented by a tuple (p, q), where p is momentum and
    q is position. The dynamics follow a canonical symplectic form:

        dq/dt = ∂H/∂p = p
        dp/dt = -∂H/∂q = force_fn(t, q, *args)

    where `force_fn` models the generalized force acting on the system. While
    it may not explicitly derive from a Hamiltonian, we adopt canonical
    notation for consistency with the symplectic integration literature.

    Parameters
    ----------
    force_fn : callable
        Function modeling the momentum dynamics, typically -∂H/∂q.
        Signature: force_fn(t, q, *args), where t is time, q is position, and
        args are optional frozen arguments.

    t_span : tuple of float
        The time interval (t0, t1) over which to integrate.

    p0 : TensorOrArray
        Initial mementum.

    q0 : TensorOrArray
        Initial position.

    args : tuple or any or None, optional
        Additional arguments passed to force_fn.

    step_size : float, optional
        Time step size.

    num_steps : int, optional
        If provided, overrides step_size with fixed number of steps.

    method : str
        Integration method ("leapfrog" supported).

    Returns
    -------
    final_p : TensorOrArray
        Final momentum after integration.

    final_q : TensorOrArray
        Final position after integration.
    """

    assert method == "leapfrog", "Other methods are not supported yet."

    # Normalize args to a tuple
    if args is None:
        args = ()
    elif not isinstance(args, tuple):
        args = (args,)

    # Unpack t_span and initial momentum & position
    t0, t1 = t_span

    # Determine number of steps from step size if not explicitly given
    if num_steps is None:
        num_steps = max(1, int(abs((t1 - t0) / step_size)))

    # Use the appropriate time grid depending on tensor type
    if isinstance(q0, np.ndarray):
        time_grid = np.linspace(t0, t1, 1 + num_steps)
    else:
        time_grid = torch.linspace(t0, t1, 1 + num_steps, device=q0.device)

    step_size = float(time_grid[1] - time_grid[0])  # Actual step size

    # Initial half-step momentum update & full-step position update
    p = p0 + 0.5 * step_size * force_fn(time_grid[0], q0, *args)
    q = q0 + step_size * p

    # Intermediate, full leapfrog steps
    for t in time_grid[1:-1]:
        p += step_size * force_fn(t, q, *args)
        q += step_size * p

    # Final half-step momentum update
    p += 0.5 * step_size * force_fn(time_grid[-1], q, *args)

    return p, q


# =============================================================================
def symplectic_odeint_safe(
    force_fn: Callable,
    t_span: Tuple[float, float],
    p0: TensorOrArray,
    q0: TensorOrArray,
    args: any = None,
    step_size: float = 1e-3,
    num_steps: int | None = None,
    method: str = "leapfrog",
    return_trajectory: bool = False
) -> Tuple[TensorOrArray, TensorOrArray]:
    """
    A non-in-place, autograd-safe implementation of `symplectic_odeint`.

    This variant is intended for testing, debugging, and analysis, and
    optionally supports trajectory recording.

    See `symplectic_odeint` for full usage details.
    """

    assert method == "leapfrog", "Other methods are not supported yet."

    # Normalize args to a tuple
    if args is None:
        args = ()
    elif not isinstance(args, tuple):
        args = (args,)

    # Unpack t_span and initial momentum & position
    t0, t1 = t_span

    # Determine number of steps from step size if not explicitly given
    if num_steps is None:
        num_steps = max(1, int(abs((t1 - t0) / step_size)))

    # Use the appropriate time grid depending on tensor type
    if isinstance(q0, np.ndarray):
        time_grid = np.linspace(t0, t1, 1 + num_steps)
    else:
        time_grid = torch.linspace(t0, t1, 1 + num_steps, device=q0.device)

    step_size = float(time_grid[1] - time_grid[0])  # Actual step size

    # Initialize state
    p, q = p0, q0

    # Optional trajectory tracking
    if return_trajectory:
        p_list = [None] * (num_steps + 2)
        q_list = [None] * (num_steps + 1)
        p_list[0] = p
        q_list[0] = q

    # Main Leapfrog integration loop
    for i, t in enumerate(time_grid[:-1]):
        # Kick: update p by half-step on the first step, full-step afterward
        p = p + (1 if i > 0 else 0.5) * step_size * force_fn(t, q, *args)
        q = q + step_size * p

        if return_trajectory:
            p_list[i + 1] = p
            q_list[i + 1] = q

    # Final half-step for momentum
    p = p + 0.5 * step_size * force_fn(time_grid[-1], q, *args)

    if return_trajectory:
        p_list[-1] = p
        return p_list, q_list

    return p, q
