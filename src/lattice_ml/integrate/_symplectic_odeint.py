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

Main Interface
--------------
- `symplectic_odeint`: Integrates symplectic systems from (p0, q0).
- `lie_symplectic_odeint`: Integrates symplectic systems on a Lie group.
"""

# pylint: disable=too-many-arguments, too-many-positional-arguments

from typing import Callable, Tuple, Any
import torch


__all__ = ["symplectic_odeint", "lie_symplectic_odeint"]


# =============================================================================
def symplectic_odeint(
    force_fn: Callable,
    t_span: Tuple[float, float],
    p0: torch.Tensor,
    q0: torch.Tensor,
    args: Any = None,
    velocity_fn: Callable | None = None,
    step_size: float = 1e-3,
    num_steps: int | None = None,
    method: str = "leapfrog"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Solve an initial value problem for a system of symplectic ODEs.

    The system state is represented by a tuple (p, q), where p is momentum and
    q is position. The dynamics follow a canonical symplectic form:

        dq/dt = ∂H/∂p = p
        dp/dt = -∂H/∂q = force_fn(t, q, *args)

    where `force_fn` models the generalized force acting on the system.
    While the force may not be derived from a Hamiltonian, we adopt canonical
    notation for consistency with the symplectic integration literature.

    Optionally, one can override the standard velocity–momentum relation by
    supplying a custom `velocity_fn`. This allows modeling systems where
    velocity is not simply equal to momentum, e.g. in non-canonical coordinates
    or generalized mechanical systems. In this case, the position dynamics are:

        dq/dt = velocity_fn(t, p, *args)

    Parameters
    ----------
    force_fn : callable
        Function modeling momentum dynamics. Signature: force_fn(t, q, *args).

    t_span : tuple of float
        Time interval (t0, t1) for integration.

    p0 : torch.Tensor
        Initial momentum.

    q0 : torch.Tensor
        Initial position.

    args : tuple or any or None, optional
        Additional arguments passed to force_fn.

    velocity_fn: callable, optional
        Function modeling the position dynamics.
        Signature: velocity_fn(t, p, *args).
        If not provided, defaults to the canonical choice dq/dt = p.

    step_size : float, optional
        Time step size. Ignored if `num_steps` is provided.

    num_steps : int, optional
        Number of integration steps. If given, overrides `step_size`.

    method : str
        Integration method. Only "leapfrog" is supported.

    Returns
    -------
    final_p : torch.Tensor
        Final momentum after integration.

    final_q : torch.Tensor
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

    time_grid = torch.linspace(t0, t1, 1 + num_steps, device=q0.device)

    step_size = float(time_grid[1] - time_grid[0])  # Actual step size

    if velocity_fn is None:
        velocity_fn = default_velocity_fn

    # Initial half-step momentum update & full-step position update
    p = p0 + 0.5 * step_size * force_fn(time_grid[0], q0, *args)
    q = q0 + step_size * velocity_fn(time_grid[0] + step_size / 2, p, *args)

    # Intermediate, full leapfrog steps
    for t in time_grid[1:-1]:
        p = p + step_size * force_fn(t, q, *args)
        q = q + step_size * velocity_fn(t + step_size / 2, p, *args)

    # Final half-step momentum update
    p = p + 0.5 * step_size * force_fn(time_grid[-1], q, *args)

    return p, q


def default_velocity_fn(t, p, *args):  # pylint: disable=unused-argument
    """Return the standard velocity–momentum relation."""
    return p


# =============================================================================
def lie_symplectic_odeint(
    force_fn: Callable,
    t_span: Tuple[float, float],
    p0: torch.Tensor,
    q0: torch.Tensor,
    args: Any = None,
    step_size: float = 1e-3,
    num_steps: int | None = None,
    method: str = "leapfrog"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Solve an initial value problem for a symplectic system on a Lie group.

    This function is a Lie-aware variant of `symplectic_odeint`, specialized
    for systems whose positions lie in a matrix Lie group (e.g., SO(n), SU(n)),
    and whose momenta lie in the associated Lie algebra.

    Unlike `symplectic_odeint`, this function:
      - Uses the matrix exponential for position updates:
          u ← exp(Δt·p) @ u
      - Assumes dynamics are separable and compatible with Lie group structure.

    The system evolves according to:

        du/dt = ∂H/∂p u,
        dp/dt = -u⁻¹ ∂H/∂u = force_fn(t, u, *args)

    Parameters
    ----------
    force_fn : callable
        Function modeling momentum dynamics.
        Signature: force_fn(t, u, *args) → tensor in Lie algebra.

    t_span : tuple of float
        Time interval (t0, t1) for integration.

    p0 : torch.Tensor
        Initial momentum (Lie algebra element).

    q0 : torch.Tensor
        Initial position (Lie group element).

    args : any, optional
        Additional arguments passed to `force_fn`.

    step_size : float, optional
        Time step size. Ignored if `num_steps` is provided.

    num_steps : int, optional
        Number of integration steps. If given, overrides `step_size`.

    method : str, optional
        Integration method. Only "leapfrog" is supported.

    Returns
    -------
    final_p : torch.Tensor
        Final momentum (Lie algebra element) after integration.

    final_u : torch.Tensor
        Final position (Lie group element) after integration.

    See Also
    --------
    symplectic_odeint : Symplectic integrator for vector-space phase spaces.
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

    time_grid = torch.linspace(t0, t1, 1 + num_steps, device=p0.device)

    step_size = float(time_grid[1] - time_grid[0])  # Actual step size

    # Initial half-step momentum update & full-step position update
    p = p0 + 0.5 * step_size * force_fn(time_grid[0], q0, *args)
    q = torch.matrix_exp(step_size * p) @ q0

    # Intermediate, full leapfrog steps
    for t in time_grid[1:-1]:
        p = p + step_size * force_fn(t, q, *args)
        q = torch.matrix_exp(step_size * p) @ q

    # Final half-step momentum update
    p = p + 0.5 * step_size * force_fn(time_grid[-1], q, *args)

    return p, q
