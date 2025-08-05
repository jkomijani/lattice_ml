# Copyright (c) 2025 Javad Komijani

"""
Symplectic ODE Flow Modules

This module defines PyTorch `nn.Module` classes for integrating Hamiltonian
systems using symplectic integrators. These flows preserve key physical
properties such as volume and energy, making them well-suited for conservative
dynamical systems and Hamiltonian neural networks.

Currently, integration is performed using the Leapfrog method via
`symplectic_odeint`.

Classes
-------
- SymplecticODEFlow: Evolves (p, q) state variables via symplectic integration.
- SymplecticODEFlow_: Same as `SymplecticODEFlow`, but also returns zero log-J.
"""


from functools import partial as ftpartial
import torch

from ._symplectic_odeint import symplectic_odeint
from ._symplectic_adjoint import adjoint_symplectic_odeint


__all__ = ["SymplecticODEFlow", "SymplecticODEFlow_"]


class SymplecticODEFlow(torch.nn.Module):
    """
    A PyTorch module for evolving a symplectic system of ODEs, with optional
    adjoint-based gradient computation.

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
        Defines the dynamics of the momentum, typically representing the force
        as -∂H/∂q. It takes the form `force_fn(t, q, *args)`, where t is time,
        q is position, and args are optional frozen arguments.

    t_span : tuple of float
        Time interval (t0, t1) over which to evolve the system.

    adjoint_method : bool, optional
        If True, uses `adjoint_symplectic_odeint` for gradient computation.
        Otherwise, uses `symplectic_odeint`. Defaults to True.

    **odeint_kwargs : dict
        Extra keyword arguments passed to the underlying symplectic integrator.
            - step_size: float, time step size (default: 1e-3).
            - num_steps: int, if given, overrides step_size.
            - method: str, integration method such as leapfrog.

    Methods
    -------
    forward(p, q, args=None)
        Evolves the state (p, q) forward from t0 to t1.

    reverse(p, q, args=None)
        Evolves the state (p, q) backward from t1 to t0.
    """

    def __init__(self, force_fn, t_span, adjoint_method=True, **odeint_kwargs):
        super().__init__()
        self.t_span = t_span
        self.force_fn = force_fn
        self.adjoint_method = adjoint_method

        # Prepare partially applied ODE integrator with provided kwargs
        if adjoint_method:
            self.odeint = ftpartial(adjoint_symplectic_odeint, **odeint_kwargs)
        else:
            self.odeint = ftpartial(symplectic_odeint, **odeint_kwargs)

    def forward(self, p, q, args=None):
        """
        Evolves the system state (p, q) forward in time.

        Args:
            p (torch.Tensor): Initial momentum.
            q (torch.Tensor): Initial position.
            args (Optional[Tuple[Tensor]]): Arguments passed to `force_fn`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Final momentum and position.
        """
        return self.odeint(self.force_fn, self.t_span, p, q, args=args)

    def reverse(self, p, q, args=None):
        """
        Evolves the system state (p, q) backward in time.

        This is equivalent to time-reversed integration from the final state.

        Args:
            p (torch.Tensor): Final momentum.
            q (torch.Tensor): Final position.
            args (Optional[Tuple[Tensor]]): Arguments passed to `force_fn`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Initial momentum and position.
        """
        reversed_t_span = self.t_span[::-1]
        return self.odeint(self.force_fn, reversed_t_span, p, q, args=args)


class SymplecticODEFlow_(SymplecticODEFlow):  # pylint: disable=invalid-name
    """
    Extension of SymplecticODEFlow for compatibility with the normflow package.

    This subclass provides an interface where the system state is passed as a
    single tuple `(p, q)`, rather than as separate arguments. It also conforms
    to the normalizing flow interface by returning the log-Jacobian of the
    transformation alongside the output, which is identically zero for
    symplectic flows.
    """

    def forward(self, var, args=None):
        """
        Evolves var = (p, q) forward in time and returns the final state and
        zero log-Jacobian.
        """
        return super().forward(var[0], var[1], args=args), 0

    def reverse(self, var, args=None):
        """
        Evolves var = (p, q) backward in time and returns the state at initial
        time and zero log-Jacobian.
        """
        return super().reverse(var[0], var[1], args=args), 0
