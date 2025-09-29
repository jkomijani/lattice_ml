# Copyright (c) 2025 Javad Komijani

"""
Batched Molecular Dynamics (MD) integrator using a symplectic solver.

This module implements the class `HamiltonianMD`, which is also alised to
`HMD`, to perform batched MD trajectories for lattice field theories or any
system with a force function.
Each trajectory is integrated with a user-supplied symplectic ODE solver,
returning the final positions and a kinetic-energy related intermediate
quantity that can be used in Metropolis accept/reject steps (e.g., in HMC).

Example usage:
    force_fn = lambda t, q: -grad_action(q)
    hmd = SUnHamiltonianMD(force_fn, t_span=(0, 1), num_steps=4)
    q_batch = torch.matrix_exp(torch.zeros((batch_size, *lattice_shape)))
    q_final, delta_log_prob_momenta = hmd.step(q_batch)
"""

from functools import partial as ftpartial

import torch
from lattice_ml.integrate import lie_symplectic_odeint
from ._lie_hmc import randn_traceless_antihermitian_like 


__all__ = ["SUnHamiltonianMD", "SUnHMD"]


class SUnHamiltonianMD:
    """
    Batched Molecular Dynamics (MD) integrator for lattice field theory.

    This class performs one MD trajectory using a symplectic integrator.
    It works in batch mode: the input is expected to have a leading batch
    dimension.

    The term "MD" (Molecular Dynamics) is kept for historical reasons: in
    the HMC literature, integrating Hamilton's equations with Gaussian momenta
    is traditionally called an MD trajectory. Conceptually, this is simply
    Hamiltonian dynamics with random initial momenta.

    The dynamics follow a canonical symplectic form:

        dq/dt = ∂H/∂p = p
        dp/dt = -∂H/∂q = force_fn(t, q, *args)

    where `force_fn` models the generalized force acting on the system.

    Initial momenta are drawn from a standard normal distribution, N(0, I).
    This corresponds to a quadratic kinetic energy

          K(p) = 0.5 * p^2.

    Attributes
    ----------
    symplectic_odeint : Callable
        Partially applied symplectic ODE solver for the given force function.

    Methods
    -------
    step(q0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        Perform one batched MD trajectory starting from configurations ``q0``.
        Returns the final positions after integration and the negative change
        in kinetic energy, which can be used in Metropolis accept/reject.
    """

    def __init__(self, force_fn, t_span, num_steps, **solver_kwargs):
        """
        Parameters
        ----------
        force_fn : Callable
            Defines the dynamics of the momentum, typically representing
            the force as -∂H/∂q. It takes the form `force_fn(t, q, *args)`,
            where t is time, q is position, and args are optional arguments.
        t_span : tuple of float
            Time interval (t0, t1) for integration.
        num_steps : int
            Number of steps to span the time interval.
        **solver_kwargs : dict
            Extra keyword arguments passed to the symplectic integrator.
            - method: str, integration method such as "leapfrog".
            - velocity_fn: callable, function modeling the position dynamics.
              If not provided, defaults to the canonical choice dq/dt = p.
        """
        # Partially applied symplectic ODE solver
        self.symplectic_odeint = ftpartial(
            lie_symplectic_odeint,
            force_fn=force_fn,
            t_span=t_span,
            num_steps=num_steps,
            **solver_kwargs
        )

    def step(self, q0: torch.Tensor):
        """
        Perform one batched MD trajectory starting from ``q0``.

        This method samples fresh Gaussian momenta, integrates the Hamiltonian
        dynamics with a symplectic solver, and computes the momentum-related
        log-probability difference.

        Because the solver is symplectic, the Jacobian determinant of the flow
        is exactly one. The returned intermediate quantity is not a Jacobian
        correction but the negative change in kinetic energy, which is required
        for the Metropolis step in HMC.

        Parameters
        ----------
        q0 : torch.Tensor
            Initial positions of shape ``(batch_size, *lattice_shape)``.

        Returns
        -------
        q : torch.Tensor
            Final positions after MD integration, same shape as ``q0``.
        delta_log_prop_momentum : torch.Tensor
            Difference in log-probability of momenta between the final and
            initial states, i.e. ``log_prob(p) - log_prob(p0)``, equal to
            the negative change in kinetic energy. This is an intermediate
            quantity that can be used in the Metropolis accept/reject step
            of HMC.
        """
        # Sample Gaussian-distributed initial momenta
        p0 = randn_traceless_antihermitian_like(q0)

        # Integrate Hamiltonian dynamics using symplectic solver
        p, q = self.symplectic_odeint(p0=p0, q0=q0)

        # Compute negative change in kinetic energy per batch element
        dim = tuple(range(1, q0.ndim))
        logp0 = -0.5 * torch.sum(p0 * p0.conj(), dim=dim).real
        logp = -0.5 * torch.sum(p * p.conj(), dim=dim).real
        delta_log_prop_momentum = logp - logp0

        return q, delta_log_prop_momentum

    def __call__(self, q0: torch.Tensor):
        """Shorthand for :meth:`step`."""
        return self.step(q0)


# Short alias for convenience
SUnHMD = SUnHamiltonianMD
