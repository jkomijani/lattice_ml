# Copyright (c) 2025 Javad Komijani

"""
Batched Molecular Dynamics (MD) integrator using a symplectic solver.

This module implements a class `HamiltonianMD` to perform batched MD
trajectories for lattice field theories or any system with a force function.
Each trajectory is integrated with a user-supplied symplectic ODE solver,
returning the final positions and a kinetic-energy related intermediate
quantity that can be used in Metropolis accept/reject steps (e.g., in HMC).

The `HamiltonianMD` class expects a force function implementing:
  - force_fn(q): returns the force ``-∂S/∂q`` given positions ``q``.

The term "MD" (Molecular Dynamics) is kept for historical reasons: in HMC
literature, integrating Hamilton's equations with Gaussian momenta is
traditionally called an MD trajectory. Conceptually, this is simply
Hamiltonian dynamics with random initial momenta.

Example usage:
    force_fn = lambda q: -grad_action(q)
    hmd = HamiltonianMD(force_fn, eps=0.25, num_steps=4)
    q_batch = torch.zeros((batch_size, *lattice_shape))
    q_final, delta_log_prob_momenta = hmd.step(q_batch)
"""

from functools import partial as ftpartial

import torch
from lattice_ml.integrate import symplectic_odeint


__all__ = ["HamiltonianMD"]


class HamiltonianMD:
    """
    Batched Molecular Dynamics (MD) integrator for lattice field theory.

    This class performs one MD trajectory using a symplectic integrator.
    It works in batch mode: the input is expected to have a leading batch
    dimension.

    The term "MD" (Molecular Dynamics) is kept for historical reasons: in
    the HMC literature, integrating Hamilton's equations with Gaussian
    momenta is traditionally called an MD trajectory. Conceptually, this
    is simply Hamiltonian dynamics with random initial momenta.

    Attributes
    ----------
    eps : float
        Step size of the symplectic integrator.
    num_steps : int
        Number of integration steps per trajectory.
    symplectic_odeint : Callable
        Partially applied symplectic ODE solver for the given force function.

    Methods
    -------
    step(q0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        Perform one batched MD trajectory starting from configurations ``q0``.
        Returns the final positions after integration and the negative change
        in kinetic energy, which can be used in Metropolis accept/reject.
    """

    def __init__(self, force_fn, eps, num_steps, **solver_kwargs):
        """
        Parameters
        ----------
        force_fn : Callable
            Function computing ``-∂S/∂q`` given ``q``.
        eps : float
            Step size of the symplectic integrator.
        num_steps : int
            Number of integration steps per trajectory.
        solver_kwargs : dict
            Additional arguments for the symplectic integrator.
        """
        self.eps = eps
        self.num_steps = num_steps

        # Partially applied symplectic ODE solver
        self.symplectic_odeint = ftpartial(
            symplectic_odeint,
            force_fn=force_fn,
            t_span=(0.0, eps * num_steps),
            num_steps=num_steps,
            **solver_kwargs
        )

    def step(self, q0: torch.Tensor):
        """
        Perform one batched MD trajectory starting from ``q0``.

        This method samples fresh Gaussian momenta, integrates the Hamiltonian
        dynamics with a symplectic solver, and computes a momentum-related
        log-probability difference. Because the solver is symplectic,
        the Jacobian determinant of the flow is exactly one. The returned
        intermediate quantity is not a Jacobian term but the negative change
        in kinetic energy, which is required for the Metropolis step in HMC.

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
        p0 = torch.randn_like(q0)

        # Integrate Hamiltonian dynamics using symplectic solver
        p, q = self.symplectic_odeint(p0=p0, q0=q0)

        # Compute negative change in kinetic energy per batch element
        dim = tuple(range(1, q0.ndim))
        logp0 = -0.5 * torch.sum(p0 * p0, dim=dim)
        logp = -0.5 * torch.sum(p * p, dim=dim)
        delta_log_prop_momentum = logp - logp0

        return q, delta_log_prop_momentum

    def __call__(self, q0: torch.Tensor):
        """Shorthand for :meth:`step`."""
        return self.step(q0)
