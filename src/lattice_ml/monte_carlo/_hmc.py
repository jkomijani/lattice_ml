# Copyright (c) 2025 Javad Komijani

"""
Batched Hybrid Monte Carlo (HMC) sampler using a symplectic integrator.

This module implements the class `HMC` to perform batched HMC updates for
lattice field theories or any system with a differentiable action. Each
trajectory is integrated with a user-supplied symplectic solver and
accepted/rejected independently per batch element using the Metropolis
criterion.

Example usage:
    action = ScalarAction(...)
    force_fn = lambda t, q: action.force(q)
    hmc = HMC(force_fn, t_span=(0, 1), num_steps=4, action=action)
    q_batch = torch.zeros((batch_size, *lattice_shape))
    q_new, accepted = hmc.step(q_batch)
"""

from functools import partial as ftpartial

import torch
from lattice_ml.integrate import symplectic_odeint

__all__ = ["HMC"]


class HMC:
    """
    Batched Hybrid Monte Carlo (HMC) sampler using a symplectic integrator.

    This class performs batched HMC updates for scalar field theories using
    Hamiltonian dynamics integrated with a symplectic solver. The sampler works
    in batch mode: the input is expected to have a leading batch dimension,
    and each trajectory is evolved and accepted/rejected independently.

    The Hamiltonian is defined as:
        H(p, q) = T(p) + S(q)
    where
        T(p) = 1/2 * sum(p^2)   is the kinetic energy,
        S(q) = action(q)        is the potential energy (user-supplied).

    Note that the kinetic term can be changed from the canonical form by
    supplying a custom `velocity_fn` to the symplectic solver.

    Attributes
    ----------
    action : Callable
        User-supplied action, returning S(q) as shape (batch_size,)
    symplectic_odeint : Callable
        Partially applied symplectic ODE solver for the given action.

    Methods
    -------
    step(q0: torch.Tensor) -> Tuple[torch.Tensor, torch.BoolTensor]
        Perform one batched HMC update starting from configurations `q0`.
        Returns updated configurations and a mask of accepted proposals.
    """

    def __init__(self, force_fn, t_span, num_steps, action, **solver_kwargs):
        """
        Parameters
        ----------
        force_fn : Callable
            Defines the dynamics of the momentum, i.e., ``-∂H/∂q``.
            To remain consistent with the underlying symplectic integrator,
            it takes the form `force_fn(t, q, *args)`, where t is time, q is
            generalized position, and args are optional arguments.
        t_span : tuple of float
            Time interval (t0, t1) for integration.
        num_steps: int
            Number of steps to span the time interval.
        action : Callable
            Defines the action and returns S(q) as shape (batch_size,)
        **solver_kwargs : dict
            Extra keyword arguments passed to the symplectic integrator.
            - method: str, integration method such as leapfrog.
            - velocity_fn: callable, function modeling the position dynamics.
              If not provided, defaults to the canonical choice dq/dt = p.
        """
        self.action = action

        # Partially applied symplectic ODE solver
        self.symplectic_odeint = ftpartial(
            symplectic_odeint,
            force_fn=force_fn,
            t_span=t_span,
            num_steps=num_steps,
            **solver_kwargs
        )

        self.accept_rate_history = []

    def step(self, q0: torch.Tensor, return_delta_energy=False):
        """
        Perform one batched HMC proposal starting from q0.

        Parameters
        ----------
        q0 : torch.Tensor
            Shape (batch_size, *lattice_shape).

        Returns
        -------
        q_new : torch.Tensor
            Updated configurations (accepted or kept old).
        is_accepted : torch.BoolTensor
            Boolean mask of accepted trajectories.
        """
        bsize = q0.shape[0]

        # sample fresh momenta
        p0 = torch.randn_like(q0)

        # integrate equations of motion
        p, q = self.symplectic_odeint(p0=p0, q0=q0)

        # initial & final Hamiltonians per batch element
        dim = tuple(range(1, q0.ndim))
        h0 = self.action(q0) + 0.5 * torch.sum(p0 * p0, dim=dim)
        h = self.action(q) + 0.5 * torch.sum(p * p, dim=dim)

        # acceptance probability
        accept_prob = torch.exp(-(h - h0)).clamp(max=1.0)

        # accept/reject per batch
        u = torch.rand(bsize, device=q0.device)
        is_accepted = u < accept_prob

        # choose accepted or keep old
        mask = is_accepted.view(-1, *([1] * (q.ndim - 1)))
        q_new = torch.where(mask, q, q0)

        self.accept_rate_history.append(is_accepted.float().mean().item())

        if return_delta_energy:
            return q_new, is_accepted, (h - h0)

        return q_new, is_accepted
