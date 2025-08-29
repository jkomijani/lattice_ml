# Copyright (c) 2025 Javad Komijani

"""
Batched Hybrid Monte Carlo (HMC) sampler using a symplectic integrator.

This module implements a class `HMC` to perform batched HMC updates for
lattice field theories or any system with a differentiable action. Each
trajectory is integrated with a user-supplied symplectic solver and
accepted/rejected independently per batch element using the Metropolis
criterion.

The `HMC` class expects an action object implementing:
  - __call__(q): returns the action S(q) as a batch of scalars.
  - force(q) or algebra_force(q): returns the force -∂S/∂q or its algebraic
    correspondence for Lie groups.

Example usage:
    action = ScalarAction(...)
    hmc = HMC(action, eps=0.25, num_steps=4)
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

    Attributes
    ----------
    action : object
        User-supplied action implementing:
          - __call__(q): returns S(q) as shape (batch_size,)
          - force(t, q): returns the force -∂S/∂q with same shape as q
    eps : float
        Step size of the symplectic integrator.
    num_steps : int
        Number of integration steps per trajectory.
    symplectic_odeint : Callable
        Partially applied symplectic ODE solver for the given action.

    Methods
    -------
    step(q0: torch.Tensor) -> Tuple[torch.Tensor, torch.BoolTensor]
        Perform one batched HMC update starting from configurations `q0`.
        Returns updated configurations and a mask of accepted proposals.
    """

    def __init__(self, action, eps, num_steps, **solver_kwargs):

        self.action = action
        self.eps = eps
        self.num_steps = num_steps

        self.symplectic_odeint = ftpartial(
            symplectic_odeint,
            force_fn=lambda t, q: action.force(q),
            t_span=(0.0, eps * num_steps),
            num_steps=num_steps,
            **solver_kwargs
        )

        self.accept_rate_history = []

    def step(self, q0: torch.Tensor):
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

        return q_new, is_accepted
