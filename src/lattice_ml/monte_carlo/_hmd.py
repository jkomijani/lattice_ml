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
    hmd = HamiltonianMD(force_fn, t_span=(0, 1), num_steps=4)
    q_batch = torch.zeros((batch_size, *lattice_shape))
    q_final, delta_log_prob_momenta = hmd.step(q_batch)
"""

from functools import partial as ftpartial

import torch
from lattice_ml.integrate import symplectic_odeint


__all__ = ["HamiltonianMD", "HMD", "ResampledHamiltonianMD", "ResampledHMD"]


class HamiltonianMD:
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
    Note that the position dynamics can be changed from the canonical form by
    supplying a custom `velocity_fn` to the symplectic solver.

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
            symplectic_odeint,
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


# Short alias for convenience
HMD = HamiltonianMD


class ResampledHamiltonianMD:
    """
    Batched resampled symplectic integrator for Molecular Dynamics (MD)
    in lattice field theory.

    This class implements a chain of single-step :class:`HamiltonianMD`
    trajectories. At the start of each segment, fresh Gaussian momenta are
    resampled, after which deterministic Hamiltonian dynamics are integrated
    with a symplectic solver. The resulting process is therefore a hybrid:

        - stochasticity at the boundaries via momentum resampling,
        - deterministic, volume-preserving dynamics within each segment.

    Conceptually, this can be viewed as a discrete, symplectic analogue of
    Langevin-type dynamics: momentum resampling provides the noise, while the
    drift is governed by the force term ``force_fn``.

    Equations of motion
    -------------------
    Within each Hamiltonian segment, the dynamics follow the canonical form
    (unless a custom velocity function is supplied):

        dq/dt = ∂H/∂p = p
        dp/dt = -∂H/∂q = force_fn(t, q, *args)

    where `force_fn` provides the generalized force acting on the system.

    Time Reparameterization
    -----------------------
    Each global step of size ``dt = (t1 - t0) / num_steps`` is mapped
    onto a local integration parameter ``s ∈ [0, ds]`` with

        ds = sqrt(2 * |dt|) * |scale|.

    - At the start of the n-th segment (s=0):
          t = t0 + n * dt
    - At the end of the segment (s=ds):
          t = t0 + (n+1) * dt

    The wrapper function handles this mapping, ensuring that the local
    symplectic solver integrates in a normalized interval [0, ds] while still
    advancing the physical time ``t`` correctly.

    The ``scale`` parameter rescales the mapping between local and physical
    time. Concretely:

        - Increasing ``scale`` stretches the local interval length ``ds``,
          making each segment's integration domain longer.
        - To compensate, the force is rescaled by ``1 / scale²`` so that the
          physical dynamics in ``t`` remain consistent.
        - Effectively, it would act as scaling the random momenta.

    The force is also multiplied by ``sign(dt)``, which guarantees consistency
    under time reversal: if dt < 0, the integration correctly runs backward
    in time.

    Attributes
    ----------
    hmd_list : List[HamiltonianMD]
        Sequence of single-step HamiltonianMD solvers covering the full
        integration interval. Each solver begins with new momentum samples.

    Methods
    -------
    step(q0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        Execute the chain of resampled MD steps from initial positions ``q0``.
        Returns the final positions along with the accumulated negative change
        in kinetic energy (used for Metropolis acceptance).
    """

    def __init__(self, force_fn, t_span, num_steps, scale=1, **solver_kwargs):
        """
        Parameters
        ----------
        force_fn : Callable
            Defines the dynamics of the momentum, typically representing
            the force as -∂H/∂q. It takes the form `force_fn(t, q, *args)`,
            where t is time, q is position, and args are optional arguments.
        t_span : tuple of float
            Time interval (t0, t1) for integration.
        num_steps: int
            Number of steps to span the time interval.
        scale : float, optional (default=1)
            Reparameterization factor controlling the mapping between physical
            time ``t`` and the local solver interval ``s``.
        **solver_kwargs : dict
            Extra keyword arguments passed to the symplectic integrator.
            - method: str, integration method such as leapfrog.
            - velocity_fn: callable, function modeling the position dynamics.
              If not provided, defaults to the canonical choice dq/dt = p.
        """
        assert num_steps > 0, "number of steps must be a positive integer."

        # Global step size (can be positive or negative).
        dt = (t_span[1] - t_span[0]) / num_steps

        # Local reparametrization length for symplectic solver.
        # Each HamiltonianMD segment integrates over [0, ds].
        ds = (2 * abs(dt)) ** 0.5 * abs(scale)

        coeff = (1 if dt > 0 else -1) / scale**2

        # Wrapper: maps local parameter (s, q) back to physical time t
        # in the n-th segment, and adjusts sign for dt < 0.
        def force_fn_symplectic_wrapper(s, q, n):
            t = t_span[0] + (n + s / ds) * dt
            return coeff * force_fn(t, q)

        # Build per-segment force functions with n fixed.
        force_fn_list = [
            ftpartial(force_fn_symplectic_wrapper, n=n)
            for n in range(num_steps)
        ]

        # Build HamiltonianMD segments, each running a single [0, ds] step.
        # At the beginning of each, new Gaussian momenta are sampled.
        self.hmd_list = [
            HamiltonianMD(f_fn, t_span=(0, ds), num_steps=1, **solver_kwargs)
            for f_fn in force_fn_list
        ]

    def step(self, q0: torch.Tensor):
        """
        Execute the stochastic MD trajectory starting from initial positions.

        Each HamiltonianMD segment:
          1. Samples fresh Gaussian momenta.
          2. Integrates a single-step trajectory over [0, ds] using
             a symplectic solver.
          3. Returns updated positions and the momentum log-probability change.

        Because the solver is symplectic, the flow is volume-preserving
        (Jacobian = 1). The quantity accumulated here is *not* a Jacobian
        correction but the cumulative negative change in kinetic energy, which
        is required in the Metropolis step of Hamiltonian Monte Carlo (HMC).

        Parameters
        ----------
        q0 : torch.Tensor
            Initial positions, shape ``(batch_size, *lattice_shape)``.

        Returns
        -------
        q : torch.Tensor
            Final positions after the MD trajectory, same shape as ``q0``.

        delta_log_prop_momentum : torch.Tensor
            Cumulative difference in log-probability of momenta across all
            sub-trajectories:

                log_prob(p_final) - log_prob(p_initial)

            Since Gaussian momenta are used, this equals the negative change
            in kinetic energy across the entire trajectory, and enters directly
            into the Metropolis accept/reject step.
        """
        # Start from given positions
        q = q0

        # Accumulate momentum log-probability differences from each segment
        accumulated_delta_log_prob_momentum = 0

        # Loop over the sequence of HamiltonianMD segments
        for hmd in self.hmd_list:
            q, delta_log_prop_momentum = hmd.step(q)
            accumulated_delta_log_prob_momentum += delta_log_prop_momentum

        # Return final positions and total kinetic-energy-based correction
        return q, accumulated_delta_log_prob_momentum

    def __call__(self, q0: torch.Tensor):
        """Shorthand for :meth:`step`."""
        return self.step(q0)


# Short alias for convenience
ResampledHMD = ResampledHamiltonianMD
