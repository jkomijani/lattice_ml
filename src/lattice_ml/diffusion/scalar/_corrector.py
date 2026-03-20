# Copyright (c) 2026 Javad Komijani

"""
Stochastic correctors for scalar field configurations.

This module provides two types of correctors that refine gauge field
configurations using stochastic dynamics:

- HMDBasedCorrector : Uses short Hamiltonian MD trajectories with
  fresh Gaussian momenta to refine configurations.

- LangevinBasedCorrector : Applies a discretized Langevin process on
  the Lie group. The number of iterations and step size are adjustable.

Examples
--------
# HMD-based corrector
corrector = HMDBasedCorrector(
    diffusion_model.score_fn,
    t_span_fn=lambda t: (0, 1),
    num_steps=10
)

# Langevin-based corrector
corrector = LangevinBasedCorrector(
    diffusion_model.score_fn,
    langevin_step_size_fn=lambda t: 0.01,
    num_langevin_iter=10
)
"""

from functools import partial as ftpartial

import torch
from lattice_ml.integrate import symplectic_odeint


__all__ = ["HMDBasedCorrector", "LangevinBasedCorrector"]


class HMDBasedCorrector:
    """
    Hamiltonian-MD-based corrector for scalar fields.

    This corrector applies a short Hamiltonian molecular dynamics (HMD)
    trajectory to refine input configurations. At each call, new Gaussian
    momenta are sampled, making the correction stochastic.

    Conceptually:
    - The force is computed at a fixed (frozen) time, which may correspond
      to a time in an external process such as a time-dependent score function.
    - The MD trajectory runs over a short, fictitious internal time used
      solely to evolve the system for the correction. This internal time
      does not affect the frozen time at which the force is evaluated.

    This construction allows the corrector to be used, for example, in reverse
    diffusion, where the score function (the gradient of the log-probability
    density) at each frozen time plays the role of the HMD force.

    Parameters
    ----------
    force_fn : callable
        Function computing the force acting on the configuration variables.
        It must have signature `force_fn(t, q)` and return a tensor. Here,
        `t` is the frozen time, distinct from the fictitious MD time.

    t_span_fn : callable
        Function mapping the frozen time to a short internal MD time interval.
        Signature: `t_span_fn(t)` → `(t0, t1)`.

    num_steps : int
        Number of symplectic integration steps used for the MD trajectory.

    **solver_kwargs : dict
        Additional keyword arguments passed to the symplectic integrator
        (e.g. integration method).
    """

    def __init__(self, force_fn, t_span_fn, num_steps, **solver_kwargs):
        self.force_fn = force_fn
        self.t_span_fn = t_span_fn
        self.num_steps = num_steps
        self.solver_kwargs = solver_kwargs

    def __call__(self, frozen_t: torch.Tensor, q0: torch.Tensor):

        _symplectic_odeint = ftpartial(
            symplectic_odeint,
            force_fn=lambda t, x: self.force_fn(frozen_t, x),
            t_span=self.t_span_fn(frozen_t),
            num_steps=self.num_steps,
            **self.solver_kwargs
        )
        # Sample Gaussian-distributed initial momenta
        p0 = torch.randn_like(q0)

        # Integrate Hamiltonian dynamics using symplectic solver
        p, q = _symplectic_odeint(p0=p0, q0=q0)

        return q


class LangevinBasedCorrector:
    """
    Langevin-based corrector for scalar fields.

    This corrector applies a stochastic trajectory to refine input
    configurations using a discretized Langevin process.

    Conceptually:
    - The drift is given by `force_fn(frozen_t, q)`, analogous to the gradient
      of the log-probability (or the HMD force) evaluated at the frozen time.
    - Gaussian noise is added at each iteration, scaled by the square root of
      2 times the step size, producing a stochastic evolution.

    This construction allows the corrector to be used in, for example, reverse
    diffusion, where the score function at each frozen time plays the role of
    the drift.

    Unlike the HMD-based corrector, Gaussian noise is sampled at each
    iteration, making the Langevin corrector exactly equivalent to performing
    one-step HMD corrections with refreshed momenta per iteration.

    Parameters
    ----------
    force_fn : callable
        Function computing the force acting on the configuration variables.
        It must have signature `force_fn(t, q)` and return a tensor. Here,
        `t` is the frozen time, distinct from the fictitious Langevin time.

    langevin_step_size_fn : callable
        Function mapping the frozen time to a Langevin step size. Signature:
        `langevin_step_size_fn(t)` → float. Determines the magnitude of each
        stochastic step.

    num_langevin_iter : int
        Number of Langevin iterations applied at each call. Controls the length
        of the stochastic corrector trajectory.
    """

    def __init__(self, force_fn, langevin_step_size_fn, num_langevin_iter):
        self.force_fn = force_fn
        self.langevin_step_size_fn = langevin_step_size_fn
        self.num_langevin_iter = num_langevin_iter

    def __call__(self, frozen_t: torch.Tensor, q0: torch.Tensor):
        dt = self.langevin_step_size_fn(frozen_t)
        noise_scale = abs(2 * dt)**0.5
        q = q0
        for _ in range(self.num_langevin_iter):
            drift = dt * self.force_fn(frozen_t, q)
            diffusion = noise_scale * torch.randn_like(q)
            q = q + drift + diffusion

        return q
