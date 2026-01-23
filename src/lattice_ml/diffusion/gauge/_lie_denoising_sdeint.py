# Created by Javad Komijani, May 2025

"""
Predictor-Corrector Sampler for Score-Based Generative Models
=============================================================

This module implements a Predictor-Corrector (PC) framework to generate samples
from complex data distributions modeled via score-based generative models and
stochastic differential equations (SDEs).

---

This approach is based on the framework introduced by:

    Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S.,
    & Poole, B. (2021). Score-Based Generative Modeling through
    Stochastic Differential Equations. arXiv:2011.13456.
    https://arxiv.org/abs/2011.13456

---

Methodology Summary
-------------------

In Section 2 of the above paper, the authors discuss two sampling strategies:

1. **Ancestral Sampling (Predictor)**
   - Integrates the reverse-time SDE using the Euler–Maruyama method.
   - Pros: Simple, fast. Cons: Discretization error may affect quality.

2. **Langevin MCMC Sampling (Corrector)**
   - Uses Langevin dynamics to refine samples at each step.
   - Pros: Improves sample fidelity. Cons: More computation and tuning.

In the paper, these are combined into a **Predictor-Corrector (PC)** framework:
The predictor proposes a sample using the reverse SDE, and the corrector
pulls it closer to the data manifold using the score function.

Unlike the mentioned paper, this module uses a **deterministic predictor**
(i.e., an ODE solver like RK4) rather than a stochastic ancestral sampler and
also estimates the log-jacobian of transformation via Hutchinson's estimator.

Key Components in This Module
-----------------------------

- **ode_doublestep_predictor_**: Implements a two-step ODE predictor with
  log-Jacobian estimation using Hutchinson's estimator.

- **langevin_corrector_**: Performs Langevin MCMC corrector steps with
  log-probability updates.
"""

from typing import Callable, Tuple

import torch

from lattice_ml.integrate._hutchinson_estimator import hutchinson_estimator

from ._lie_sdeint import special_unitary_euler_maruyama_algebra_step
from lattice_ml.integrate._lie_group_odeint import special_unitary_rk4_step


__all__ = [
    "integrate_denoising_sde",  # returns y
    "integrate_denoising_sde_"  # returns (y, log_prob)
]

Tensor = torch.Tensor


# =============================================================================
def integrate_denoising_sde(
    drift_fn: Callable,
    score_fn: Callable,
    t_span: Tuple[float, float],
    y0: Tensor,
    step_size: float = 0.001,
    num_steps: int | None = None,
    langevin_step_size: float = 0.001,
    num_langevin_iters: int = 1,
    ode_step: Callable = None,
    t_eval = None
) -> Tensor:
    """
    Solves an initial value problem for stochastic differential equations
    (SDEs) using an ODE-based predictor and Langevin corrector.

    Applies a predictor-corrector scheme where the predictor evolves the
    system via ODE integration, and the corrector refines it using Langevin
    dynamics based on a score function (derivative of log-probability).

    Args:
        drift_fn: Callable(t, y) -> Tensor
            Drift function used in the ODE predictor.
        score_fn: Callable(t, y) -> Tensor
            Score function for Langevin correction.
        t_span: Tuple (t0, t1)
            Time interval over which to integrate.
        y0: Tensor
            Initial sample tensor to evolve.
        step_size: float
            Time step size for the ODE integration (default: 0.001).
        num_steps: int, optional
            Number of integration steps (overrides step_size if set).
        langevin_step_size: float, optional
            Step size for Langevin correction (default: 0.001).
        num_langevin_iters: int, optional
            Number of Langevin iterations per correction step (default: 1).
        ode_step: Callable(func, t, y, dt) -> Tensor, optional
            Method for a single ODE step (e.g., RK4). Defaults to RK4.
            Temporary default setting:
               The default method for the predictor is 'RK4:SU(n)',
               The default method for the corrector is 'Euler-Maruyama:su(n)'.
            TODO: make it stanadard in future.

    Returns:
        Tensor: Final state tensor.

    Notes:
        - The integration uses `ode_step_predictor_` as the predictor,
          which performs one ODE steps per iteration for improved accuracy.
        - The corrector applies Langevin dynamics after each predictor step
          (except the final one) to refine the samples.
    """

    # Select integration method and define predictor and corrector
    if ode_step is None:
        ode_step = special_unitary_rk4_step  # default ODE step function

    predictor = ode_step
    corrector = special_unitary_euler_maruyama_algebra_step

    # Determine number of steps and time_grid
    t0, t1 = t_span

    if num_steps is None:
        num_steps = max(1, int(abs((t1 - t0) / step_size)))

    time_grid = torch.linspace(t0, t1, num_steps + 1, device=y0.device)

    step_size = time_grid[1] - time_grid[0]  # Actual step size

    # Make sure Langevin step size is not negative
    assert langevin_step_size >= 0, "Langevin step size cannot be negative"

    y = y0

    noise_scale = 2 ** 0.5  # to pass to the corrector

    # Apply predictor and corrector at each time step
    out_list = [y]
    for t, next_t in zip(time_grid[:-1], time_grid[1:]):
        y = predictor(drift_fn, t, y, step_size)
        for _ in range(num_langevin_iters):
            y = corrector(score_fn, next_t, y, langevin_step_size, noise_scale)
        out_list.append(y)

    return out_list


# =============================================================================
def integrate_denoising_sde_(
    drift_fn: Callable,
    score_fn: Callable,
    t_span: Tuple[float, float],
    y0: Tensor,
    logp0: Tensor | float,
    step_size: float = 0.001,
    num_steps: int | None = None,
    langevin_step_size: float = 0.001,
    num_langevin_iters: int = 2,
    ode_step: Callable = None,
    num_hutchinson_samples: int | None = 1,
    with_logp_components: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    Solves an initial value problem for stochastic differential equations
    (SDEs) using an ODE-based predictor and Langevin corrector.

    Applies a predictor-corrector scheme where the predictor evolves the
    system via ODE integration, and the corrector refines it using Langevin
    dynamics based on a score function (derivative of log-probability).

    Args:
        drift_fn: Callable(t, y) -> Tensor
            Drift function used in the ODE predictor.
        score_fn: Callable(t, y) -> Tensor
            Score function for Langevin correction.
        t_span: Tuple (t0, t1)
            Time interval over which to integrate.
        y0: Tensor
            Initial sample tensor to evolve.
        logp0: Tensor | float
            Initial log-probability per sample.
        step_size: float
            Time step size for the ODE integration (default: 0.001).
        num_steps: int, optional
            Number of integration steps (overrides step_size if set).
        langevin_step_size: float, optional
            Step size for Langevin correction (default: 0.001).
        num_langevin_iters: int, optional
            Number of Langevin iterations per correction step (default: 2).
        ode_step: Callable(func, t, y, dt) -> Tensor, optional
            Method for a single ODE step (e.g., RK4). Defaults to built-in.
        num_hutchinson_samples: int | None, optional
            Number of Hutchinson samples for log-Jacobian estimation
            (default: 1). If None, the log-Jacobian is calculated exactly that
            is expensive for high dimensional problems.
        with_logp_components: bool, optional
            If True, returns internal diagnostic values (default: False).

    Returns:
        If with_logp_components is False:
            Tuple[Tensor, Tensor]: final state and log-probability.
        If with_logp_components is True:
            Tuple[Tensor, Tensor, Tensor, Tensor]:
            final state, log-prob, predictor log-Jacobian, corrector log-prob.

    Notes:
        - The integration uses `ode_doublestep_predictor_` as the predictor,
          which performs two ODE steps per iteration for improved accuracy.
        - This double-step approach enhances log-Jacobian estimation by
          applying Hutchinson's trace estimator with Simpson's rule.
        - Due to this method, the predictor’s `step_size` is effectively
          doubled, and `num_steps` is halved to preserve the total integration
          interval.
        - The corrector applies Langevin dynamics after each predictor step
          (except the final one) to refine the samples.
    """
    t0, t1 = t_span

    # --- Adjust step size for double-step predictor ---
    if num_steps is None:
        num_doublesteps = max(1, int(abs((t1 - t0) / (2 * step_size))))
    else:
        num_doublesteps = num_steps // 2

    # Generate a uniform time grid
    time_doublestep_grid = torch.linspace(
        t0, t1, num_doublesteps + 1, device=y0.device
    )

    # Adjust actual step size based on time grid
    doublestep_size = time_doublestep_grid[1] - time_doublestep_grid[0]

    # Make sure Langevin step size is not negative
    assert langevin_step_size >= 0, "Langevin step size cannot be negative"

    # --- Define predictor function (ODE update using a double-step method) ---
    def predictor_(t, y, logj):
        return ode_doublestep_predictor_(
            drift_fn=drift_fn,
            t=t,
            y=y,
            logj=logj,
            doublestep_size=doublestep_size,
            num_hutchinson_samples=num_hutchinson_samples,
            ode_step=ode_step
        )

    # --- Define corrector function (Langevin update using score function) ---
    def corrector_(t, y, logp):
        return langevin_corrector_(
            score_fn=score_fn,
            t=t,
            y=y,
            logp=logp,
            langevin_step_size=langevin_step_size,
            num_iters=num_langevin_iters
        )

    # --- Run predictor-corrector integration loop ---
    return run_predictor_corrector_loop(
        predictor_=predictor_,
        corrector_=corrector_,
        time_grid=time_doublestep_grid,
        y0=y0,
        logp0=logp0,
        with_logp_components=with_logp_components,
    )


# =============================================================================
def run_predictor_corrector_loop(
    corrector_: Callable,
    predictor_: Callable,
    time_grid: Tensor,
    y0: Tensor,
    logp0: Tensor | float,
    with_logp_components: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    Runs a predictor-corrector loop over a time grid to evolve a stochastic
    system.

    This function alternates between applying a predictor (e.g., ODE
    integration) and a corrector (e.g., Langevin dynamics) to advance the
    system and refine its state over time.

    The predictor evolves the system along the time grid and manages its own
    step size internally. The corrector, applied at the same time point,
    does not advance time but refines the current state and log-probability.

    Args:
        predictor_: Callable(t, y, logj) -> (y, logj)
            Deterministic update that advances time and estimates the
            log-Jacobian of the transformation.
        corrector_: Callable(t, y, logp) -> (y, logp)
            Stochastic refinement step (e.g., Langevin) that improves sample
            quality without changing time.
        time_grid: Tensor
            1D tensor of equally spaced time points for integration.
        y0: Tensor
            Initial state of the system.
        logp0: Tensor | float
            Initial log-probability or log-density of each sample.
        with_logp_components: bool, optional
            If True, returns internal diagnostic values (default: False).

    Returns:
        If with_logp_components is False:
            Tuple[Tensor, Tensor]: final state and log-probability.
        If with_logp_components is True:
            Tuple[Tensor, Tensor, Tensor, Tensor]:
            final state, log-prob, predictor log-Jacobian, corrector log-prob.

    Notes:
        - The predictor is responsible for stepping along the time grid.
        - The corrector is applied at each step except the final one.
        - The final step uses only the predictor to reach the last time point.
        - Final log-probability includes contributions from both stages.
    """

    y = y0
    dlogp_corrector = 0  # Accumulates change in log-probability from corrector
    logj_predictor = 0  # Accumulates log-Jacobian from predictor

    # Apply predictor and corrector at each time step except the final one
    for t in time_grid[:-2]:
        y, logj_predictor = predictor_(t, y, logj_predictor)
        y, dlogp_corrector = corrector_(t, y, dlogp_corrector)

    # Final step: apply only the predictor from t_{N-2} to t_{N-1}
    y, logj_predictor = predictor_(time_grid[-2], y, logj_predictor)

    # Combine contributions to compute final log-probability
    logp = logp0 + dlogp_corrector - logj_predictor

    if with_logp_components:
        return y, logp, logj_predictor, dlogp_corrector

    return y, logp


# =============================================================================
def langevin_corrector_(
    score_fn: Callable,
    t: Tensor,
    y: Tensor,
    logp: Tensor | float,
    langevin_step_size: float,
    num_iters: int = 1
) -> Tuple[Tensor, Tensor]:
    """
    Performs Langevin correction on noisy samples using a score function.

    This refines the sample y by applying Langevin dynamics guided by the
    score function and updates the log-probability estimate.

    Args:
        score_fn: Callable(t, y) -> score (gradient of log-density).
        t: Current time step.
        y: Current state tensor to be corrected.
        logp: Log-probability tensor of each sample in the batch.
        langevin_step_size: Size of each Langevin step.
        num_iters: Number of Langevin iterations to apply.

    Returns:
        y: Corrected sample tensor (Tensor).
        logp: Updated log-probability tensor (Tensor).
    """
    bsize = y.shape[0]  # Batch size

    score = score_fn(t, y)  # Initial score (gradient of log-probability)

    for _ in range(num_iters):
        # Sample Gaussian noise of the same shape as y
        noise = torch.randn_like(y)

        # Compute Langevin update: score (drift) + noise (diffusion)
        dy = langevin_step_size * score + (2 * langevin_step_size)**0.5 * noise
        y = y + dy  # Update the sample

        # Compute updated score at the new sample location
        score_updated = score_fn(t, y)

        # Estimate change in log-probability using the symmetrized product
        dlogp = ((score + score_updated) * dy).view(bsize, -1).sum(dim=1) / 2
        logp = logp + dlogp  # Update log-probability

        score = score_updated  # Reuse updated score in next iteration

    return y, logp


# =============================================================================
def ode_doublestep_predictor_(
    drift_fn: Callable,
    t: float,
    y: Tensor,
    logj: Tensor | float,
    doublestep_size: float = 0.01,
    num_hutchinson_samples: int = 1,
    ode_step: Callable = None,
) -> Tuple[Tensor, Tensor]:
    """
    Performs a double-step ODE prediction and estimates the log-Jacobian
    of transformation using Hutchinson's estimator and Simpson's rule.

    Args:
        drift_fn: Callable(t, y) -> dy/dt, the ODE drift function.
        t: Currnet time (float).
        y: Currnet state (Tensor).
        logj: log-jacobian of previous steps (Tensor).
        doublestep_size: Total time step for two ODE steps (float).
        num_hutchinson_samples: Number of Hutchinson samples (int).
        ode_step: Function for a single ODE step, defaults to RK4 method.

    Returns:
        y: Final predicted state after two ODE steps (Tensor).
        logj: Estimated log-Jacobian of the transformation (Tensor).
    """

    if ode_step is None:
        ode_step = rk4_step  # default ODE step function

    def wrapped_fn(t):
        # Fix time t, return function of y only
        return lambda y: drift_fn(t, y)

    singlestep_size = doublestep_size / 2
    t0 = t
    t1 = t + singlestep_size
    t2 = t + doublestep_size

    # Estimate Jacobian trace at t0
    logj_t0 = hutchinson_estimator(wrapped_fn(t0), y, num_hutchinson_samples)

    # Step to midpoint t1
    y = ode_step(drift_fn, t0, y, singlestep_size)

    # Estimate Jacobian trace at t1
    logj_t1 = hutchinson_estimator(wrapped_fn(t1), y, num_hutchinson_samples)

    # Step to final time t2
    y = ode_step(drift_fn, t1, y, singlestep_size)

    # Estimate Jacobian trace at t2
    logj_t2 = hutchinson_estimator(wrapped_fn(t2), y, num_hutchinson_samples)

    # Simpson's rule for integration over [t0, t2]
    dlogj = (logj_t0 + 4 * logj_t1 + logj_t2) * (singlestep_size / 3)

    return y, logj + dlogj


# =============================================================================
def rk4_step(func: Callable, t: float, y: Tensor, dt: float) -> Tensor:
    """Perform a single Runge-Kutta-4 step."""
    eps = dt / 2
    k_1 = func(t, y)
    k_2 = func(t + eps, y + eps * k_1)
    k_3 = func(t + eps, y + eps * k_2)
    k_4 = func(t + dt, y + dt * k_3)
    return y + (k_1 + 2 * k_2 + 2 * k_3 + k_4) * (dt / 6)
