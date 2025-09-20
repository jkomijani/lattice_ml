# Created by Javad Komijani, 2024

"""Implements the denosing process with different methods.

It is meant for tests for now.
"""

# pylint: disable=too-many-arguments, too-many-positional-arguments

import numpy as np

from lattice_ml.integrate import ODEFlow
from lattice_ml.integrate import ODEFlow_
from lattice_ml.integrate import AdjODEFlow_

from ._sdeint import SDEIntegrator
from ._denoising_sdeflow import DenoisingSDEFlow


__all__ = [
    "build_denoising_sde_integrator",
    "build_denoising_odeflow",
    "build_denoising_odeflow_"
]


# =============================================================================
def build_denoising_sde_integrator(
    diffusion_process,
    t_span: tuple = (1, 0),
    method: str = 'predictor-corrector',
    step_size: float = 0.001,
    sigma_tilde: float = np.pi ** 0.5,
    langevin_step_size: float = 0.001,
    **solver_kwargs
):
    """
    Builds and returns an instance of `SDEIntegrator` that performs
    denoising using a stochastic ordinary differential equation (SDE).

    Parameters
    ----------
    diffusion_process: object
        An instance of DiffusionProcess.

    t_span : tuple of float
        The time span for the denoising flow, usually from a higher time
        (e.g., 1) to a lower time (e.g., 0).

    method : str, optional
        The method for solving the SDE. Default is 'predictor-corrector'.

    step_size : float, optional
        The step size for discretizing the SDE. Default is 0.001.

    langevin_step_size: float, optional
        Step size for Langevin correction in 'predictor-corrector' method.
        Default is 0.001. Note that `langevin_step_size` is equivalent to
        `step_size * sigma_tilde ** 2`.

    sigma_tilde : float, optional
        Coefficient of the noise term in the SDE (up to squared root of 2).
        Defaults to `sqrt(pi)`. Relavant only for Euler-Maruyam method.

    **solver_kwargs:
        Additional keyword arguments relavant only for predictor-corrector
        method.

    Returns
    -------
    SDEIntegrator | DenoisingSDEFlow
        An instance of either of these classes depending on `method`.
    """
    if method == 'predictor-corrector':
        # Put all key-word args in a dictionary to pass to the solver
        solver_kwargs.update({
            'step_size': step_size,
            'langevin_step_size': langevin_step_size
        })

        # Build the drift function used in the predictor, an ODE
        # (sigma_tilde=0 for deterministic behavior of the predictor)
        drift_fn = diffusion_process.build_denoising_drift(sigma_tilde=0)

        # Instantiate & return DenoisingSDEFlow
        return DenoisingSDEFlow(
                drift_fn, diffusion_process.score_fn, t_span, **solver_kwargs
                )

    if method == 'Euler-Maruyama':
        # Put all keyword args in a dictionary to pass to the solver

        solver_kwargs = {
            'method': method,
            'step_size': step_size,
            'noise_scale': np.sqrt(2) * sigma_tilde
        }

        # Build the drift function used in the SDE
        drift_fn = diffusion_process.build_denoising_drift(sigma_tilde)

        # Instantiate & return SDEIntegrator
        return SDEIntegrator(drift_fn, t_span, **solver_kwargs)

    raise ValueError(f"Unsupported integration method: {method}")


def build_denoising_odeflow(
    diffusion_process,
    t_span: tuple = (1, 0),
    method: str = 'RK4',
    step_size: float = 0.001
):
    """
    Builds and returns an instance of `ODEFlow` that performs denoising
    using a deterministic ordinary differential equation (ODE).

    Unlike the denoising process, which solves a stochastic differential
    equation (SDE) for denoising, this method follows a deterministic
    approach.

    Parameters
    ----------
    diffusion_process: object
        An instance of DiffusionProcess.

    t_span : tuple of float
        The time span for the denoising flow, usually from a higher time
        (e.g., 1) to a lower time (e.g., 0).

    method : str, optional
        The method to be used for solving the ODE. Default is 'RK4'.

    step_size : float, optional
        The step size for discretizing the ODE. Default is 0.001.

    Returns
    -------
    ODEFlow
        An instance of the `ODEFlow` class that can be used to perform
        denoising using the deterministic ODE method.
    """
    # Build the drift function used in the ODE
    # (sigma_tilde=0 for deterministic behavior)
    drift_fn = diffusion_process.build_denoising_drift(sigma_tilde=0)

    # Instantiate & return the ODE flow with drift_fn & solver parameters
    return ODEFlow(drift_fn, t_span, method=method, step_size=step_size)


def build_denoising_odeflow_(
    diffusion_process,
    t_span: tuple = (1, 0),
    method: str = 'RK4',
    step_size: float = 0.001,
    num_hutchinson_samples: int = 1,
    adjoint_backprop: bool = True
):
    """
    Builds and returns an instance of `ODEFlow_` or `AdjODEFlow_` that
    performs denoising using a deterministic ordinary differential equation
    (ODE) and also returns the log-Jacobian of the flow.

    Unlike the denoising process, which solves a stochastic differential
    equation (SDE) for denoising, this method follows a deterministic
    approach.

    This method is used to compute the log Jacobian of the flow along with
    the denoised state, which is useful for tasks like likelihood
    estimation and gradient-based optimization.

    Parameters
    ----------
    diffusion_process: object
        An instance of DiffusionProcess.

    t_span : tuple of float
        The time span for the denoising flow, usually from a higher time
        (e.g., 1) to a lower time (e.g., 0).

    method : str, optional
        The method to be used for solving the ODE. Default is 'RK4'.

    step_size : float, optional
        The step size for discretizing the ODE. Default is 0.001.

    num_hutchinson_samples : int, optional
        The number of samples for Hutchinson estimator of the log-Jacobian
        of the flow. Default is 1.

    adjoint_backprop : bool, optional
        If True, the adjoint method is used for backpropagation, otherwise
        the standard method is used.

    Returns
    -------
    ODEFlow_ or AdjODEFlow_
        An instance of either the `ODEFlow_` or `AdjODEFlow_` class,
        depending on the choice of backpropagation method.
    """
    # Put all key-word arguments in a dictionary to pass to the solver
    solver_kwargs = {
        'method': method,
        'step_size': step_size,
        'num_hutchinson_samples': num_hutchinson_samples  # for Jacobian
    }

    # Build the drift function used in the ODE
    # (sigma_tilde=0 for deterministic behavior)
    drift_fn = diffusion_process.build_denoising_drift(sigma_tilde=0)

    # Choose the appropriate ODE flow class: adjoint or standard
    flow_class_ = AdjODEFlow_ if adjoint_backprop else ODEFlow_

    # Instantiate & return the ODE flow with drift_fn & solver parameters
    return flow_class_(drift_fn, t_span, **solver_kwargs)
