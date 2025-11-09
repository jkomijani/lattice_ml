# Created by Javad Komijani, 2025

"""Implements the denosing process with different methods.

It is meant for tests for now.
"""

# pylint: disable=too-many-arguments, too-many-positional-arguments


from lattice_ml.integrate import LieODEFlow
from lattice_ml.integrate import LieODEFlow_


__all__ = [
    "build_denoising_odeflow",
    "build_denoising_odeflow_"
]


# =============================================================================
def build_denoising_odeflow(
    diffusion_process,
    t_span: tuple = (1, 0),
    method: str = 'RK4:SU(n)',
    step_size: float = 0.001,
    **solver_kwargs
):
    """
    Build a `LieODEFlow` for deterministic ODE denoising.

    Unlike SDE-based denoising, this follows a deterministic approach.

    Parameters
    ----------
    diffusion_process : object
        An instance of DiffusionProcess.
    t_span : tuple of float
        Time span for the ODE, usually from higher to lower (e.g., 1 to 0).
    method : str, optional
        ODE solver method. Default is 'RK4:SU(n)'.
    step_size : float, optional
        Step size for discretizing the ODE. Default is 0.001.
    **solver_kwargs : dict
        Additional keyword arguments passed to `LieODEFlow`.

    Returns
    -------
    LieODEFlow
        A `LieODEFlow` instance for deterministic ODE-based denoising.
    """
    # Put all key-word arguments in a dictionary to pass to the solver
    solver_kwargs.update({'method': method, 'step_size': step_size})

    # Build the drift function used in the ODE
    drift_fn = diffusion_process.build_denoising_drift(
        rev2fwd_noise_ratio=0,
        algebra_valued='SU(n)' not in method
    )

    # Instantiate & return the ODE flow with drift_fn & solver parameters
    return LieODEFlow(drift_fn, t_span, **solver_kwargs)


def build_denoising_odeflow_(
    diffusion_process,
    t_span: tuple = (1, 0),
    method: str = 'RK4:SU(n)',
    step_size: float = 0.001,
    num_hutchinson_samples: int = 1,
    **solver_kwargs
):
    """
    Build a `LieODEFlow_` for deterministic ODE denoising with log-Jacobian.

    Unlike SDE-based denoising, this follows a deterministic approach.

    Parameters
    ----------
    diffusion_process : object
        An instance of DiffusionProcess.
    t_span : tuple of float
        Time span for the ODE, usually from higher to lower (e.g., 1 to 0).
    method : str, optional
        ODE solver method. Default is 'RK4:SU(n)'.
    step_size : float, optional
        Step size for discretizing the ODE. Default is 0.001.
    num_hutchinson_samples : int, optional
        Number of Hutchinson estimator samples for log-Jacobian. Default is 1.
    **solver_kwargs : dict
        Additional keyword arguments passed to `LieODEFlow_`.

    Returns
    -------
    LieODEFlow_
        A `LieODEFlow_` instance for deterministic ODE-based denoising.
    """
    # Put all key-word arguments in a dictionary to pass to the solver
    solver_kwargs.update({
        'method': method,
        'step_size': step_size,
        'num_hutchinson_samples': num_hutchinson_samples
    })

    # Build the drift function used in the ODE
    drift_fn = diffusion_process.build_denoising_drift(
        rev2fwd_noise_ratio=0,
        algebra_valued='SU(n)' not in method
    )

    # Instantiate & return the ODE flow with drift_fn & solver parameters
    return LieODEFlow_(drift_fn, t_span, **solver_kwargs)
