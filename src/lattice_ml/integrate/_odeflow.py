# Copyright (c) 2024 Javad Komijani

"""
This module defines classes for integrating systems of ordinary differential
equations (ODEs) using PyTorch. It includes functionality for evolving state
variables and computing the logarithm of the Jacobian determinant of
transformations.
"""

import functools
import torch

from ._odeint import odeint
from ._hutchinson_estimator import hutchinson_estimator


class ODEflow(torch.nn.Module):
    """
    A PyTorch module for evolving state variables governed by ODEs.

    This class integrates an ODE system using a specified derivative function.
    Given a function `f` that defines the system's dynamics, the state `y`
    evolves according to:

        dy/dt = f(t, y; p)

    where:
        - `t` is the time variable,
        - `y` represents the system's state at time `t`,
        - `p` denotes optional parameters influencing the dynamics.

    Fixed parameters can be embedded in `f` or explicitly provided via `args`.

    Args:
        - func (Callable): The function `f` that computes the time derivative
          of the state variable `y` at time `t`.
        - t_span (Tuple[float, float]): A tuple specifying initial and final
          times.
        - **odeint_kwargs: Additional keyword arguments for the ODE solver.
    """

    def __init__(self, func, t_span, **odeint_kwargs):

        super().__init__()
        self.func = func
        self.t_span = t_span
        self.odeint = functools.partial(odeint, **odeint_kwargs)

    def forward(self, var, args=None):
        """
        Evolves the system from initial to final time in `t_span`.

        Args:
            var (torch.Tensor): The initial state of the system.
            args (Optional[Tuple]): Additional arguments for `func`.

        Returns:
            torch.Tensor: The evolved system state at the final time.
        """
        return self.odeint(self.func, self.t_span, var, args=args)

    def reverse(self, var, args=None):
        """
        Evolves the system in reverse, from final to initial time.

        Args:
            var (torch.Tensor): The state of the system at the final time.
            args (Optional[Tuple]): Additional arguments for `func`.

        Returns:
            torch.Tensor: The evolved system state at the initial time.
        """
        return self.odeint(self.func, self.t_span[::-1], var, args=args)


class ODEflow_(ODEflow):  # pylint: disable=invalid-name
    """
    An extension of `ODEflow` that also returns the log-Jacobian of the flow.

    This class evolves a system of ODEs while also tracking the log-determinant
    of the Jacobian transformation. If `func` has a method `calc_logj_rate`, it
    is used to compute the log-Jacobian rate directly. Otherwise, the Jacobian
    trace is estimated using the Hutchinson estimator with a specified number
    of samples.

    If `num_samples` is `None`, the Hutchinson estimator is not actually used.
    Instead, the Jacobian trace is computed exactly via automatic
    differentiation, which can be computationally expensive in high dimensions.

    Args:
        - func (Callable): A function `f(t, y, *args)` that computes the time
          derivative of the state variable `y` at time `t`.
        - t_span (Tuple[float, float]): A tuple specifying initial and final
          times.
        - num_samples (Optional[int | None]): The number of random samples used
          in the Hutchinson estimator. If `None`, the Jacobian trace is
          computed exactly. Defaults to 1.
        - **odeint_kwargs: Additional keyword arguments for the ODE solver.
    """

    def __init__(self, func, t_span, num_samples=1, **odeint_kwargs):

        if hasattr(func, 'calc_logj_rate'):
            loss_rate = func.calc_logj_rate
        else:
            def loss_rate(t, var, *args):
                return hutchinson_estimator(lambda x: func(t, x, *args),
                                            var, num_samples)

        super().__init__(func, t_span, loss_rate=loss_rate, **odeint_kwargs)
