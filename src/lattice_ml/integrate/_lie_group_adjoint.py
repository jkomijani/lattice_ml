# Copyright (c) 2024-2025 Javad Komijani

"""
Defines the AdjLieODEflow_ module for adjoint-based ODE integration with
log-Jacobian tracking.
"""

from abc import abstractmethod, ABC
from functools import partial as ftpartial

import torch

from ._lie_group_odeint import lie_group_odeint
from ._adjoint import TupleVar
from ._adjoint import tie_adjoints
from ._hutchinson_estimator import hutchinson_estimator


# =============================================================================
class AdjLieODEflow_(torch.nn.Module):  # pylint: disable=invalid-name
    """
    A module for solving ODEs with adjoint-based backprop and log-Jacobian
    tracking.

    AdjODEflow_ is similar to `ODEflow_`, which extends `ODEflow` by also
    returning the log-determinant of the Jacobian of the flow. While `ODEflow`
    only evolves the state variable, `ODEflow_` and `AdjODEflow_` additionally
    compute the log-Jacobian of the transformation.

    This class uses the adjoint method to compute gradients during backward
    passes, which is memory efficient for long sequences.

    If `func` defines a method `calc_logj_rate`, it is used directly to compute
    the log-Jacobian rate. Otherwise, the trace of the Jacobian is estimated
    using the Hutchinson estimator with a given number of random samples.

    If `num_samples` is `None`, the Jacobian trace is computed exactly via
    automatic differentiation, which may be slow in high-dimensional settings.

    If `func` is not an instance of `DynamicsAdjModule`, it will automatically
    be wrapped with `DynamicsAdjWrapper`.

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

    def __init__(
        self, func, t_span, num_samples=1,
        methods=('RK4:SU(n)', 'RK4:SU(n):aug'),
        **odeint_kwargs
    ):
        """Initializes the AdjLieODEflow_ module."""

        super().__init__()

        if isinstance(func, AdjLieModule):
            self.func = func
        else:
            self.func = AdjLieModuleWrapper(func, num_samples)

        self.t_span = t_span

        self.odeints = [
            ftpartial(lie_group_odeint, method=methods[0], **odeint_kwargs),
            ftpartial(lie_group_odeint, method=methods[1], **odeint_kwargs)
        ]

    def forward(self, var, args=None, log0=0):
        """
        Evolves the state forward in time and accumulates the log-Jacobian.

        Args:
            var (torch.Tensor): Initial state variable to evolve.
            args (Optional[Tuple[torch.Tensor]]): Frozen variables of the
                dynamics. Defaults to None. At most one argument is supported.
            log0 (float): Initial value of the log-determinant of the Jacobian.
                Defaults to 0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The evolved state and the
            accumulated log-determinant of the Jacobian.
        """

        frozen_var = args

        params = self.func.params_

        var, logj = LieAdjointWrapper_.apply(
            self.odeints, self.func, self.t_span, var, frozen_var, *params
        )
        return var, logj + log0

    def reverse(self, var, args=None, log0=0):
        """
        Evolves the state backward in time and accumulates the log-Jacobian.

        Args:
            var (torch.Tensor): Final state variable to evolve backward.
            args (Optional[Tuple[torch.Tensor]]): Frozen variables of the
                dynamics. Defaults to None. At most one argument is supported.
            log0 (float): Initial value of the log-determinant of the Jacobian.
                Defaults to 0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The reversed state and the
            accumulated log-determinant of the Jacobian.
        """

        frozen_var = args

        params = self.func.params_

        var, logj = LieAdjointWrapper_.apply(
           self.odeints, self.func, self.t_span[::-1], var, frozen_var, *params
        )
        return var, logj + log0


# =============================================================================
class LieAdjointWrapper_(torch.autograd.Function):
    """
    A custom autograd Function to perform ODE integration using the adjoint
    method. This wraps around `odeint`, allowing gradients to flow through the
    integration.
    """

    @staticmethod
    def forward(ctx, odeints, func, t_span, var, frozen_var, *params):
        """
        Forward pass using the ODE solver.

        Args:
            ctx: Autograd context for saving information for backward pass.
            odeint: The ODE integration function.
            func: A DynamicsAdjModule instance defining the ODE system.
            t_span: Time span for integration.
            var: Initial state variable.
            frozen_var: Auxiliary variables, potentially requiring gradients.
            *params: Additional parameters to differentiate with respect to.

        Returns:
            var: The integrated state.
            logj: Accumulated log-Jacobian determinant, if applicable.

        NOTE:
            1. `frozen_var` must be a tensor if its gradient is needed.
            2. `*params` should be given explicitely in order to calculate the
            derivatives with respect to them.
        """
        assert isinstance(func, AdjLieModule), \
            ("Expected `func` to be an instance of AdjLieModule")

        # Perform ODE integration
        var, logj = odeints[0](
            func.forward,
            t_span,
            var,
            args=frozen_var,
            loss_rate=func.calc_logj_rate
        )

        # Save necessary tensors for the backward pass
        ctx.odeints = odeints
        ctx.func = func
        ctx.t_span = t_span
        ctx.save_for_backward(var, logj, frozen_var, *params)

        return var, logj

    @staticmethod
    def backward(ctx, grad_var, grad_logj):
        """Evaluated by integrating the augmented system backwards in time."""
        # grad_{var, logj} are $\bar var$ and $\bar logj$ in terminology of AD
        # grad_{var} is $\lambda$ in the terminology of adjoint method
        # grad_{var} input/output are the $\lambda$ at terminal/initial times

        func = ctx.func
        odeints = ctx.odeints
        t_span = ctx.t_span
        var, logj, frozen_var, *params = ctx.saved_tensors

        # Instead of calculating `grad_var`, for lie group, we first calculate
        # `grad_alg_var`, which is the `grad` for the algebra variable, and in
        # the end we convert it to `grad_var`.
        # `\grad_var` is $\lambda$
        # `\grad_alg_var` is $\Lambda = \lambda U^\dagger$
        grad_alg_var = grad_var @ var.adjoint()

        # Define augmented variable, which will flow backward in time:
        aug_var = TupleVar(var, grad_alg_var)
        # Define augmented frozen variable:
        aug_frozen_var = TupleVar(frozen_var, grad_logj, *params)

        fzn = frozen_var
        if len(params) == 0 and (fzn is None or not fzn.requires_grad):
            aug_var = odeints[1](
                func.aug_reverse, t_span[::-1], aug_var, args=aug_frozen_var
                )
            aug_loss = None
        else:
            aug_var, aug_loss = odeints[1](
                func.aug_reverse, t_span[::-1], aug_var, args=aug_frozen_var,
                loss_rate=func.calc_grad_params_rate
                )

        var, grad_alg_var = aug_var.tuple
        # grad_alg_var must be already anti-hermitian, yet we project it again.
        grad_var = anti_hermitian(grad_alg_var) @ var

        if aug_loss is None:
            grad_frozen_var, grad_params = None, ()
        elif (fzn is None or not fzn.requires_grad):
            grad_frozen_var, grad_params = None, aug_loss.tuple
        else:
            grad_frozen_var, *grad_params = aug_loss.tuple

        return None, None, None, grad_var, grad_frozen_var, *grad_params


def anti_hermitian(mtrx):
    """Returns the anit-Hermitian part of the input matrix."""
    return (mtrx - mtrx.adjoint()) / 2.


# =============================================================================
class AdjLieModule(torch.nn.Module, ABC):
    """
    Abstract base class for ODE systems with adjoint backpropagation and
    log-Jacobian tracking, useful for `AdjLieODEflow_`.

    This class is a subclass of `torch.nn.Module` and provides the necessary
    structure for defining ODE systems that can be solved using adjoint-based
    differentiation. It is used in conjunction with modules like
    `AdjLieODEflow_` to compute gradients, perform augmented reverse
    integration, and calculate the log-Jacobian rate of the flow.

    Integrate a system of ODEs of the form::

        dU / dt = F(t, U; p) U

    where `U = U(t)` belongs to a Lie group and `F(t, U; p)` belongs to
    corresponding algebra. Here `t` is the flow time and `p` is used to
    denote fixed parameters that specify the flow (dynamic) of the system.

    The dynamics governing the algebra variable, i.e., `F(t, U; p)` is
    supposed to be defined in the abstract `algebra_dynamics` method.

    Key methods:
        - `forward`: Computes the time derivative of the state group variable.
        - `algbebra_dynamics`: Computes the time derivative of the state
          algebra variable (abstract).
        - `calc_logj_rate`: Computes the log-Jacobian rate using the Hutchinson
          estimator.
        - `aug_reverse`: Performs augmented reverse integration for adjoint
          backpropagation.
        - `calc_grad_params_rate`: Computes the gradient of parameters.

    Args:
        num_samples (int, optional): Number of samples used for the Hutchinson
          estimator to approximate the Jacobian trace. If `None`, the Jacobian
          trace is computed exactly. Defaults to 1.
    """

    def __init__(self, num_samples=1):
        super().__init__()
        self.num_samples = num_samples

    def forward(self, t, var, *frozen_var):
        """The function defining the evolution of the state variable."""
        return self.algebra_dynamics(t, var, *frozen_var) @ var

    @abstractmethod
    def algebra_dynamics(self, t, var, *frozen_var):
        """The function defining `F(t, U; p)`."""

    def calc_logj_rate(self, t, var, *frozen_var):
        """
        Computes and returns the log-Jacobian rate of the system's flow using
        the Hutchinson estimator to approximate the trace of `df/dx` for volume
        scaling.

        Args:
            t (float): Current time.
            var (torch.Tensor): Current state variable.
            *frozen_var: Additional frozen variables for the system's dynamics.

        Returns:
            torch.Tensor: The estimated log-Jacobian rate of the flow.
        """
        logj_rate = hutchinson_estimator(
            lambda x: self.forward(t, x, *frozen_var),
            var,
            self.num_samples
        )
        return logj_rate

    def aug_reverse(self, t, aug_var, aug_frozen_var):
        """Here `aug_var` contains the state variable and the adjoint state
        variable at time `t`. Similarly, `aug_frozen_var` contains the frozen
        variables, `grad_logj`, defined as gradient of the loss function w.r.t.
        `logj`, and the parameters of the model.
        """
        var, grad_alg_var = aug_var.tuple
        frozen_var, grad_logj, *params = aug_frozen_var.tuple

        with torch.enable_grad():
            var = var.detach().requires_grad_(True)

            if frozen_var is None:
                alg_var_dot = self.algebra_dynamics(t, var)
                logj_dot = self.calc_logj_rate(t, var)
            else:
                alg_var_dot = self.algebra_dynamics(t, var, frozen_var)
                logj_dot = self.calc_logj_rate(t, var, frozen_var)

            hamilton = torch.sum(
                grad_logj * logj_dot + tie_adjoints(grad_alg_var, alg_var_dot)
            )

            grad_var_dot, = \
                torch.autograd.grad(-hamilton, (var,), retain_graph=False)

        # Note that:
        #    `var_dot = alg_var_dot @ var`
        #    `grad_alg_var_dot = grad_var_dot @ var.adjoint()`

        return TupleVar(alg_var_dot @ var, grad_var_dot @ var.adjoint())

    def calc_grad_params_rate(self, t, aug_var, aug_frozen_var):
        """Computes the rate of gradeint with respect to the parameters."""
        var, grad_alg_var = aug_var.tuple
        frozen_var, grad_logj, *params = aug_frozen_var.tuple

        if (frozen_var is not None) and frozen_var.requires_grad:
            frozen_var = frozen_var.detach().requires_grad_(True)
            params = (frozen_var, *params)

        with torch.enable_grad():
            var = var.detach()

            if frozen_var is None:
                alg_var_dot = self.algebra_dynamics(t, var)
                logj_dot = self.calc_logj_rate(t, var)
            else:
                alg_var_dot = self.algebra_dynamics(t, var, frozen_var)
                logj_dot = self.calc_logj_rate(t, var, frozen_var)

            hamilton = torch.sum(
                grad_logj * logj_dot + tie_adjoints(grad_alg_var, alg_var_dot)
            )

            grad_params_rate = torch.autograd.grad(
                -hamilton, params, retain_graph=False, materialize_grads=True
            )

        return TupleVar(*grad_params_rate)

    @property
    def params_(self):
        """Returns all parameters of the module as a list."""
        return [par for par in self.parameters() if par.requires_grad]


class AdjLieModuleWrapper(AdjLieModule):
    """
    A wrapper class for a function that defines the ODE system dynamics,
    enabling adjoint-based backpropagation with log-Jacobian tracking.

    This class wraps a function that computes the time derivative of the
    state variable in an ODE system. It provides the `forward` method
    directly from the wrapped function. If the wrapped function does not
    implement `calc_logj_rate`, the methods from `DynamicsAdjModule`, such
    as `calc_logj_rate`, are inherited.

    Args:
        func (Callable): A function `f(t, y, *args)` that computes the time
            derivative of the state variable `y` at time `t`.
        num_samples (int, optional): The number of samples for the Hutchinson
            estimator when computing the Jacobian trace. Defaults to 1.

    Methods:
        - `forward`: Directly uses the provided function to compute the time
          derivative of the state variable.
        - `calc_logj_rate`: Inherited from `DynamicsAdjModule` if the wrapped
          function doesn't define a `calc_logj_rate` method. If the function
          has its own `calc_logj_rate`, that will be used.
    """

    def __init__(self, func, num_samples=1):
        super().__init__(num_samples)
        self.func = func

        # If the function has its own `calc_logj_rate`, use it.
        if hasattr(func, 'calc_logj_rate'):
            self.calc_logj_rate = func.calc_logj_rate

    def forward(self, t, var, *args):
        """
        Computes the time derivative of the state variable by calling the
        wrapped function.

        This method simply delegates the call to the wrapped `func` to compute
        the time derivative of the state variable `var` at time `t`.

        Args:
            t (float): The current time.
            var (torch.Tensor): The current state variable.
            *args: Additional arguments passed to the wrapped function.

        Returns:
            torch.Tensor: The time derivative of the state variable.
        """
        return self.func(t, var, *args)

    def algebra_dynamics(self, t, var, *args):
        return self.func(t, var, *args) @ var.adjoint()
