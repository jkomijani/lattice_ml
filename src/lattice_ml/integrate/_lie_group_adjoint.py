# Copyright (c) 2024-2025 Javad Komijani

"""
Defines the AdjLieODEFlow_ module for adjoint-based ODE integration with
log-Jacobian tracking.
"""

from abc import abstractmethod, ABC
from functools import partial as ftpartial

import torch

from ._lie_group_odeint import lie_odeint
from ._adjoint import TupleVar
from ._adjoint import tie_adjoints
from ._hutchinson_estimator import hutchinson_estimator


# =============================================================================
class AdjLieODEFlow_(torch.nn.Module):  # pylint: disable=invalid-name
    """
    A module for solving ODEs with adjoint-based backprop and log-Jacobian
    tracking.

    AdjODEFlow_ is similar to `ODEFlow_`, which extends `ODEFlow` by also
    returning the log-determinant of the Jacobian of the flow. While `ODEFlow`
    only evolves the state variable, `ODEFlow_` and `AdjODEFlow_` additionally
    compute the log-Jacobian of the transformation.

    This class uses the adjoint method to compute gradients during backward
    passes, which is memory efficient for long sequences.

    If `func` defines a method `calc_logj_rate`, it is used directly to compute
    the log-Jacobian rate. Otherwise, the trace of the Jacobian is estimated
    using the Hutchinson estimator with a given number of random samples.

    If `num_hutchinson_samples` is None, the Hutchinson estimator is not used.
    Instead, the Jacobian trace is computed exactly via automatic
    differentiation, which can be computationally expensive in high dimensions.

    If `func` is not an instance of `DynamicsAdjModule`, it will automatically
    be wrapped with `DynamicsAdjWrapper`.

    Args:
        - func (Callable): A function `f(t, y, *args)` that computes the time
          derivative of the state variable `y` at time `t`.
        - t_span (Tuple[float, float]): A tuple specifying initial and final
          times.
        - num_hutchinson_samples (Optional[int | None]): The number of random
          samples used in the Hutchinson estimator. If None, the Jacobian trace
          is computed exactly. Defaults to 1.
        - **odeint_kwargs: Additional keyword arguments for the ODE solver.

    Warning:
        This implementation is currently tailored for flows on the Lie group
        of special unitary matrices (SU(n)). It assumes that the dynamics are
        defined using anti-Hermitian, traceless matrices.

        For support with general unitary matrices (U(n)), refer to the
        documentation of the `LieAdjointWrapper_` class used in this flow,
        which outlines the required changes.
    """

    def __init__(
        self, func, t_span, num_hutchinson_samples=1,
        methods=('RK4:SU(n)', 'RK4:SU(n):aug'),
        **odeint_kwargs
    ):
        """Initializes the AdjLieODEFlow_ module."""

        super().__init__()

        if isinstance(func, AdjLieModule):
            self.func = func
        else:
            self.func = AdjLieModuleWrapper(func, num_hutchinson_samples)

        self.t_span = t_span

        self.odeints = [
            ftpartial(lie_odeint, method=methods[0], **odeint_kwargs),
            ftpartial(lie_odeint, method=methods[1], **odeint_kwargs)
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

        # Normalize args to a tuple
        if args is None:
            args = ()
        elif not isinstance(args, tuple):
            args = (args,)

        frozen_var = args

        params = self.func.params_

        var, logj = LieAdjointWrapper_.apply(
            self.odeints, self.func, self.t_span, var,
            len(frozen_var), *frozen_var, *params
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

        # Normalize args to a tuple
        if args is None:
            args = ()
        elif not isinstance(args, tuple):
            args = (args,)

        frozen_var = args

        params = self.func.params_

        var, logj = LieAdjointWrapper_.apply(
           self.odeints, self.func, self.t_span[::-1], var,
           len(frozen_var), *frozen_var, *params
        )
        return var, logj + log0


# =============================================================================
# pylint: disable=invalid-name
class LieAdjointWrapper_(torch.autograd.Function):
    """
    A custom autograd Function to perform ODE integration using the adjoint
    method. This wraps around `odeint`, allowing gradients to flow through the
    integration.

    Warning:
        This implementation is currently tailored for flows on the Lie group
        of special unitary matrices (SU(n)). It assumes that the dynamics are
        defined using anti-Hermitian, traceless matrices.

        To support general unitary matrices (U(n)), you will need to replace
        the hardcoded use of `anti_hermitian_traceless` in the backward pass
        with `anti_hermitian`. We plan to extend the implementation to support
        both cases when needed.
    """

    @staticmethod
    def forward(ctx, odeints, func, t_span, var, n_frozen_var, *all_args):
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

        frozen_var = all_args[:n_frozen_var]
        params = all_args[n_frozen_var:]  # pylint: disable=unused-variable

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
        ctx.n_frozen_var = n_frozen_var
        ctx.save_for_backward(var, logj, *all_args)

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
        # pylint: disable=unused-variable  # for logj
        var, logj, *all_args = ctx.saved_tensors

        frozen_var = all_args[:ctx.n_frozen_var]
        params = all_args[ctx.n_frozen_var:]

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

        # Get mask for frozen_var that require gradients
        fzn_requiring_grad = [item.requires_grad for item in frozen_var]

        if len(params) + sum(fzn_requiring_grad) == 0:
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
        # Although grad_alg_var should ideally be anti-Hermitian, in practice
        # it often isn't. We project it again to enforce anti-Hermitian
        # structure, ensuring numerical stability and consistency.
        #
        # Warning: We use `anti_hermitian_traceless`, which is appropriate only
        # for special unitary matrices (i.e., matrices with determinant 1).
        # For general unitary matrices, use `anti_hermitian` instead.
        grad_var = anti_hermitian_traceless(grad_alg_var) @ var

        # Initialize all frozen gradients as None
        grad_frozen_var = [None] * len(frozen_var)
        grad_params = ()

        if aug_loss is not None:
            # Extract gradient chunks
            n = sum(fzn_requiring_grad)
            grad_fzn = iter(aug_loss.tuple[:n])
            grad_params = aug_loss.tuple[n:]

            # Assign gradients to positions requiring grad
            grad_frozen_var = [
                next(grad_fzn) if req else None for req in fzn_requiring_grad
            ]

        return None, None, None, grad_var, None, *grad_frozen_var, *grad_params


def anti_hermitian(mtrx):
    """Returns the anit-Hermitian part of the input matrix."""
    return (mtrx - mtrx.adjoint()) / 2.


def anti_hermitian_traceless(mtrx: torch.Tensor) -> torch.Tensor:
    """
    Project a square matrix (or batch of matrices) onto the Lie algebra su(n).

    This function returns an anti-Hermitian, traceless version of the input
    matrix `mtrx`.
    """
    # Make anti-Hermitian
    mtrx = (mtrx - mtrx.adjoint()) / 2.

    # Compute average diagonal value (trace / n) over the last two axes
    reduced_trace = mtrx.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True)

    # Subtract the average from the diagonal to make it traceless
    return mtrx - torch.diag_embed(reduced_trace.expand(mtrx.shape[:-1]))


# =============================================================================
class AdjLieModule(torch.nn.Module, ABC):
    """
    Abstract base class for ODE systems with adjoint backpropagation and
    log-Jacobian tracking, useful for `AdjLieODEFlow_`.

    This class is a subclass of `torch.nn.Module` and provides the necessary
    structure for defining ODE systems that can be solved using adjoint-based
    differentiation. It is used in conjunction with modules like
    `AdjLieODEFlow_` to compute gradients, perform augmented reverse
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
        - num_hutchinson_samples (Optional[int | None]): The number of random
          samples used in the Hutchinson estimator. If None, the Jacobian trace
          is computed exactly. Defaults to 1.
    """

    def __init__(self, num_hutchinson_samples=1):
        super().__init__()
        self.num_hutchinson_samples = num_hutchinson_samples

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
        n_samples = self.num_hutchinson_samples
        # Use a mask of shape (n, n, 2) where 2 is for complex numbers
        if n_samples is None or n_samples % (2 * var.shape[-1]**2) != 0:
            noise_mask_ndim = 0
        else:
            noise_mask_ndim = 2

        logj_rate = hutchinson_estimator(
            lambda x: self.forward(t, x, *frozen_var),
            var,
            n_samples,
            noise_mask_ndim=noise_mask_ndim
        )
        return logj_rate

    def aug_reverse(self, t, aug_var, aug_frozen_var):
        """
        Computes the reverse-time dynamics for the augmented system
        in the adjoint method.

        This method calculates the time derivatives of both the original state
        variable and its corresponding adjoint state variable.
        For the state variable `self.algebra_dynamics` is called,
        and for the adjoint state variable automatic differentiation is used.

        Args:
            t (torch.Tensor): Current time step (scalar tensor).
            aug_var (TupleVar): Tuple of (state, adjoint of state) at time t.
            aug_frozen_var (TupleVar): Tuple containing:
                - frozen variables (constants during backward pass),
                - grad_logj: gradient of the loss w.r.t. log-Jacobian,
                - model parameters.

        Returns:
            TupleVar: A tuple of:
                - Time derivative of the (group-valued) state variable
                  (var_dot),
                - Time derivative of the (algebra-valued) adjoint variable
                  (grad_alg_var_dot).
        """
        var, grad_alg_var = aug_var.tuple
        # pylint: disable=unused-variable  # for params
        frozen_var, grad_logj, *params = aug_frozen_var.tuple

        with torch.enable_grad():
            # Ensure var can track gradients
            var = var.detach().requires_grad_(True)

            # Forward-mode derivatives
            alg_var_dot = self.algebra_dynamics(t, var, *frozen_var)
            logj_dot = self.calc_logj_rate(t, var, *frozen_var)

            # Hamiltonian combines contributions from log-Jacobian and state
            hamilton = torch.sum(
                grad_logj * logj_dot + tie_adjoints(grad_alg_var, alg_var_dot)
            )

            # Compute gradient of Hamiltonian w.r.t. state variable
            grad_var_dot, = torch.autograd.grad(
                -hamilton, (var,), retain_graph=False
            )

        # Compute time derivatives for group-valued state & algebra-valued
        # adjoint state, using:
        #   var_dot = alg_var_dot @ var
        #   grad_alg_var_dot = grad_var_dot @ var.adjoint()

        return TupleVar(alg_var_dot @ var, grad_var_dot @ var.adjoint())

    def calc_grad_params_rate(self, t, aug_var, aug_frozen_var):
        """
        Computes the rate of change of gradients with respect to parameters
        and frozen variables if they require gradients.

        Forms the Hamiltonian from the adjoint variables and system dynamics,
        then computes the gradient of the negative Hamiltonian with respect
        to model parameters.

        Args:
            t (torch.Tensor): Current time step (scalar tensor).
            aug_var (TupleVar): Tuple containing state and adjoint variables.
            aug_frozen_var (TupleVar): Tuple containing frozen variables,
                adjoint of the log-Jacobian, and model parameters.

        Returns:
            TupleVar: Gradient of the Hamiltonian w.r.t. parameters and frozen
            variables if they require gradients.
        """
        var, grad_alg_var = aug_var.tuple
        frozen_var, grad_logj, *params = aug_frozen_var.tuple

        def reenable_grad(x):
            return x.detach().requires_grad_(True) if x.requires_grad else x

        # Detach frozen vars but re-enable grad tracking if needed
        frozen_var = [reenable_grad(fzn) for fzn in frozen_var]

        # Include frozen vars requiring grad into parameter list; order matters
        for fzn in frozen_var[::-1]:
            if fzn.requires_grad:
                params = (fzn, *params)

        with torch.enable_grad():
            var = var.detach()

            # Forward dynamics and log-Jacobian rate
            alg_var_dot = self.algebra_dynamics(t, var, *frozen_var)
            logj_dot = self.calc_logj_rate(t, var, *frozen_var)

            # Compute Hamiltonian combining adjoints and dynamics
            hamilton = torch.sum(
                grad_logj * logj_dot + tie_adjoints(grad_alg_var, alg_var_dot)
            )

            # Gradient of negative Hamiltonian w.r.t. parameters
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
        - func (Callable): A function `f(t, y, *args)` that computes the time
          derivative of the state variable `y` at time `t`.
        - num_hutchinson_samples (Optional[int | None]): The number of random
          samples used in the Hutchinson estimator. If None, the Jacobian trace
          is computed exactly. Defaults to 1.

    Methods:
        - `forward`: Directly uses the provided function to compute the time
          derivative of the state variable.
        - `calc_logj_rate`: Inherited from `DynamicsAdjModule` if the wrapped
          function doesn't define a `calc_logj_rate` method. If the function
          has its own `calc_logj_rate`, that will be used.
    """

    def __init__(self, func, num_hutchinson_samples=1):
        super().__init__(num_hutchinson_samples)
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
