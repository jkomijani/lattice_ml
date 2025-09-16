# Copyright (c) 2025 Javad Komijani

"""
Symplectic ODE Solver Utilities with adjoint-based backpropagation

This module evolves systems governed by Hamiltonian (or more generally,
symplectic) dynamics using a structure-preserving ODE solver. Gradients
are computed efficiently via the adjoint method, enabling memory-efficient
backpropagation through long trajectories.

Main Interface
--------------
- `adj_symplectic_odeint`: Integrates Hamiltonian system from (p0, q0).
"""

# pylint: disable=relative-beyond-top-level, arguments-differ, too-many-locals
# pylint: disable=too-many-arguments, too-many-positional-arguments

from typing import Callable, Tuple
from abc import abstractmethod, ABC
from functools import partial as ftpartial

import torch

from ._symplectic_odeint import symplectic_odeint
from ._adjoint import TupleVar


__all__ = ["adjoint_symplectic_odeint"]


# =============================================================================
def adjoint_symplectic_odeint(
    force_fn: Callable,
    t_span: Tuple[float, float],
    p0: torch.Tensor,
    q0: torch.Tensor,
    args: any = None,
    velocity_fn: Callable | None = None,
    **odeint_kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Solve an initial value problem for a system of symplectic ODEs using
    adjoint-based backpropagation.

    The system state is represented by a tuple (p, q), where p is momentum and
    q is position. The dynamics follow a canonical symplectic form:

        dq/dt = ∂H/∂p = p
        dp/dt = -∂H/∂q = force_fn(t, q, *args)

    where `force_fn` models the generalized force acting on the system.
    While the force may not be derived from a Hamiltonian, we adopt canonical
    notation for consistency with the symplectic integration literature.

    Optionally, one can override the standard velocity–momentum relation by
    supplying a custom `velocity_fn`. This allows modeling systems where
    velocity is not simply equal to momentum, e.g. in non-canonical coordinates
    or generalized mechanical systems. In this case, the position dynamics are:

        dq/dt = velocity_fn(t, p, *args)

    This routine wraps a symplectic ODE solver such as `symplectic_odeint` and
    supports automatic differentiation through the integration.
    Gradients w.r.t. initial conditions and parameters are computed efficiently
    via the adjoint method by solving an augmented system backward in time.

    If `force_fn` is not already an instance of `AdjSymplecticModule`, it will
    be wrapped automatically using `AdjSymplecticModuleWrapper` to provide the
    necessary adjoint interface.

    Parameters
    ----------
    force_fn : callable
        Function modeling the momentum dynamics, typically -∂H/∂q.
        Signature: force_fn(t, q, *args), where t is time, q is position, and
        args are optional frozen arguments.

    t_span : tuple of float
        The time interval (t0, t1) over which to integrate.

    p0 : torch.Tensor
        Initial momentum.

    q0 : torch.Tensor
        Initial position.

    args : tuple or any or None, optional
        Additional arguments passed to force_fn.

    velocity_fn: callable, optional
        Function modeling the position dynamics.
        Signature: velocity_fn(t, p, *args).
        If not provided, defaults to the canonical choice dq/dt = p.

    **odeint_kwargs : dict
        Extra keyword arguments passed to the underlying symplectic integrator.
        Typical options include:
            - step_size: float, time step size (default: 1e-3).
            - num_steps: int, if given, overrides step_size.
            - method: str, integration method such as leapfrog.

    Returns
    -------
    final_p : torch.Tensor
        Final momentum after integration.

    final_q : torch.Tensor
        Final position after integration.

    Note:
        This implementation is intentionally simple and not optimized.
        In principle, efficiency could be improved by separating the parameters
        of `force_fn` and `velocity_fn`, and using only the relevant subsets
        when computing derivatives. Currently, all parameters are treated
        together, leading to unnecessary computations.
    """

    # Ensure force_fn supports adjoint-based differentiation
    if not isinstance(force_fn, AdjSymplecticModule):
        force_fn = AdjSymplecticModuleWrapper(force_fn)

    if velocity_fn is not None:
        velocity_fn = VelocityAdjSymplecticModuleWrapper(velocity_fn)

    # Bind integration options
    odeint = ftpartial(symplectic_odeint, **odeint_kwargs)

    # Use custom autograd-enabled symplectic integrator
    p, q = SymplecticAdjointWrapper.apply(
        odeint, force_fn, velocity_fn, t_span, p0, q0,
        *get_all_frozen_and_differentiable_items(args, force_fn, velocity_fn)
    )

    # Return position and momentum at the terminal time
    return p, q


def get_all_frozen_and_differentiable_items(args, force_fn, velocity_fn):
    """
    Internal helper to unpack frozen arguments and trainable parameters.

    Returns:
        Tuple[int, *args, *params]: A tuple where the first element is the
        number of frozen arguments of the dynamics, followed by all trainable
        parameters.
    """
    # Normalize args to tuple
    if args is None:
        args = ()
    elif not isinstance(args, tuple):
        args = (args,)

    frozen_args = args
    trainable_params = [p for p in force_fn.parameters() if p.requires_grad]
    if velocity_fn is not None:
        vel_params = [p for p in velocity_fn.parameters() if p.requires_grad]
        trainable_params.extend(vel_params)
    return (len(frozen_args), *frozen_args, *trainable_params)


# =============================================================================
class SymplecticAdjointWrapper(torch.autograd.Function):
    """
    A custom autograd Function for symplectic ODE integration using
    adjoint-based backpropagation.

    This wrapper enables memory-efficient gradient computation through
    symplectic ODE solvers by integrating an augmented system backward in time.
    It is specifically designed for symplectic systems whose dynamics are
    defined by an `AdjSymplecticModule` instance.

    The original forward system evolves the canonical state variables (p, q)
    using a structure-preserving integrator. In the backward pass, the adjoint
    equations are constructed and integrated in reverse time to compute
    gradients with respect to the initial conditions and all differentiable
    inputs (e.g., learnable parameters or frozen variables).

    Forward:
        Integrates the ODE system over the given time span.

    Backward:
        Computes gradients by solving the adjoint equations backward in time.
    """

    @staticmethod
    def forward(
        ctx,
        odeint, force_fn, velocity_fn, t_span, p, q, n_frozen_args, *all_args
    ):
        """
        Performs the forward pass by integrating the symplectic ODE system.

        Args:
            ctx: Autograd context for saving information for backward pass.
            odeint (Callable): Symplectic ODE integrator (e.g., Verlet).
            force_fn (AdjSymplecticModule): Module defining the dynamics.
            t_span (Tuple[float, float]): Time interval for integration.
            p (Tensor): Initial momentum.
            q (Tensor): Initial position.
            n_frozen_args (int): Number of frozen arguments in all_args.
            *all_args: Tuple of frozen arguments and parameters for force_fn.

        Returns:
            Tuple[Tensor, Tensor]: Final momentum and position.
        """
        assert isinstance(force_fn, AdjSymplecticModule), (
            "Expected `force_fn` to be an instance of AdjSymplecticModule"
        )

        frozen_args = all_args[:n_frozen_args]

        # Perform ODE integration
        p, q = odeint(
            force_fn.forward, t_span, p, q,
            velocity_fn=velocity_fn and velocity_fn.forward, args=frozen_args
        )

        # Save tensors and meta info needed for backward pass
        ctx.odeint = odeint
        ctx.force_fn = force_fn
        ctx.velocity_fn = velocity_fn
        ctx.t_span = t_span
        ctx.n_frozen_args = n_frozen_args
        ctx.save_for_backward(p, q, *all_args)

        return p, q

    @staticmethod
    def backward(ctx, grad_p, grad_q):
        """
        Computes the backward pass using the adjoint method.

        Given gradients of a scalar loss with respect to the final states
        (p, q), this function computes gradients w.r.t:
            - initial momentum p0 and position q0,
            - differentiable items in `all_args` of the forward pass.

        Args:
            grad_p (Tensor): Gradient of the loss w.r.t. final momentum p.
            grad_q (Tensor): Gradient of the loss w.r.t. final position q.

        Returns:
            Tuple[None, None, None, grad_p0, grad_q0, None, *grad_args]

        Note:
            This implementation is intentionally simple and not optimized.
            In principle, efficiency could be improved by separating the
            parameters of `force_fn` and `velocity_fn`, and using only
            the relevant subsets when computing derivatives. Currently,
            all parameters are treated together, leading to unnecessary
            computations.
        """
        # grad_{p,q} ≡ adjoint variables (λ_p, λ_q) at terminal time
        # In AD: grad_p = \bar p = ∂Loss/∂p

        force_fn = ctx.force_fn
        velocity_fn = ctx.velocity_fn
        odeint = ctx.odeint
        t_span = ctx.t_span
        p, q, *all_args = ctx.saved_tensors

        # Extract differentiable inputs from all_args to compute their gradient
        theta = [arg for arg in all_args if arg.requires_grad]

        # Initialize the adjoint variables (gradients) for theta with zeros
        grad_theta = [torch.zeros_like(arg) for arg in theta]

        # Construct augmented initial conditions for adjoint system
        # aug_p = (p, λ_q, ∂Loss/∂θ); aug_q = (q, -λ_p, ∂Loss/∂θ)
        # We could pass only θ of force_fn/velocity_fn to aug_p/aug_q
        aug_p = TupleVar(p, grad_q, *grad_theta)
        aug_q = TupleVar(q, -grad_p, *grad_theta)

        # Pack frozen arguments and theta parameters for augmented reverse call
        frozen_args = all_args[:ctx.n_frozen_args]
        aug_frozen_args = TupleVar(frozen_args, theta)

        # Integrate the augmented system backward in time to compute gradients
        aug_p, aug_q = odeint(
            force_fn.aug_reverse, t_span[::-1], aug_p, aug_q,
            velocity_fn=velocity_fn and velocity_fn.aug_reverse,
            args=aug_frozen_args
        )

        # Unpack gradients from the augmented output
        _, grad_q0, *grad_theta = aug_p.tuple
        _, grad_p0 = aug_q.tuple[0], -aug_q.tuple[1]  # Reverse sign

        if len(aug_q.tuple) > 2:
            grad_theta = [a + b for a, b in zip(grad_theta, aug_q.tuple[2:])]

        # Match computed gradients with original inputs
        grad_all_args = align_grads_with_inputs(all_args, grad_theta)

        # Return gradients in order matching input signature
        return None, None, None, None, grad_p0, grad_q0, None, *grad_all_args


def align_grads_with_inputs(all_args, grad_theta):
    """
    Reconstructs a list of gradients aligned with all_args, inserting None
    for tensors that do not require gradients.

    Args:
        all_args (Iterable[torch.Tensor]): Original input tensors.
        grad_theta (Iterator[torch.Tensor]): Gradients corresponding to tensors
        in all_args with requires_grad=True.

    Returns:
        List[torch.Tensor | None]: A list matching the structure of all_args,
        where gradients are placed for tensors requiring gradients, and None
        elsewhere.
    """
    grad = iter(grad_theta)
    return [next(grad) if item.requires_grad else None for item in all_args]


# =============================================================================
class AdjSymplecticModule(torch.nn.Module, ABC):
    """
    Abstract base class for symplectic ODE systems with adjoint sensitivity.

    This class defines the interface and core logic for Hamiltonian or
    symplectic systems where the momentum evolution is governed by a force
    function typically representing `-∂H/∂q`. It supports adjoint-based
    backpropagation by defining the augmented reverse dynamics needed for
    efficient gradient computation.

    Key methods:
        - forward: Computes the instantaneous time derivative of the momentum
          p given the current position q and optionally frozen arguments.
        - aug_reverse: Defines the augmented system for backward-in-time
          integration, used to compute gradients via the adjoint method.

    Usage:
        Subclasses must implement the `forward` method specifying how the force
        depends on time, position, and frozen arguments.

    Args:
        None directly in constructor; subclasses may extend as needed.

    Methods:
        forward(t, q, *frozen_args):
            Abstract method. Returns time derivative of momentum p at time t.
        aug_reverse(t, aug_q, aug_frozen_args):
            Returns time derivative of augmented momentum for backward
            integration of adjoint system. The augmented momentum includes

    Notes:
        - The augmented variables include the original states and their
          adjoints.
        - The adjoint Hamiltonian is formed by coupling the adjoint variables
          with the forward dynamics, enabling calculation of gradients with
          respect to states and parameters.
    """

    @abstractmethod
    def forward(self, t, q, *frozen_args):
        """
        Computes the time derivative of the momentum p at time t.

        This function corresponds to the right-hand side of the ODE:
        `dp/dt = force_fn(t, q, *frozen_args)`

        Args:
            t (float or torch.Tensor): Current time.
            q (torch.Tensor): Current position.
            *frozen_args (tuple): Optional frozen arguments of the dynamics.

        Returns:
            torch.Tensor: The time derivative of the momentum p at time t.
        """

    def aug_reverse(self, t, aug_q, aug_frozen_args):
        r"""
        Computes the time derivative of the augmented momentum at time t
        used in backward integration for the adjoint method.

        This method takes in augmented position, containing the position q and
        also the negative of the adjoint of the momentum p as a TupleVar.
        Additional frozen arguments and parameters requiring gradients can be
        passed too as a TupleVar. This method returns the time derivatives of
        the augmented momentum, containing the momentum p, the ajdoint of the
        position q, and the adjoint of all parameters and frozen arguments that
        require gradients.

        Note:
            `TupleVar` is a lightweight container for elementwise algebraic
            operations on a tuple of variables.

        Args:
            t (float or torch.Tensor): Current time.
            aug_q (TupleVar): The augmented position, containing the position q
                and also the negative of the adjoint of the momentum p as
                `TupleVar(q, -grad_p)`.
            aug_frozen_args (TupleVar): The augmented frozen arguments,
                containing all frozen arguments and all parameters that require
                gradients.

        Returns:
            TupleVar: Time derivatives of the augmented momentum, containing
            the momentum p, the ajdoint of the position q, and the adjoint of
            all parameters and frozen arguments that require gradients.
            It is the time derivative of `TupleVar(p, grad_q, *grad_theta)`.

        Process:
            1. Compute forward dynamics `\dot p = forward(t, q, *frozen_args)`.
            2. Form the adjoint Hamiltonian coupling `-grad_p` with `\dot p`.
            3. Compute gradients of the adjoint Hamiltonian with respect to
               `q` and parameters, which gives the adjoint system dynamics.
            4. Return the augmented derivatives wrapped in `TupleVar`.
        """
        q, minus_grad_p = aug_q.tuple[:2]

        # Separate frozen arguments and all differentiable parameters
        frozen_args, theta = aug_frozen_args.tuple

        with torch.enable_grad():
            # Enable gradient tracking on q for adjoint computation
            q = q.detach().requires_grad_(True)

            # Compute momentum dynamics
            p_dot = self.forward(t, q, *frozen_args)

            # Construct adjoint Hamiltonian for sensitivity analysis
            hamilton = -torch.sum(minus_grad_p.conj() * p_dot)

            # Compute gradients (adjoint dynamics) w.r.t. q and parameters
            grad_q_dot, *grad_theta_dot = torch.autograd.grad(
                -hamilton,
                (q, *theta),
                retain_graph=False,
                materialize_grads=True
            )

        # Return augmented derivatives: forward dynamics and adjoint dynamics
        aug_p_dot = TupleVar(p_dot, grad_q_dot, *grad_theta_dot)

        return aug_p_dot


class AdjSymplecticModuleWrapper(AdjSymplecticModule):
    """
    A wrapper class for a function that defines the ODE system dynamics,
    enabling adjoint-based backpropagation.
    """

    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, t, q, *frozen_args):
        """
        Computes the instantaneous time derivative of the momentum p at time t
        by calling the wrapped function.

        This function corresponds to the right-hand side of the ODE:
        `dp/dt = force_fn(t, q, *frozen_args)`

        Args:
            t (float or torch.Tensor): Current time.
            q (torch.Tensor): Current position.
            *frozen_args (tuple): Optional frozen variables.

        Returns:
            torch.Tensor: The time derivative of the momentum `p` at time t.
        """
        return self.func(t, q, *frozen_args)


class VelocityAdjSymplecticModuleWrapper(AdjSymplecticModuleWrapper):
    """A wrapper class suitable for `velocity_fn`."""

    def aug_reverse(self, t, aug_p, aug_frozen_args):
        r"""
        Computes the time derivative of the augmented position at time t
        used in backward integration for the adjoint method.

        This method takes in augmented momentum, containing the momentum p,
        the adjoint of the position q, and also the adjoint of all parameters
        and frozen arguments that require gradients as a TupleVar.

        Additional frozen arguments and parameters requiring gradients can be
        passed too as a TupleVar. This method returns the time derivatives of
        the augmented position, containing the position q and the negative of
        the ajdoint of the momentum p.

        Note:
            `TupleVar` is a lightweight container for elementwise algebraic
            operations on a tuple of variables.

        Args:
            t (float or torch.Tensor): Current time.
            aug_p (TupleVar): The augmented momentum, containing the momentum
                p, the adjoint of the position q, and also the adjoint of all
                parameters and frozen arguments that require gradients as
                `TupleVar(p, grad_q, grad_theta)`.
            aug_frozen_args (TupleVar): The augmented frozen arguments,
                containing all frozen arguments and all parameters that require
                gradients.

        Returns:
            TupleVar: Time derivatives of the augmented position, containing
            the position q and the negative of the ajdoint of the momentum p.
            It is the time derivative of `TupleVar(q, -grad_p)`.

        Process:
            1. Compute forward dynamics `\dot q = forward(t, p, *frozen_args)`.
            2. Form the adjoint Hamiltonian coupling `grad_q` with `\dot q`.
            3. Compute gradients of the adjoint Hamiltonian with respect to
               `q` and parameters, which gives the adjoint system dynamics.
            4. Return the augmented derivatives wrapped in `TupleVar`.
        """
        p, grad_q = aug_p.tuple[:2]

        # Separate frozen arguments and all differentiable parameters
        frozen_args, theta = aug_frozen_args.tuple

        with torch.enable_grad():
            # Enable gradient tracking on q for adjoint computation
            p = p.detach().requires_grad_(True)

            # Compute momentum dynamics
            q_dot = self.forward(t, p, *frozen_args)

            # Construct adjoint Hamiltonian for sensitivity analysis
            hamilton = torch.sum(grad_q.conj() * q_dot)

            # Compute gradients (adjoint dynamics) w.r.t. p and parameters
            grad_p_dot, *grad_theta_dot = torch.autograd.grad(
                -hamilton,
                (p, *theta),
                retain_graph=False,
                materialize_grads=True
            )

        # Return augmented derivatives: forward dynamics and adjoint dynamics
        aug_q_dot = TupleVar(q_dot, -grad_p_dot, *grad_theta_dot)

        return aug_q_dot
