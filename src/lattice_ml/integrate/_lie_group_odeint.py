# Copyright (c) 2024-2025 Javad Komijani

"""
Lie group ODE integration methods.

This module provides functionality for numerically integrating ordinary
differential equations (ODEs) defined on Lie groups, such as SU(n).
It includes the `lie_odeint` function, which acts as a wrapper around
a general-purpose ODE integrator (`odeint`) by selecting Lie-group-aware
integration methods and passing them as custom step functions.

Available integration methods include:
- "RK4:SU(n)", Standard Runge-Kutta 4 on SU(n)
- "RK4:SU(n):aug", Augmented RK4 method
- "RK3:su(n):auto", Third-order autonomous RK on Lie groups
- "Euler:SU(n)", Basic Euler method adapted to Lie groups
- "Euler:SU(n):aug", Augmentedc Euler method adapted to Lie groups
- "Euler:su(n):aug", Augmentedc Euler method adapted to Lie groups

The following methods are mainly for testing:
- "RK4:SU(n):grad-projected", A version of "RK4:SU(n)" mainly for tests.
- "Euler:SU(n):grad-projected", A version of "Euler:SU(n)" mainly for tests.
"""

from typing import Callable, Tuple, Union
import torch

from lattice_ml.linalg import eigh
from lattice_ml.linalg import project_grad_sun

from ._odeint import odeint
from ._adjoint import TupleVar


__all__ = ["lie_odeint"]


eye3x3 = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


# =============================================================================
def lie_odeint(
    func: Callable,
    t_span: Tuple[float, float],
    y0: torch.Tensor,
    method: str = "RK4:SU(n)",
    **odeint_kwargs
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""
    Integrate a system of ODEs on a Lie group using a Lie-specific step method.

    This function solves differential equations of the form:

        dU / dt = f(t, U; p) = F(t, U; p) @ U

    where `U = U(t)` evolves on a Lie group (e.g., SU(n)) and `F(t, U; p)`
    belongs to its Lie algebra. Here, `t` is time and `p` represents
    fixed parameters defining the system dynamics.

    This is a wrapper around `odeint` that selects a Lie-group-aware method of
    integration and passes it as the `ode_step` argument. All other options are
    forwarded to `odeint`.

    Parameters
    ----------
    func : callable
        Computes the right-hand side of the ODE, i.e., `f(t, U; p)` or
        `F(t, U; p)` depending on the method.
    t_span : tuple of float
        (t0, t1), the time interval to integrate over.
    y0 : torch.Tensor
        Initial state on the Lie group (e.g., an SU(n) matrix).
    method : str, optional
        Name of the Lie group integration method. Determines which ODE step
        function is passed to `odeint` via `ode_step`. Default is "RK4:SU(n)".
    **odeint_kwargs : dict
        Additional keyword arguments passed to `odeint`, such as `args`,
        `step_size`, `num_steps`, or `loss_rate`.

    Returns
    -------
    final_state : same type as `y0`
        Final state after integration.
    loss : torch.Tensor, optional
        Accumulated loss if `loss_rate` is provided.
    """
    ode_step = _get_lie_ode_step(method)
    return odeint(func, t_span, y0, ode_step=ode_step, **odeint_kwargs)


# =============================================================================
def _get_lie_ode_step(method: str) -> Callable:
    """Return the appropriate Lie group ODE step function based on method."""
    match method:
        # SU(n):
        case 'RK4:SU(n)':
            ode_step = special_unitary_rk4_step
        case 'RK4:SU(n):aug':
            ode_step = augmented_special_unitary_rk4_step
        case 'Euler:SU(n)':
            ode_step = lie_euler_step
        case 'Euler:SU(n):aug':
            ode_step = augmented_lie_euler_step
        # su(n):
        case 'RK3:su(n):auto':
            ode_step = lie_autonomous_rk3_algebra_step
        case 'Euler:su(n)':
            ode_step = lie_euler_algebra_step
        # Mainly for tests:
        case 'RK4:SU(n):grad-projected':
            ode_step = grad_projected_special_unitary_rk4_step
        case 'Euler:SU(n):grad-projected':
            ode_step = grad_projected_lie_euler_step
        case _:
            raise ValueError(f"Lie group method '{method}' is not supported.")

    return ode_step


# =============================================================================
def augmented_special_unitary_rk4_step(func, t, var, dt, *args):
    r"""
    Generalized RK4 step for special unitary matrices with frozen variables.

    This is an extension of `special_unitary_rk4_step` that supports
    augmented systems where the primary variable evolves on the Lie group
    SU(n) and additional auxiliary variables are updated alongside with RK4
    method.

    Specifically, the function integrates systems of the form:

        dU/dt = F(t, {U, z}; p) @ U,    where U ∈ SU(n),
        dz/dt = g(t, {U, z}; p),    where z may be frozen or auxiliary.

    The update for `U` is projected back onto SU(n) using a truncated
    matrix exponential, ensuring group structure is preserved. The update
    for the auxiliary variable is handled additively (no projection).

    This method assumes that `var` is a `TupleVar(U, z)` and that
    `func(t, var, *args)` returns a `TupleVar(dU, dz)`.

    Parameters
    ----------
    func : callable
        The ODE function returning a `TupleVar` of updates `(dU, dz)`.
    t : float
        Current integration time.
    var : TupleVar
        Current state of the system, must be `TupleVar(U, z)` where
        `U` ∈ SU(n) and `z` is auxiliary.
    dt : float
        Time step.
    *args : tuple
        Optional additional arguments passed to `func`.

    Returns
    -------
    TupleVar
        Updated state `TupleVar(U_next, z_next)` after one RK4 step.
        `U_next` remains on SU(n).
    """
    # Compute RK4 deltas for SU(n) and auxiliary components
    delta, d_other = delta_from_rk4_step(func, t, var, dt, *args).tuple

    # Unpack current SU(n) variable and auxiliary variable
    var, other = var.tuple

    # Update SU(n) variable with projection to remain on the manifold
    var = construct_rk4_special_unitary(delta @ var.adjoint()) @ var

    # Update auxiliary variable via standard RK4 increment
    other = other + d_other

    return TupleVar(var, other)


# =============================================================================
def special_unitary_rk4_step(func, t, var, dt, *args):
    r"""
    Performs one Runge-Kutta-4 step preserving the special unitary property.

    This method evolves `var` in time using the classical RK4 integration
    scheme, then projects the update back onto the special unitary group SU(n).

    The RK4 method gives

    .. math::

        U_{t + dt} = U_t + \text{shift} + O(h^5)
                   = (I + \eps + O(h^5)) U_t

    We rewrite the coefficient of `U_t` as a special unitary matrix.
    Then, we multiply it to the current value `U(t)` to obtain `U(t + dt)`.

    Parameters
    ----------
    func : callable
        Function of the form `f(t, U, *args)`.
    t : float
        Current time.
    var : torch.Tensor
        The current special unitary matrix `U` of shape (..., n, n).
    dt : float
        Time step for the integration.
    *args : tuple
        Additional arguments passed to `func`.

    Returns
    -------
    torch.Tensor
        Updated special unitary matrix after one RK4 step.
    """
    # Compute the eps term from the RK4 step (Lie algebra element)
    eps = delta_from_rk4_step(func, t, var, dt, *args) @ var.adjoint()

    # Project eps back onto SU(n) and update var
    return construct_rk4_special_unitary(eps) @ var


def construct_rk4_special_unitary(eps: torch.Tensor) -> torch.Tensor:
    r"""
    Construct a special unitary matrix close to `I + eps`, using the most
    efficient method depending on matrix size and device.

    This is a high-level wrapper that dispatches to optimized routines:

    - For 3×3 matrices (shape (..., 3, 3)), it uses `naive_project_su3`,
      which is faster and sufficient for small matrices.

    - For larger matrices:
        * On CPU: uses `construct_rk4_special_unitary_type0`
          (matrix exponentiation, fast on CPU).
        * On CUDA: uses `construct_rk4_special_unitary_type1`
          (avoids slow matrix exponentiation on GPU).

    In all cases, the result is a special unitary matrix (SU(n)), accurate
    to O(eps^5).

    Parameters
    ----------
    eps : torch.Tensor
        A small perturbation matrix of shape (..., n, n).

    Returns
    -------
    torch.Tensor
        Special unitary matrix of shape (..., n, n).

    See also
    --------
    naive_project_su3
    construct_rk4_special_unitary_type0
    construct_rk4_special_unitary_type1
    """
    # Use naive_project_su3 for 3x3 matrices
    if eps.shape[-1] == 3:
        eye = eye3x3.to(eps.device).reshape(*(1,) * (eps.ndim - 2), 3, 3)
        return naive_project_su3(eps + eye)

    if eps.is_cuda:
        func = construct_rk4_special_unitary_type1
    else:
        func = construct_rk4_special_unitary_type0

    return func(eps)


def construct_rk4_special_unitary_type0(eps: torch.Tensor) -> torch.Tensor:
    r"""
    Constructs a special unitary matrix close to `I + eps`.

    This method is designed to address the following situation:
    In RK4 methods applied to SU(n)-valued variables, the naive update
    `I + eps` is only approximately special unitary.
    This function corrects it to lie exactly in SU(n), with an error of O(h^5),
    which matches the RK4 accuracy.

    The method uses a truncated matrix logarithm expansion and a projection
    onto the Lie algebra su(n), ensuring unitarity and det = 1 after
    exponentiation.

    Parameters
    ----------
    eps : torch.Tensor
        A small matrix of shape (..., n, n), anti-Hermitian to leading order.

    Returns
    -------
    torch.Tensor
        Special unitary matrix of shape (..., n, n).
    """
    # Start with eps as the first term in the matrix logarithm expansion
    dummy = eps
    exponent = dummy

    # Compute truncated series: eps - eps^2/2 + eps^3/3 - eps^4/4
    # Powers of 5 and larger are not needed as error is O(eps^5)
    for power in range(2, 5):
        dummy = dummy @ (-eps)
        exponent = exponent + dummy / power

    # Project the result to su(n): anti-Hermitian and traceless
    projected_exponent = anti_hermitian_traceless(exponent)

    # Exponentiate to get a special unitary matrix
    return torch.matrix_exp(projected_exponent)


def construct_rk4_special_unitary_type1(eps: torch.Tensor) -> torch.Tensor:
    r"""
    Constructs a special unitary matrix close to `I + eps`.

    This method is designed to address the following situation:
    In RK4 methods applied to SU(n)-valued variables, the naive update
    `I + eps` is only approximately special unitary.
    This function corrects it to lie exactly in SU(n), with an error of O(h^5),
    which matches the RK4 accuracy.

    Although we do not use this assumption fully here, we use the fact that:
    - eps is small, eps ~ h.
    - eps is anti-Hermitian to leading order: eps + eps^\dagger vanishes
      at order h.

    From this, we can recover angles of the eigenvalues (up to h^5)
    and construct a matrix in SU(n) close to the target `(I + eps)`.

    Parameters
    ----------
    eps : torch.Tensor
        A small matrix of shape (..., n, n), anti-Hermitian to leading order.

    Returns
    -------
    torch.Tensor
        Special unitary matrix of shape (..., n, n).
    """
    # Note: if U = I + eps + O(h^5), and U is unitary to O(h^5), then:
    # U^\dagger U = I + (eps + eps^\dagger) + ...
    # So (eps + eps^\dagger) is 0 up to h^2, and thus eps is
    # anti-Hermitian up to h^2.
    #
    # The exact U should have eigenvalues exp(i theta) and determinant 1.
    # So: the eigenvalues of `I + eps` are
    # `exp(i theta) ≈ I + i sin(theta) + (1 - cos(theta))
    # The anti-Hermitian part of eps gives ~ sin(theta)
    #
    # Therefore, diagonalizing the anti-Hermitian part of eps allows
    # us to approximately recover theta angles.

    # Extract anti-Hermitian part of eps, and scale it to make it Hermitian
    anti_herm = -1j * (eps - eps.adjoint())

    # Eigen-decomposition: vals \approx 2 sin(theta), vecs are eigenvectors
    vals, vecs = eigh(anti_herm)

    # Recover theta angles from eigenvalues
    theta = torch.asin(vals / 2)

    # Enforce traceless condition: det = 1 for SU(n)
    theta -= theta.mean(dim=-1, keepdim=True)

    # Compute SU(n) eigenvalues: e^{i theta}
    vals = torch.cos(theta) + 1j * torch.sin(theta)

    # Reconstruct the special unitary matrix: U = V diag(e^{i theta}) V^\dagger
    unitary_matrix = vecs @ (vals.unsqueeze(-1) * vecs.adjoint())

    return unitary_matrix


def delta_from_rk4_step(func, t, var, dt, *args):
    """Calculate the shift in var obtained from a single Runge-Kutta-4 step."""
    half_dt = dt / 2
    k_1 = func(t, var, *args)
    k_2 = func(t + half_dt, var + half_dt * k_1, *args)
    k_3 = func(t + half_dt, var + half_dt * k_2, *args)
    k_4 = func(t + dt, var + dt * k_3, *args)
    return (k_1 + 2 * k_2 + 2 * k_3 + k_4) * (dt / 6)


def anti_hermitian_traceless(x: torch.Tensor) -> torch.Tensor:
    """
    Project the input onto the space of traceless anti-Hermitian matrices.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor with square matrices in the last two dimensions.

    Returns
    -------
    torch.Tensor
        Tensor of the same shape as `x`, where each matrix is projected to be
        anti-Hermitian and traceless.
    """
    # Anti-Hermitian part
    x = (x - x.adjoint()) / 2

    # Remove trace
    trace = torch.einsum("...ii->...", x)[..., None, None]
    n = x.shape[-1]
    eye = torch.eye(n, device=x.device, dtype=x.dtype)

    return x - (trace / n) * eye


# =============================================================================
def lie_autonomous_rk3_algebra_step(algebra_func, t, var, dt, *args):
    r"""
    Performs one step of a 3-stage exponential Lie group integrator.

    This method is based on the Appendix C of [arXiv:1006.4518], and is
    designed for integrating autonomous systems of the form:

        dU / dt = F(t, U; p) U,

    where `U(t)` evolves on a Lie group (e.g., SU(n)), and `F(t, U; p)` lies in
    the corresponding Lie algebra (e.g., su(n)). The flow is assumed autonomous
    (i.e., time-invariant), so `t` serves as a dummy variable.

    Parameters
    ----------
    algebra_func : callable
        Function of the form `F(t, U, *args)` returning an element of the Lie
        algebra that defines the flow.
    t : float
        Time (a dummy argument in the autonomous case).
    var : torch.Tensor
        Current state matrix `U` in the Lie group, shape (..., n, n).
    dt : float
        Time step.
    *args : tuple
        Additional parameters to pass to `algebra_func`.

    Returns
    -------
    torch.Tensor
        Updated Lie group element.
    """
    for ind in range(3):
        func_value = algebra_func(t, var, *args)  # F(t, U; p)

        # Construct the algebra increment zee for each sub-step
        # Note: zee = eps * Z defined in (C.1) & (1.4) of [arXiv:1006.4518]
        if ind == 0:
            zee = (1 / 4 * dt) * func_value
        elif ind == 1:
            zee = (8 / 9 * dt) * func_value - (17 / 9) * zee
        else:
            zee = (3 / 4 * dt) * func_value - zee

        # Apply exponential map to update the Lie group variable
        var = torch.matrix_exp(zee) @ var

    return var


# =============================================================================
def lie_euler_algebra_step(algebra_func, t, var, dt, *args):
    """Perform a single Euler step for unitary matrices."""
    return torch.matrix_exp(algebra_func(t, var, *args) * dt) @ var


def lie_euler_step(func, t, var, dt, *args):
    """Perform a single Euler step for unitary matrices."""
    return torch.matrix_exp(func(t, var, *args) @ var.adjoint() * dt) @ var


def augmented_lie_euler_step(func, t, var, dt, *args):
    """Perform a single Euler step for unitary matrices."""
    delta, d_other = (dt * func(t, var, *args)).tuple

    # Unpack current SU(n) variable and auxiliary variable
    var, other = var.tuple

    # Update SU(n) variable with the Euler method
    var = torch.matrix_exp(delta @ var.adjoint()) @ var

    # Update auxiliary variable via standard Euler increment
    other = other + d_other

    return TupleVar(var, other)


# =============================================================================
def grad_projected_special_unitary_rk4_step(*args):
    """RK4 step for SU(n) with corrected grad; mainly for tests."""
    return project_grad_sun(special_unitary_rk4_step(*args))


def grad_projected_lie_euler_step(*args):
    """Euler step for SU(n) with corrected grad; mainly for tests."""
    return project_grad_sun(lie_euler_step(*args))


# =============================================================================
def naive_project_su3(y):
    """
    Naively projects a 3x3 complex matrix to SU(3) by orthonormalizing rows.

    This method assumes the input matrix is close to the identity. It first
    orthonormalizes the first two rows, then reconstructs the third row to
    enforce unitarity and determinant = 1.

    Notes:
    1. Although not necessary, the matrix is initially normalized to ensure
       determinnat 1.
    2. The changes are not in-place because PyTorch cannot handle
       backpropagation of derivatives (if the adjointstate method is not used).
    """
    # Normalize matrix to ensure determinant is 1 (special unitary)
    # Explicit calculation of determinant is faster than torch.linalg.det!
    y_00, y_01, y_02 = torch.unbind(y[..., 0, :], dim=-1)
    y_10, y_11, y_12 = torch.unbind(y[..., 1, :], dim=-1)
    y_20, y_21, y_22 = torch.unbind(y[..., 2, :], dim=-1)
    det = (
        y_20 * (y_01 * y_12 - y_02 * y_11)
        + y_21 * (y_02 * y_10 - y_00 * y_12)
        + y_22 * (y_00 * y_11 - y_01 * y_10)
    )

    y = y / det[..., None, None]**(1/3.)

    # Unbind rows for further calculations
    y_0, y_1, _ = torch.unbind(y, dim=-2)

    # Normalize the first row
    norm_sq = torch.sum(y_0.conj() * y_0, dim=-1, keepdim=True)
    y_0 = y_0 / torch.sqrt(norm_sq)

    # Compute inner product of first two rows
    vdot = torch.sum(y_0.conj() * y_1, dim=-1, keepdim=True)
    # Orthogonalize second row against the first
    y_1 = y_1 - y_0 * vdot

    # Normalize the second row
    norm_sq = torch.sum(y_1 * y_1.conj(), dim=-1, keepdim=True)
    y_1 = y_1 / torch.sqrt(norm_sq)

    # Reconstruct third row as complex conjugate of cross product of first two
    y_2 = torch.stack(
        ((y_0[..., 1] * y_1[..., 2] - y_0[..., 2] * y_1[..., 1]).conj(),
         (y_0[..., 2] * y_1[..., 0] - y_0[..., 0] * y_1[..., 2]).conj(),
         (y_0[..., 0] * y_1[..., 1] - y_0[..., 1] * y_1[..., 0]).conj()
        ),
        dim = -1
    )

    y = torch.stack((y_0, y_1, y_2), dim=-2)

    return y


# =============================================================================
def _benchmark_construct_rk4_special_unitary():
    """
    Benchmark for comparing different methods for constructing special unitary
    matrices for RK4 method. For SU(3) matrices in general it is cheaper to use
    `naive_project_su3` on GPU.
    """

    def run_timeit(command_str, x, n):
        time = timeit.timeit(
                command_str, number=4, globals={**globals(), **locals()}
                ) / 4 / n
        print(f"   {command_str}: \t Execution time: {time} seconds")

    for n in [2**10, 2**15, 2**20]:
        for device in ['cpu', 'cuda']:
            try:
                print(f"\n*** device is {device} ***\n")
                torch.set_default_device(device)
                x = torch.randn(n, 3, 3, dtype=torch.complex128)
            except:
                break

            h = 0.01
            eps = h * (x - x.adjoint())

            eye = torch.zeros_like(x)
            for i in range(3):
                eye[:, i, i] += 1

            u = torch.matrix_exp(eps)
            x = u - eye + h**5 * torch.randn_like(x)

            y = construct_rk4_special_unitary(x + 0)  # calls naive_project_su3
            z = construct_rk4_special_unitary_type0(x)
            w = construct_rk4_special_unitary_type1(x)
            print("Difference bw/ type0 and type1:", (z - w).abs().mean())
            print("Difference bw/ 3x3 case vs type1:", (y - w).abs().mean())

            import timeit
            m = n / (1024 * 1024)
            run_timeit("construct_rk4_special_unitary(x)", x, m)
            run_timeit("construct_rk4_special_unitary_type0(x)", x, m)
            run_timeit("construct_rk4_special_unitary_type1(x)", x, m)
