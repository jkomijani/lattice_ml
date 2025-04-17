# Copyright (c) 2025 Javad Komijani

"""A module for computing/estimating Jacobian trace."""

from typing import Callable, Optional
from types import SimpleNamespace
import torch


__all__ = ['hutchinson_estimator']


# =============================================================================
def hutchinson_estimator(
    func: Callable,
    x: torch.Tensor,
    num_samples: Optional[int | None] = 1,
    requires_grad: Optional[bool] = True,
    return_stats: Optional[bool] = False
):
    """
    Estimates the trace of the Jacobian of `func` at `x` using Hutchinson's
    method.

    This estimator is useful for high-demensional problems where explicit
    compution of Jacobian trace may be impractical.

    Args:
        func: The function whose Jacobian trace will be estimated.
        x: Input tensor at which to compute the Jacobian trace.
        num_samples: The number of random samples in the estimator.
            - Defaults to 1.
            - Higher values (e.g., 100) improve the accuracy of the estimation
              but increase computational cost.
            - If `None`, the function does not use the Hutchinson estimator and
              instead calls `calc_jacobian_trace` to compute the Jacobian trace
              exactly via automatic differentiation.
        requires_grad: Whether higher-order derivatives are needed.
            Defaults to True, indicating the output is differentiable.
        return_stats: Whether to return the mean and error as a namespace.
            Defaults to False.

    Returns:
        torch.Tensor: The estimated trace of the Jacobian of `func(x)`.

    Note:
        - The input tensor `x` is assumed to have a batch axis.
        - When the input tensort `x` does not require gradient, we create a
          view of it that requires gradient so that we use torch.autograd.grad
          to calculate the tensor of jacobian. Therfore, by default the output
          of this function is differentiable even if the input tensor `x` and
          all other parameters do not require gradient.
    """

    if num_samples is None:
        return calc_jacobian_trace(func, x, requires_grad=requires_grad)

    # The commands in this function are based on the assumption that x.ndim is
    # at least 2, where the first axis is always the batch axis. Therefore, we
    # unsqueeze `x` if it is 1D.
    if x.ndim == 1:
        x = x[:, None]

    with torch.enable_grad():
        # Ensure input requires gradients
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)

        # Compute the function output
        if not torch.is_complex(x):
            x_ = x
            y_ = func(x_)
        else:
            x_ = torch.view_as_real(x)
            y_ = torch.view_as_real(func(torch.view_as_complex(x_)))

        # Note that x_ & y_ are always real.

    # If `num_samples` is greater than 1, `retain_graph=True` is used to
    # retain the computation graph for multiple backward passes.
    grad_kwargs = {
        'retain_graph': requires_grad or (num_samples > 1),
        'create_graph': requires_grad
    }

    bsize = x.shape[0]  # batch size
    dim = tuple(range(1, x_.ndim))  # excluding batch axis

    estimates = \
        torch.zeros((num_samples, bsize), device=x_.device, dtype=x_.dtype)

    # Estimate the trace using multiple random samples
    for n in range(num_samples):
        # Generate random vector
        random_vector = torch.randn_like(x_)
        norm_sq = (random_vector**2).mean(dim=dim)  # for vector narmalization

        # Compute the Jacobian-vector product
        vjp = torch.autograd.grad(y_, x_, random_vector, **grad_kwargs)[0]

        # Compute the trace estimate for this sample (element-wise product)
        estimates[n] = (vjp * random_vector).sum(dim=dim) / norm_sq

    # Return the mean of the trace estimates across all random samples
    if num_samples == 1:
        return estimates[0]

    if not return_stats:
        return estimates.mean(dim=0)

    return SimpleNamespace(
        mean=estimates.mean(dim=0),
        error=estimates.std(dim=0) / (num_samples - 1)**0.5
    )


# =============================================================================
def calc_jacobian_trace(func, x, requires_grad=True):
    """
    Computes the exact trace of the Jacobian by iterating over the components.

    For large compenents, use `hutchinson_estimator`.

    Args:
        func (Callable): The function whose Jacobian trace is computed.
        x (torch.Tensor): Input tensor at which to compute the Jacobian trace.
        requires_grad (bool): Whether higher-order derivatives are needed.

    Returns:
        torch.Tensor: Exact trace of the Jacobian of `func(x)`.

    Note:
        - The input tensor `x` is assumed to have a batch axis.
        - When the input tensort `x` does not require gradient, we create a
          view of it that requires gradient so that we use torch.autograd.grad
          to calculate the tensor of jacobian. Therfore, by default the output
          of this function is differentiable even if the input tensor `x` and
          all other parameters do not require gradient.
    """

    shape = x.shape
    bsize = x.shape[0]  # batch size

    # Track gradients for the input `x`
    with torch.enable_grad():
        # Ensure input requires gradients
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)

        # To easily iterate over components of `x` in torch.autograd.grad,
        # we reshape to `(bsize, -1)`

        # Compute the function output
        if not torch.is_complex(x):
            x_ = x.reshape(bsize, -1)
            y_ = func(x_.reshape(*shape)).reshape(bsize, -1)
        else:
            x_ = torch.view_as_real(x).reshape(bsize, -1)
            y = func(torch.view_as_complex(x_.reshape(*shape, 2)))
            y_ = torch.view_as_real(y).reshape(bsize, -1)

        # Note that x_ & y_ are always real.

        y_ = y_.sum(dim=0).unbind()

    grad_kwargs = {'retain_graph': True, 'create_graph': requires_grad}
    jacobian_trace = 0

    # Sum of gradients for each component of the output to compute the trace
    for ind, y_ind in enumerate(y_):  # Iterate over each component of y_
        grad_ind = torch.autograd.grad(y_ind, x_, **grad_kwargs)[0]
        jacobian_trace += grad_ind[:, ind]

    return jacobian_trace


# =============================================================================
def _test_hutchinson(num_samples=100, complex_=True):
    """
    Tests the Hutchinson method for estimating the trace of a Jacobian.

    This function compares the Hutchinson estimation against the exact Jacobian
    trace for a given nonlinear function and verifies gradient computation.

    Steps performed:
    1. Defines `func(x) = sin(x * params)`.
    2. Computes the trace of the Jacobian using the Hutchinson method.
    3. Computes the exact trace of the Jacobian.
    4. Compares the estimated trace against the exact trace.
    5. Computes gradient of the Jacobian trace w.r.t. parameters.
    """
    if complex_:
        params = torch.tensor([0.55 + 0.82j, 0.73, 0.14], requires_grad=True)
    else:
        params = torch.tensor([0.55, 0.73, 0.14], requires_grad=True)

    def func(x):
        p = params[None, :]  # x has batch axis, but p does not have
        return torch.sin(p * x)

    def func_jacobian_trace(x):
        """Returns the analytic value for the Jacobian trace of `func(x)`."""
        p = params[None, :]  # x has batch axis, but p does not have
        jacobian_diag = torch.cos(p * x) * p
        if not torch.is_complex(x):
            res = jacobian_diag.sum(dim=1)
        else:
            res = 2 * torch.real(jacobian_diag).sum(dim=1)
        return res

    # x = torch.randn(4, 3)
    x = torch.tensor(
        [[-1.1720, -0.3929,  0.5265],
         [1.1065,   0.9273, -1.7421],
         [-0.7699,  0.7864, -1.9963],
         [0.5836,   1.0392,  0.8023]]
    )

    if complex_:
        x = x + 1j

    result = hutchinson_estimator(func, x, num_samples, return_stats=True)
    print("\nHutchinson estimate:")
    print(result)
    autodiff_result = calc_jacobian_trace(func, x)
    analytic_result = func_jacobian_trace(x)

    print("\nAutodiff value:")
    print(autodiff_result)

    print("\nAnalytic value:")
    print(analytic_result)

    print("\nDifference of Hutchinson estimate and exact (in units of sigma):")
    print((result.mean - autodiff_result) / result.error)

    # Gradients of the Jacobian trace w.r.t `params`
    result.mean.mean().backward()
    params_grad_1 = params.grad

    # Reset gradients and compute exact gradients
    params.grad = None
    autodiff_result.mean().backward()
    params_grad_2 = params.grad

    print("\nGradient of Jacobian trace wrt `params`; ratio: Hutchinson/exact")
    print((params_grad_1 / params_grad_2).ravel())  # Print exact gradients


if __name__ == '__main__':
    _test_hutchinson()
