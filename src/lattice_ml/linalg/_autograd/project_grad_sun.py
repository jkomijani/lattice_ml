# Copyright (c) 2025 Javad Komijani

"""
This module defines a custom PyTorch autograd Function that enables
differentiable optimization over the special unitary group SU(n).

The class ProjectGrad2SUn implements a gradient projection mechanism that
respects the geometry of SU(n) by projecting gradients onto the tangent space
at a given point on the manifold. This is particularly useful for tasks where
the model parameters are constrained to be unitary matrices with determinant 1,
such as in quantum computing, signal processing, or manifold optimization.
"""

import torch

__all__ = ["project_grad_sun"]


class ProjectGrad2SUn(torch.autograd.Function):
    """
    Custom autograd function to project gradients onto the Lie algebra su(n).

    This function allows for gradient-based optimization over the special
    unitary group SU(n), where gradients must respect the group's manifold
    structure.

    In the forward pass, the function simply returns the input unchanged.

    In the backward pass, it projects the incoming gradient grad_u onto the
    tangent space of SU(n) at the point u, and then returns it in group space.
    This ensures compatibility with PyTorch's gradient propagation system
    while maintaining manifold consistency.
    """

    @staticmethod
    def forward(ctx, u: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: identity mapping.

        Parameters:
        - ctx: context object for saving variables for backward pass
        - u: input matrix (should be special unitary with det(u) = 1)

        Returns:
        - The same input u, unchanged
        """
        # Save u for use in the backward pass
        ctx.save_for_backward(u)
        return u

    @staticmethod
    def backward(ctx, grad_u: torch.Tensor) -> torch.Tensor:
        # grad_u is $\bar u$ in the terminology of AD
        """
        Backward pass: projects grad_u onto the tangent space of SU(n) at u,
        and then returns it in group coordinates, suitable for use in PyTorch's
        backpropagation.

        Parameters:
        - ctx: context object containing saved tensors
        - grad_u: gradient of the loss with respect to the output u

        Returns:
        - A projected gradient.
        """
        # Retrieve saved forward input
        (u,) = ctx.saved_tensors

        # Compute unprojected gradient g = grad_u @ u†, and project it onto ...
        g = anti_hermitian_traceless(grad_u @ u.adjoint())

        # Calculate and return it in group coordinates
        return g @ u


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


project_grad_sun = ProjectGrad2SUn.apply
