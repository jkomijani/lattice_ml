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
from torch.nn.functional import normalize


__all__ = ["project_grad_sun", "project_data_and_grad_sun"]


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


class ProjectDataAndGrad2SUn(ProjectGrad2SUn):
    """
    Same as ProjectGrad2SUn, but additionally corrects small numerical
    deviations from SU(n) in the forward pass due to numerical errors.

    Forward:
        Projects u back onto SU(n).

    Backward:
        Same Lie-algebra projection as ProjectGrad2SUn:
            g = anti_hermitian_traceless(grad_u @ u†)
            grad = g @ u
    """

    @staticmethod
    def forward(ctx, u: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: identity mapping.

        Parameters:
        - ctx: context object for saving variables for backward pass
        - u: input matrix (should be special unitary up to numerical errors)

        Returns:
        - The same input u corrected for numerical errors
        """
        # Save u for use in the backward pass
        u = naive_project_onto_su3(u)  # correct for small deviation from SU(n)
        ctx.save_for_backward(u)
        return u


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


def naive_project_onto_su3(y: torch.Tensor) -> torch.Tensor:
    """
    Naively projects a 3x3 complex matrix to SU(3) by orthonormalizing rows.

    This function compute the SU(3) matrix Q that approximately maximizes
    `ReTr(Q† M)` for the input matrix M. This method works well if M is close
    to a unitary matrix. It first orthonormalizes the first two rows, then
    reconstructs the third row to enforce unitarity and determinant = 1.
    """
    # Normalize matrix to ensure determinant is 1 (special unitary)
    y = y / torch.linalg.det(y)[..., None, None] ** (1/3.)

    # Unbind rows for further calculations
    y_0, y_1, _ = torch.unbind(y, dim=-2)

    # Normalize the first row
    y_0 = normalize(y_0, dim=-1)

    # Orthonormalize second row against the first
    vdot = torch.sum(y_0.conj() * y_1, dim=-1, keepdim=True)
    y_1 = normalize(y_1 - y_0 * vdot, dim=-1)

    # Reconstruct third row as complex conjugate of cross product of first two
    y_2 = torch.linalg.cross(y_0, y_1).conj()

    y = torch.stack((y_0, y_1, y_2), dim=-2)

    return y


project_grad_sun = ProjectGrad2SUn.apply
project_data_and_grad_sun = ProjectDataAndGrad2SUn.apply
