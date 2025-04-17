# Copyright (c) 2023 Javad Komijani

r"""
In this module we implement our own custom autograd Functions by subclassing
torch.autograd.Function and implementing the forward and backward passes
which operate on Tensors.

For example, one can see "An extended collection of matrix derivative results
for forward and reverse mode algorithmic differentiation."
"""

import torch
import warnings

from .._svd import svd, AttributeDict4SVD

TOL = 1e-8
CHECK_ARBITRARY_PHASE = False


# =============================================================================
class SVD(torch.autograd.Function):
    r"""
    For more details see
    "An extended collection of matrix derivative results for forward and
    reverse mode algorithmic differentiation."
    We follow the notation introduced in the above reference as
    "We consider a computation which begins with a single scalar input variable
    :math:`s_i` and eventually, through a sequence of calculations, computes a
    single scalar output :math:`s_o` . Using standard automatic differentiation
    (AD) terminology, if :math:`A` is a matrix which is an intermediate
    variable within the computation, then
    1. :math:`\dot A` denotes the derivative of A with respect to s_i
    2. :math:`\bar A` denotes the derivative of s_o w.r.t each element of A.

    https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function
    """

    @classmethod
    def apply_wrapper(clc, matrix):
        u, s, vh = clc.apply(matrix)
        return AttributeDict4SVD(U=u, S=s, Vh=vh)

    @staticmethod
    def forward(ctx, matrix):
        svd_ = svd(matrix)
        ctx.save_for_backward(svd_.U, svd_.S, svd_.Vh)
        return svd_.U, svd_.S, svd_.Vh

    @staticmethod
    def backward(ctx, grad_u, grad_s, grad_vh):
        u, s, vh = ctx.saved_tensors

        nabla_s = calc_nabla(s)
        nabla_plus = calc_nabla_plus(s)

        uh_grad_u = u.adjoint() @ grad_u
        vh_grad_v = vh @ grad_vh.adjoint()

        if CHECK_ARBITRARY_PHASE:
            check_for_arbitrary_phase(uh_grad_u, vh_grad_v)

        uh_grad_u = (uh_grad_u - uh_grad_u.adjoint()) / 2
        vh_grad_v = (vh_grad_v - vh_grad_v.adjoint()) / 2

        grad_s = torch.diag_embed(grad_s)  # for matrix operations
        grad_matrix = u @ (
                grad_s
                + nabla_s * (uh_grad_u + vh_grad_v)
                + nabla_plus * (uh_grad_u - vh_grad_v)
                ) @ vh

        return grad_matrix


# =============================================================================
class ADSimplifiedSVD(SVD):
    r"""Similar to SVD except that its AD is simpler because it is assumed that
    the dependence on U and Vh is only through combination of U @ Vh, which
    is unique for non-vanishing singular values.
    """

    @staticmethod
    def backward(ctx, grad_u, grad_s, grad_vh):
        u, s, vh = ctx.saved_tensors

        nabla_plus = calc_nabla_plus(s)

        uh_grad_u = u.adjoint() @ grad_u

        uh_grad_u_times2 = (uh_grad_u - uh_grad_u.adjoint())

        grad_s = torch.diag_embed(grad_s)  # for matrix operations
        grad_matrix = u @ (grad_s + nabla_plus * uh_grad_u_times2) @ vh

        return grad_matrix


# =============================================================================
def check_for_arbitrary_phase(uh_grad_u, vh_grad_v):
    """Check if the imaginary part of diagonal elements of the sum of inputs
    are 0.
    """
    x = torch.diagonal(uh_grad_u.imag + vh_grad_v.imag, dim1=-2, dim2=-1).abs()
    cond = x > TOL
    if torch.sum(cond):
        warnings.warn("SVD: AD for arbitrary phase does not vanish")
        # str_ = f"measure: {torch.sum(cond)} x {torch.mean(x[cond]):g}"
        # warnings.warn(f"SVD: AD for arbitrary phase does not vanish; {str_}")


def calc_nabla(s):
    """s is the list of singular values"""
    n = s.shape[-1]
    delta = s.view(-1, 1, n).repeat(1, n, 1) - s.view(-1, n, 1).repeat(1, 1, n)
    delta = delta.view(*s.shape[:-1], n, n)
    nabla = 1 / delta
    nabla[ delta == 0 ] = 0
    return nabla


def calc_nabla_plus(s):
    """s is the list of singular values"""
    n = s.shape[-1]
    zeta = s.view(-1, 1, n).repeat(1, n, 1) + s.view(-1, n, 1).repeat(1, 1, n)
    zeta = zeta.view(*s.shape[:-1], n, n)
    nabla_plus = 1 / zeta
    nabla_plus[ zeta == 0 ] = 0
    return nabla_plus
