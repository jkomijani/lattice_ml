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

from .._eig import eigh, eigu

TOL = 1e-12
CHECK_ARBITRARY_PHASE = False


# =============================================================================
class NormalMatrixEig(torch.autograd.Function):
    r"""A class designed to compute the eigenvalues and eigenvectors of normal
    matrices, with support for reverse-mode algorithmic differentiation.

    Description:
    ------------
    This class is specifically intended for matrices that are *normal*, meaning
    they commute with their conjugate transpose. This property ensures that the
    matrix is unitarily diagonalizable, making it possible to compute its
    eigenvalues and eigenvectors more efficiently and accurately.

    In the context of reverse-mode algorithmic differentiation, normal matrices
    have the advantage that we can use adjoint (Hermitian transpose) operations
    rather than matrix inversions. By avoiding the need for an explicit inverse
    of the eigenvector matrix, this approach improves numerical stability and
    computational efficiency, especially for large matrices.

    For more details on algorithmic differentiation see

        "An extended collection of matrix derivative results for forward and
         reverse mode algorithmic differentiation."

    We follow the notation introduced in the above reference. In particular:

        We consider a computation which begins with a single scalar input
        variable :math:`s_i` and eventually, through a sequence of calculations,
        computes a single scalar output :math:`s_o` . Using standard automatic
        differentiation (AD) terminology, if :math:`A` is a matrix which is an
        intermediate variable within the computation, then


        1. :math:`\dot A` denotes the derivative of A with respect to s_i
        2. :math:`\bar A` denotes the derivative of s_o w.r.t each element of A


    For implementing AD, see

        https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function
    """

    @staticmethod
    def forward(ctx, matrix):
        """Computes the eigenvalues and eigenvectors of the input normal matrix
        and provides gradients using reverse-mode differentiation.
        """
        u, v = torch.linalg.eig(matrix)
        ctx.save_for_backward(u, v)
        return u, v

    @staticmethod
    def backward(ctx, grad_u, grad_v):
        # grad_{u & v} are $\bar u$ and $\bar v$ in the terminology of AD
        u, v = ctx.saved_tensors

        nabla = calc_nabla(u)
        vh_grad_v = calc_vh_grad_v(v, grad_v)

        grad_u = torch.diag_embed(grad_u)  # for matrix operations
        grad_matrix = v @ (grad_u + nabla.conj() * vh_grad_v) @ v.adjoint()

        return grad_matrix


Eign = NormalMatrixEig  # alias

Eig = NormalMatrixEig  # alias for legacy


class InverseEign(torch.autograd.Function):
    """A class to reconstruct a matrix from a given set of eigenvalues and
    eigenvectors, effectively inverting the decomposition performed by the
    `NormalMatrixEig` class (or simular classes) for normal matrices.
    """

    @staticmethod
    def forward(ctx, vals, vecs):
        matrix = vecs @ (vals.unsqueeze(-1) * vecs.adjoint())
        ctx.save_for_backward(vals, vecs)
        return matrix

    @staticmethod
    def backward(ctx, grad_matrix):
        # grad_{u & v} are $\bar u$ and $\bar v$ in the terminology of AD
        vals, vecs = ctx.saved_tensors

        grad_matrix = vecs.adjoint() @ grad_matrix @ vecs

        grad_vals = torch.linalg.diagonal(grad_matrix)
        grad_vecs = vecs @ (calc_eig_delta(vals).conj() * grad_matrix)

        return grad_vals, grad_vecs


# =============================================================================
class Eigh(torch.autograd.Function):
    """Similar to `NormalMatrixEig`, but specialized for Hermitian matrices."""

    @staticmethod
    def forward(ctx, matrix):
        u, v = eigh(matrix)
        ctx.save_for_backward(u, v)
        return u, v

    @staticmethod
    def backward(ctx, grad_u, grad_v):
        # grad_{u & v} are $\bar u$ and $\bar v$ in the terminology of AD
        u, v = ctx.saved_tensors

        nabla = calc_nabla(u)  # u and nabla are real
        vh_grad_v = calc_vh_grad_v(v, grad_v)
        vh_grad_v = (vh_grad_v - vh_grad_v.adjoint()) / 2

        grad_u = torch.diag_embed(grad_u)  # for matrix operations
        grad_matrix = v @ (grad_u + nabla * vh_grad_v) @ v.adjoint()

        return grad_matrix


# =============================================================================
class Eigu(torch.autograd.Function):
    """Similar to `NormalMatrixEig`, but specialized for unitary matrices."""

    @staticmethod
    def forward(ctx, matrix):
        u, v = eigu(matrix)
        ctx.save_for_backward(u, v)
        return u, v

    @staticmethod
    def backward(ctx, grad_u, grad_v):
        # grad_{u & v} are $\bar u$ and $\bar v$ in the terminology of AD
        u, v = ctx.saved_tensors

        nabla = calc_nabla(u)  # u and nabla are complex
        vh_grad_v = calc_vh_grad_v(v, grad_v)
        vh_grad_v = (vh_grad_v - vh_grad_v.adjoint()) / 2

        grad_u = (grad_u - u * u * grad_u.conj()) / 2
        grad_u = torch.diag_embed(grad_u)  # for matrix operations

        grad_matrix = v @ (grad_u + nabla.conj() * vh_grad_v) @ v.adjoint()

        return grad_matrix


# =============================================================================
class NaiveEigh(Eig):  # just for test
    """Similar to `Eigh`, but AD is not specialized for Hermitian matrices."""

    @staticmethod
    def forward(ctx, matrix):
        u, v = eigh(matrix)
        ctx.save_for_backward(u, v)
        return u, v


class NaiveEigu(Eig):  # just for test
    """Similar to `Eigu`, but AD is not specialized for unitary matrices."""

    @staticmethod
    def forward(ctx, matrix):
        u, v = eigu(matrix)
        ctx.save_for_backward(u, v)
        return u, v


# =============================================================================
def calc_vh_grad_v(v, grad_v):
    r"""Return :math:`v^\dagger \bar v` for AD; & check the diagonal terms."""

    vh_grad_v = v.adjoint() @ grad_v

    if CHECK_ARBITRARY_PHASE:
        # check if the imaginary part of diagonal elements of vh_grad_v is 0
        cond = torch.diagonal(vh_grad_v.imag, dim1=-2, dim2=-1).abs() > TOL
        if torch.sum(cond):
            warnings.warn("AD for eig: nonzero derivative for arbitrary phase")

    return vh_grad_v


def calc_eig_delta(u):
    """u is the list of eigenvalues"""
    n = u.shape[-1]
    delta = u.view(-1, 1, n).repeat(1, n, 1) - u.view(-1, n, 1).repeat(1, 1, n)
    return delta.view(*u.shape[:-1], n, n)


def calc_nabla(u):
    """u is the list of eigenvalues"""
    n = u.shape[-1]
    delta = u.view(-1, 1, n).repeat(1, n, 1) - u.view(-1, n, 1).repeat(1, 1, n)
    delta = delta.view(*u.shape[:-1], n, n)
    nabla = 1 / delta
    nabla[ delta == 0 ] = 0
    return nabla
