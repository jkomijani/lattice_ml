# Copyright (c) 2024 Javad Komijani

import torch

from abc import abstractmethod, ABC

from torch_linalg_ext import eigh, eigu, inverse_eign, reciprocal


# =============================================================================
class MatrixFunctionTemplate(ABC):
    r"""A template class for handling a matrix transformation as :math:`f(M)`,
    where the matrix :math:`M` is supposed to be normal.

    Exploiting the spectral decomposition of :math:`H`, we have

    .. math::

        F = \Omega f(\Lambda) \Omega^\dagger

    and

    .. math::

       dF = \Omega (df(\Lambda) + [d\Gamma, \Lambda]) \Omega^\dagger

    where the square bracket denotes the commutator and
    :math:`d\Gamma = \Omega^\dagger d\Omega`, which is anti-Herimtian matrix.

    Any subclass should have a method called `scalar_func` for transforming the
    eigenvalues, which also returns the derivative of the function.
    """

    forward_mode_eig = torch.linalg.eig

    def __call__(self, matrix):
        return self.forward(matrix)

    def forward(self, matrix):
        vals, vecs = self.forward_mode_eig(matrix)
        f_vals, f_prime = self.scalar_func(vals)
        matrix = inverse_eign(f_vals, vecs)
        return matrix, self.calc_jacobian_matrix(vals, vecs, f_vals, f_prime)

    def calc_jacobian_matrix(self, eigvals, eigvecs, f_eigvals, f_prime):
        nabla = reciprocal(calc_eig_delta(eigvals))
        delta_f = calc_eig_delta(f_eigvals)
        mat = torch.diag_embed(f_prime) + delta_f * nabla
        jac1 = torch.diag_embed(mat.reshape(*nabla.shape[:-2], -1))
        jac2 = kronecker_product(eigvecs, eigvecs.conj())
        return jac2 @ jac1 @ jac2.adjoint()

    @abstractmethod
    def scalar_func(self, args):
        pass


class MatrixExp1jh(MatrixFunctionTemplate):
    """Return :math:`U = \exp(i H)` with :math:`H` being a Hermitian matrix."""

    forward_mode_eig = eigh

    def scalar_func(self, h):
        u = torch.exp(1j * h)
        return u, 1j * u


class MatrixAngleU(MatrixFunctionTemplate):
    """Return :math:`H = -i \log(U)` with :math:`U` being a unitary matrix."""

    forward_mode_eig = eigu

    def scalar_func(self, u):
        h = torch.angle(u)
        return h, -1j * u.conj()


# =============================================================================
def inverse_eign_and_jacobian(eigvals, eigvecs, mode='Gamma'):
    """
    For calculating :math:`\Omega \Lambda \Omega^\dagger`, where the modal
    matrix is untiary and the spectral matrix is diagonal.

    It also returns the Jacobian matrix with respect to inputs with three
    different options specified with `mode`.
    Using :math:`\d Gamma = \Omega^\dagger d\Omega`, which is an anti-hermtian
    matrix with vanishing diagonal terms (the redundancy is fixed),
    the three options are:

    1. `mode = 'Gamma'`: Jacobian w.r.t. math:`\int d\Gamma` (default option).

    2. `mode = 'Full'`: Jacobian wrt the set :math:`{\Lambda, \int d\Gamma}`.

    3. `mode = 'Omega'`: Jacobian w.r.t. math:`\Omega`.

    Note that the Jacobian is singular with 'Gamma' and 'Omega' modes because
    in formet one the eigenvalues do not vary and the latter one the redundnacy
    is not fixed.
    """
    matrix = inverse_eign(eigvals, eigvecs)
    delta = calc_eig_delta(eigvals)
    eye = eyes_like(delta)
    jac2 = kronecker_product(eigvecs, eigvecs.conj())

    shape = [*delta.shape[:-2], -1]
    match mode:
        case 'Gamma':
            jac1 = torch.diag_embed(delta.reshape(*shape))  # det(jac1) = 0
        case 'Full':
            jac1 = torch.diag_embed((eye + delta).reshape(*shape))
        case 'Omega':
            jac1 = torch.diag_embed(delta.reshape(*shape))  # det(jac1) = 0
            jac1 = jac1 @ kronecker_product(eigvecs.adjoint(), eye)

    return matrix, jac2 @ jac1


def commutator_and_jacobian(mat1, mat2):
    """Return the commutator of two square matrices, :math:`[P, Q]`, and the
    Jacobian matrix with respect to the first input.
    """
    mat = mat1 @ mat2 - mat2 @ mat1
    eye = eyes_like(mat1)
    mat2_t = mat2.transpose(-2, -1)
    jac = kronecker_product(eye, mat2_t) - kronecker_product(mat2, eye)
    return mat, jac


# =============================================================================
def kronecker_product(mat1, mat2):
    """Return the Kronecker product of two input matrices."""
    shp1 = mat1.shape
    shp2 = mat2.shape
    assert shp1[:-2] == shp2[:-2], f"{shp1[:-2]} != {shp2[:-2]}"
    mat1 = mat1.repeat_interleave(shp1[-2], -2).repeat_interleave(shp1[-1], -1)
    mat2 = mat2.repeat(*[1]*(len(shp1) - 2) + list(shp1[-2:]))
    return mat1 * mat2


def eyes_like(matrix):
    """Return identity matrices of the same size of the input matrix."""
    eye = torch.zeros_like(matrix)
    for k in range(matrix.shape[-1]):
        eye[..., k, k] = 1
    return eye


def calc_eig_delta(u):
    """u is the list of eigenvalues"""
    n = u.shape[-1]
    delta = u.view(-1, 1, n).repeat(1, n, 1) - u.view(-1, n, 1).repeat(1, 1, n)
    return delta.view(*u.shape[:-1], n, n)
