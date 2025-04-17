# Copyright (c) 2024 Javad Komijani

import torch

from abc import abstractmethod, ABC

from torch_linalg_ext import eigh, eigu, inverse_eign


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
    eigenvalues.
    """

    forward_mode_eig = torch.linalg.eig

    def __call__(self, matrix):
        return self.forward(matrix)

    def forward(self, matrix):
        vals, vecs = self.forward_mode_eig(matrix)
        f_vals = self.scalar_func(vals)
        matrix = inverse_eign(f_vals, vecs)
        return matrix

    @abstractmethod
    def scalar_func(self, args):
        pass


class MatrixExp1jh(MatrixFunctionTemplate):
    """Return :math:`U = \exp(i H)` with :math:`H` being a Hermitian matrix."""

    forward_mode_eig = eigh

    def scalar_func(self, h):
        u = torch.exp(1j * h)
        return u


class MatrixAngleU(MatrixFunctionTemplate):
    """Return :math:`H = -i \log(U)` with :math:`U` being a unitary matrix."""

    forward_mode_eig = eigu

    def scalar_func(self, u):
        h = torch.angle(u)
        return h


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
