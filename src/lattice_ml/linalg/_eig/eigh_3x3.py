# Copyright (c) 2023 Javad Komijani

import torch
import numpy as np

from .eig_3x3 import eign3x3


# =============================================================================
def eigh3x3(matrix, **kwargs):
    u, v = eign3x3(matrix, func_4_eigvals = eigvalsh3x3, **kwargs)
    return u.real, v


# =============================================================================
def eigvalsh3x3(matrix, return_invariants=False):
    r"""
    Return eigenvalues of 3x3 hermitian matrices using closed form expressions.

    Parameters
    ----------
    matrix : tensor
        can be a single matrix or a batch of matrices.
        We assume `matrix` is hermition (over last two dimensions),
        otherwise we treat it similar to pytorch by discarding the
        upper diagonal terms.

    return_invariants : boolean
        in addition to the eigenvalues, return parameters mu, theta and phi;
        see the discription of algorithm below. (Default is False.)
    
    We now give the algorithm for eigenvalue determination of 3x3 Hermitian
    matrices.

    Without loss of generality we write matrix :math:`A` as

    .. math::

        A = \mu I + \theta M

    where :math:`\text{Tr} M = 0` and :math:`\text{Tr} M^2 = 2`,
    indicating :math:`\theta^2 = \frac{1}{2} \text{Tr} (A - \mu I)^2`.
    Exploiting Cayley-Hamilton theorem for 3x3 matrices, matrix :math:`M`
    satisfies

    .. math::

        M^3 = M + \text{det}(M) I

    and therefore :math:`\frac{1}{3} \text{Tr} M^3 = \text{det} M`.
    Likewise, the eigenvalues of :math:`M` satisfy:

    .. math::

        \lambda^3 = \lambda + \text{det}(M).

    The three solutions to this equation can be expressed in terms of
    trigonometric functions with real arguments for Hermitian matrices.
    To this end we first perform a change of variable as

    .. math::

         \text{det}(M) = -\frac{2}{\sqrt{27}} \sin 3 \phi

    Then, exploiting the triple-angle trigonometric identity

    .. math::

         \sin 3\phi = 3\sin \phi - 4 \sin^3\phi

    the three eigenvalues of matrix :math:`M` read

    .. math::

        \lambda_k = \frac{2}{\sqrt{3}} \sin \left(\phi + 2\pi k /3\right).

    Note that the sum of the eigenvalues is zero.
    """

    assert matrix.shape[-2:] == (3, 3), "matrix is supposed to be 3x3"

    # We use following parametrization of the hermitian matrices
    # H = [[a,       x - I y,  s - I t]
    #      [x + I y,    b,     p - I q]
    #      [s + I t, p + I q,    c    ]]
    
    a = torch.real(matrix[..., 0, 0])
    b = torch.real(matrix[..., 1, 1])
    c = torch.real(matrix[..., 2, 2])
    x = torch.real(matrix[..., 1, 0])
    y = torch.imag(matrix[..., 1, 0])
    s = torch.real(matrix[..., 2, 0])
    t = torch.imag(matrix[..., 2, 0])
    p = torch.real(matrix[..., 2, 1])
    q = torch.imag(matrix[..., 2, 1])

    # We now define H = \mu I + \theta M, where M is a traceless
    # Hermitian matrix, in which c = - a - b, thus below we won't have c.
    # Moreover, we choose \theta such that Tr M^2 = 2;
    # then \theta^2 = 1/2 Tr(H - \mu I)^2

    mu = (a + b + c) * (1/3.)
    a = a - mu
    b = b - mu
    # c = c - mu  # no need to calculate c, which is set to -(a + b)
    
    theta = torch.sqrt(a*a + b*b + a*b + x*x + y*y + s*s + t*t + p*p + q*q)

    # minus_det is minus determinant of (H - \mu I)
    minus_det = a * (p*p + q*q) + b * (s*s + t*t) \
       + (a + b) * (a*b - x*x - y*y) - 2*(p*s*x + q*t*x - q*s*y + p*t*y)

    argument = minus_det / theta**3 * (27**0.5 / 2)
    phi = torch.asin(argument) / 3  # phi \in [-pi/6, pi/6]

    # Let us now fix phi for case theta = 0 or for numerical error the
    # absolute value of the argument inside arcsin is larger than 1.
    # case 1: theta = 0: phi is irrelevant and we set it to zero
    phi[theta == 0] = 0
    phi[argument > 1] = np.pi/6
    phi[argument < -1] = -np.pi/6

    eigvals = mu.unsqueeze(-1) \
            + (2/3**0.5) * theta.unsqueeze(-1) \
              * torch.sin(
                   torch.stack([phi - 2*np.pi/3, phi, phi + 2*np.pi/3], dim=-1)
                )  # does not have contribution from mu

    if return_invariants:
        return eigvals, (mu, theta, phi)
    else:
        return eigvals
