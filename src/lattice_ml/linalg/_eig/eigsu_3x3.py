# Copyright (c) 2023 Javad Komijani

import torch

from .eig_3x3 import eign3x3
from .eigh_3x3 import eigh3x3


# =============================================================================
def eigsu3x3(matrix, **kwargs):
    """
    Althought this function is special for SU(3) matrices, it is better to
    avoid it because if the matrix is slightly different from a SU(3) matrix,
    the eigenvalues become slightly wrong too, which then affects the precision
    of eigenvectors and the fact that they must be perpendicular for
    non-degenerate eigenvalues. As a result, this implementation can accumulate
    errors unlike using `eign3x3`.
    """
    return eign3x3(matrix, func_4_eigvals = eigvalssu3x3, **kwargs)


# =============================================================================
def eigvalssu3x3(matrix, return_invariants=False):
    r"""
    Return eigenvalues of 3x3 special unitary matrices using closed form
    expressions.

    Parameters
    ----------
    matrix : tensor
        can be a single matrix or a batch of matrices.
        We assume `matrix` is special unitary (over last two dimensions),
        for which the eigenvalues depend only on the trace of the matrix.
        (We do not check if the matrix is truely special unitary.)

    For special unitary matrices, we can exploit the fact that the deteriminant
    is unity and the absolute value of eigenvalues are 1 to simplify the
    characteristic equation to

    .. math::

        \lambda^3 - 3 \mu \lambda^2 + 3 \mu^* \lambda - 1 = 0,

    where :math:`\mu` is one third of the trace of the matrix. A change of
    variable as :math:`\lambda = x + \mu` turns the above equation to

    .. math::

        x^3 - 3 p x = 2 q

    where 

    .. math::

        p = \mu^2 - \mu^*
        q = \mu^3 + (1 - 3 |\mu|^2) / 2

    The damped cubic equation can be solved exactly. For instance, one can
    follow Appendix A in [http://arxiv.org/abs/physics/0610206],
    "Efficient numerical diagonalization of hermitian 3x3 matrices",
    to orgainze the solutions.

    Note that when :math:`|\mu| \approx 1`, i.e., when the SU(3) matrix is
    close to its center, we have :math:`|r_2| \approx |r_3|`, but with
    different phases. As a result the cancelation between these two terms is
    vulnerable to round-off error.

    Let us formulate this problem in a slightly different way: we could write
    the special unitary matrix as

    .. math::

        U = \mu I + \theta M

    where :math:`\text{Tr}\, M = 0` and :math:`\text{Tr}\, M M^\dagger = 2`,
    indicating

    .. math::

         \theta^2 = \frac{1}{2} \text{Tr}\ (A - \mu I) (A - \mu I)^\dagger 
         \theta^2 = \frac{3}{2} (1 - |\mu|^2)

    we can then use :math:`theta` as a measure that implies how close the SU(3)
    matrix is to its center, where :math:`|\mu| = 1`.
    """
    assert matrix.shape[-2:] == (3, 3), "matrix is supposed to be 3x3"

    mu = torch.mean(matrix.diagonal(dim1=-1, dim2=-2), dim=-1)

    p = mu*mu - mu.conj()
    q = p * mu + 0.5 * (1 - mu.real*mu.real - mu.imag*mu.imag)

    r_1 = mu
    r_2 = (q + torch.sqrt((q*q - p*p*p).real))**(1/3.)
    r_3 = p / r_2

    # Note that (q*q - p*p*p) is always real and typically positive

    cond = (torch.abs(p) < 1e-16)
    r_2[cond] = (2 * q[cond])**(1/3)
    r_3[cond] = 0
    cond = cond & (torch.abs(q) < 1e-16)
    r_2[cond] = 0

    w1 = (-1 + 1j * 3**0.5) / 2  # w = exp(i * 2 pi /3)
    w2 = (-1 - 1j * 3**0.5) / 2  # w = exp(i * 4 pi /3)

    eigvals = torch.stack(
               [r_1 + r_2 + r_3,
                r_1 + r_2 * w1 + r_3 * w2,
                r_1 + r_2 * w2 + r_3 * w1
               ],
               dim=-1
               )

    _, sorted_ind = torch.sort(eigvals.real, dim=-1)
    eigvals = eigvals.gather(-1, sorted_ind)

    if return_invariants:
        return eigvals, (mu,)
    else:
        return eigvals


# =============================================================================
def eigu3_from_h(x):
    r"""Return eigenvalues and eigenvectors of unitary matrices via converting
    them to Hermitian matrices.

    We use

    .. math::

        U \Omega = \Omega \Lambda
        U^\dagger \Omega = \Omega \Lambda^\dagger

    to write

    .. math::

       (U + U^\dagger) \Omega = \Omega (\Lambda + \Lambda^\dagger)
       (U - U^\dagger) \Omega = \Omega (\Lambda - \Lambda^\dagger)

    to obtain eigenvalues and eigencetors of unitary matrices.

    Warning: The algorithm used here can lead to wrong decomposition if there
    is a degeneracy in :math:`\sin(\theta_i)` while corresponding
    :math:`\cos(\theta_i) are not degenerate; e.g., this happend when
    :math:`\theta_0 = \pi - \theta_1`.
    However, it is unlikely to happend with random matrices.
    Moreover, if this function is called only for SU(3) matrices that are close
    to their centers, this situation will not happen because if
    :math:`\sin(\theta_i)` are degenerate, corresponding :math:`\cos(\theta_i)`
    will be degenerate too. More specifically, as long as :math:`|\mu| >
    \sqrt{5} / 3`, where :math:`\mu` is one third of the trace,
    the trouble-making situation does not happen; at the boundary
    we have :math:`(\theta_0, \theta_1, \theta_2) = (\pi/2, \pi/2, \pi)`.
    """
    eig_2sin, modal_matrix = eigh3x3(1J * (x.adjoint() - x))
    eig_2cos = torch.diagonal(
            modal_matrix.adjoint() @ (x.adjoint() + x) @ modal_matrix,
            dim1=-1, dim2=-2
            )
    eig = (eig_2cos + eig_2sin * 1J) / 2
    return eig, modal_matrix
