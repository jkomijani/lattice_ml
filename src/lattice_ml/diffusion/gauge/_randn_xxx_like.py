# Created by Javad Komijani, 2024-2025

"""
Provides functions to generate unitary, special unitary, anti-Hermitian,
and traceless anti-Hermitian matrices using Gaussian distributions.
"""

from typing import Tuple

import torch


__all__ = [
    'randn_unitary_like',
    'randn_special_unitary_like',
    'randn_antihermitian_like',
    'randn_traceless_antihermitian_like'
]


# =============================================================================
def randn_unitary_like(
    x: torch.Tensor, scale: float | torch.Tensor = 1, n_steps: int = 4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate random U(n) matrices and corresponding u(n) matrices with
    the same shape as the input.

    Constructs U(n) matrices by exponentiating random anti-Hermitian matrices.
    The result is built as a product of `n_steps` small exponentials,
    simulating a Gaussian random walk on U(n). Smaller values for `n_steps`
    give a rougher approximation; larger values yield smoother samples.

    Args:
        x (torch.Tensor): Input of shape (..., n, n) defining output's shape.
        scale (float or torch.Tensor, optional): A scaling factor applied to
           the Lie-algebra elements. If a tensor, it must be broadcastable to
           the input. Default is 1.
        n_steps (int, optional): Number of random-walk steps. Default is 4.

    Returns:
        tuple:
            - randn_grp (torch.Tensor): Generated random U(n) matrix.
            - randn_alg (torch.Tensor): Accumulated Lie-algebra element.
    """
    # Normalize standard deviation by sqrt(n_steps)
    scale = scale / n_steps ** 0.5

    # Initial scaled normal anti-Hermitian matrix
    randn_alg = scale * randn_antihermitian_like(x)

    # Exponentiate first step
    randn_grp = torch.matrix_exp(randn_alg)

    # Iteratively perform random walk if n_steps > 1
    for _ in range(n_steps - 1):
        step = scale * randn_antihermitian_like(x)
        randn_alg += step  # accumulate the total Lie algebra element
        randn_grp = torch.matrix_exp(step) @ randn_grp

    return randn_grp, randn_alg


def randn_special_unitary_like(
    x: torch.Tensor, scale: float | torch.Tensor = 1., n_steps: int = 4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate random SU(n) matrices and corresponding su(n) matrices with
    the same shape as the input.

    Constructs SU(n) matrices by exponentiating random traceless anti-Hermitian
    matrices. The result is built as a product of `n_steps` small exponentials,
    simulating a Gaussian random walk on SU(n). Smaller values for `n_steps`
    give a rougher approximation; larger values yield smoother samples.

    Args:
        x (torch.Tensor): Input of shape (..., n, n) defining output's shape.
        scale (float or torch.Tensor, optional): A scaling factor applied to
           the Lie-algebra elements. If a tensor, it must be broadcastable to
           the input. Default is 1.
        n_steps (int, optional): Number of random-walk steps. Default is 4.

    Returns:
        tuple:
            - randn_grp (torch.Tensor): Generated random SU(n) matrix.
            - randn_alg (torch.Tensor): Accumulated Lie-algebra element.
    """
    # Normalize standard deviation by sqrt(n_steps)
    scale = scale / n_steps ** 0.5

    # Initial scaled normla traceless anti-Hermitian matrix
    randn_alg = scale * randn_traceless_antihermitian_like(x)

    # Exponentiate first step
    randn_grp = torch.matrix_exp(randn_alg)

    # Iteratively perform random walk if n_steps > 1
    for _ in range(n_steps - 1):
        step = scale * randn_traceless_antihermitian_like(x)
        randn_alg += step  # accumulate the total Lie algebra element
        randn_grp = torch.matrix_exp(step) @ randn_grp

    return randn_grp, randn_alg


def randn_antihermitian_like(x: torch.Tensor) -> torch.Tensor:
    r"""
    Generates random anti-Hermitian matrices with the same shape as the input.

    Both the real and imaginary parts of the non-diagonal elements are drawn
    from Gaussian distributions with zero mean and variance of 1/2. The real
    part of the diagonal elements is zero, while the imaginary parts are drawn
    from a Gaussian distribution with zero mean and unit variance.

    This normalization is consistent with generating an anti-Hermitian matrix
    using the generators of unitary matrices, where the coefficients are
    normally distributed with zero mean and unit variance, and the generators
    are normalized as :math:`Tr (T_a T_b) = - \delta_{a,b}`.

    By making the resulting matrix traceless, the variance of the imaginary
    parts of the diagonal elements is reduced, which is in agreement with the
    generation of a traceless anti-Hermitian matrix using the generators of
    special unitary matrices, where the coefficients are normally distributed
    with zero mean and unit variance.

    Args:
        x (torch.Tensor): Input of shape (..., n, n) defining output's shape.

    Returns:
        torch.Tensor: Anti-Hermitian tensors of the same shape as x.

    Note:
        The function is intended for complex input, so the description above
        assumes a complex x. However, it can also be used with real x.
        In that case, the function generates real anti-symmetric matrices.
        The independent entries are drawn from a Gaussian distribution with
        zero mean and unit variance, which results in a different normalization
        for the real generators.
    """
    assert x.shape[-1] == x.shape[-2], "Not a square matrix!"

    # Note: For complex input, torch.randn_like returns entries distributed as
    # standard complex normal CN(0,1), i.e. real and imaginary parts are i.i.d.
    # N(0, 1/2), so that E[|z|^2] = 1.
    noise = torch.randn_like(x)

    return (noise - noise.adjoint()) / 2 ** 0.5


def randn_traceless_antihermitian_like(x: torch.Tensor) -> torch.Tensor:
    """
    Generates random, traceless, anti-Hermitian matrices with the same shape
    as the input.

    For details see `randn_antihermitian_like` and `make_traceless`.

    Args:
        x (torch.Tensor): Input of shape (..., n, n) defining output's shape.

    Returns:
        torch.Tensor: Traceless anti-Hermitian tensors of the same shape as x.
    """
    return make_traceless(randn_antihermitian_like(x))


def make_traceless(x: torch.Tensor) -> torch.Tensor:
    """
    Given a tensor of shape (..., n, n), makes the last two axes traceless
    by subtracting the mean of the diagonal elements from the diagonal entries.

    Args:
        x (torch.Tensor): Input tensor of shape (..., n, n).

    Returns:
        torch.Tensor: A traceless tensor with the same shape as x.
    """
    assert x.shape[-1] == x.shape[-2], "Not a square matrix!"

    # Compute the mean of diagonal elements -> reduced trace
    reduced_trace = x.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True)
    return x - torch.diag_embed(reduced_trace.expand(x.shape[:-1]))
