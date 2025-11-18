# Created by Javad Komijani, 2025

"""For computing generalized generalized wrapped Gaussian distribution.
"""

# pylint: disable=invalid-name  # for logU

import torch

from lattice_ml.functions import enumerate_sun_preimages

__all__ = ["wrapped_gaussian_sun_score"]


def wrapped_gaussian_sun_score(
    logU: torch.Tensor, sigma: float, max_branch_shift: int = 1
):
    """Compute the score (derivative of log-probability) of a generalized
    wrapped Gaussian distribution on su(n).

    This function computes integer-shifted branches of logU and computes the
    score matrix proportional to the weighted combination of the branches.

    Each branch is weighted by a Gaussian factor based on its Frobenius norm
    (trace of X†X).

    Args:
        logU (torch.Tensor): AntiHermitian su(n) matrix of shape `(..., n, n)`.
        sigma (float): Standard deviation controlling the unwrapped Gaussian.
        max_branch_shift (int): Maximum integer shift for eigenvalues.

    Returns:
        torch.Tensor: Weighted sum of integer-shifted branches of logU.
    """
    # Enumerate integer-shifted preimages
    stack = torch.stack(enumerate_sun_preimages(logU, max_branch_shift), dim=0)

    def frobenius_norm_sq(x):
        """Return squared Frobenius norm, i.e. Tr(X X†)."""
        return (x.real**2 + x.imag**2).sum(dim=(-2, -1), keepdim=True)

    # Compute Gaussian exponents for each preimage
    exponents = -(0.5 / sigma**2) * frobenius_norm_sq(stack)

    # Stabilization: subtract max exponent to avoid overflow
    max_expon, _ = exponents.max(dim=0, keepdim=True)

    weights = torch.exp(exponents - max_expon)
    tot_weight = weights.sum(dim=0)

    # Weighted sum of preimages
    weighted_preimages_sum = (stack * weights).sum(dim=0)

    score = (-1 / sigma**2) * weighted_preimages_sum / tot_weight

    return score
