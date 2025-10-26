# Copyright (c) 2025 Javad Komijani

"""
modal_analysis.py

Tools for analyzing the modality (number of peaks or modes) in the distribution
of order parameters, such as the magnetization in the Ising model.

Included methods
----------------
1. Binder cumulant test:
   - Uses the fourth-order Binder cumulant
     U4 = 1 - <M^4> / (3 <M^2>^2)
     to distinguish between disordered (unimodal) and ordered (bimodal)
     phases. This is a simple and physically motivated heuristic widely
     used in Monte Carlo studies of the Ising model.


Planned additions
-----------------
2. Spline-based mode detection:
   - Fits a smooth, monotonic approximation to the empirical CDF using
     either a Rational Quadratic Spline (RQS) or a monotone cubic spline
     (e.g., PCHIP). The derivative of this CDF provides a smooth density
     estimate, and the number of local maxima of this density corresponds
     to the number of modes. This approach avoids kernel bandwidth
     selection and provides analytically differentiable estimates
     (Durkan et al., 2019).


Additional methods
------------------
- Silverman’s bootstrap test for multimodality in KDEs.
- Hartigan’s dip test for nonparametric unimodality assessment.
"""

from typing import Tuple
import torch


__all__ = ["binder_modes", "binder_cumulant"]


def binder_modes(x: torch.Tensor, threshold: float = 1/3) -> Tuple[int, float]:
    """
    Estimate the number of modes in the x distribution using the Binder
    cumulant criterion and return the Binder cumulant value.

    The Binder cumulant is defined as:
        U4 = 1 - <M^4> / (3 <M^2>^2)
    where <.> denotes the average over equilibrium samples.

    It quantifies the shape of the x distribution:
      - U4 ≈ 0   → unimodal (symmetric / disordered phase)
      - U4 ≈ 2/3 → bimodal  (broken-symmetry / ordered phase)
      - U4 ≈ 1/3 → near the critical region

    Args:
        x (torch.Tensor): 1D tensor of x samples.
        threshold (float): Cutoff value to distinguish phases. Default 1/3.

    Returns:
        n_modes (int): Estimated number of modes:
            1 → unimodal (symmetric)
            2 → bimodal (broken symmetry)
        u_4 (float): Computed Binder cumulant value.
    """
    u_4 = binder_cumulant(x)
    n_modes = 2 if u_4 > threshold else 1
    return n_modes, u_4


def binder_cumulant(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the Binder cumulant (U4) for an Ising-like x dataset.

    The Binder cumulant is defined as: :math:`U4 = 1 - <M^4> / (3 <M^2>^2)`,
    where :math:`<.>` denotes the average over equilibrium samples.

    Args:
        x (torch.Tensor): 1D tensor of x samples

    Returns:
        torch.Tensor: scalar Binder cumulant value
    """
    m2 = torch.mean(x ** 2)
    m4 = torch.mean(x ** 4)
    return 1 - m4 / (3 * m2 ** 2)
