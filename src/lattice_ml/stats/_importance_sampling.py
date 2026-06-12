# Created by Javad Komijani

"""
Importance sampling utilities for estimating expectations and diagnostics.

This module provides functions to compute importance weights, effective sample
size (ESS), and expectation estimates with uncertainty from samples drawn from
a proposal distribution q(x) and an unnormalized target distribution p(x).
"""

import torch
import numpy as np


__all__ = [
    "compute_importance_weights",
    "compute_ess",
    "estimate_expectation_with_importance_sampling"
]


def compute_importance_weights(logq, logp):
    """
    Compute importance sampling weights p(x) / q(x) from log-probabilities.

    The target density p(x) is assumed to be known only up to an unknown
    normalization constant (i.e. `logp` is unnormalized).

    Parameters
    ----------
    logq : torch.Tensor, shape (N,)
        Log-probabilities under the proposal distribution q(x).
    logp : torch.Tensor, shape (N,)
        Log of the *unnormalized* target density p(x).

    Returns
    -------
    weights : torch.Tensor, shape (N,)
        Importance weights proportional to p(x) / q(x), normalized such that
        their mean is exactly 1 over the sample (i.e. sum = N).

    Notes
    -----
    The partition function Z is estimated as:
        Z = (1/N) * sum_i exp(logp_i - logq_i)
    """

    # Estimate log(Z) using Monte Carlo under q
    logz = torch.logsumexp(logp - logq, dim=0) - np.log(logp.shape[0])

    # Normalize logp
    logp_normalized = logp - logz

    # Importance weights: p(x) / q(x)
    weights = torch.exp(logp_normalized - logq)

    return weights


def compute_ess(logq, logp, normalized_ess=True):
    """
    Compute the effective sample size (ESS) from importance weights.

    Parameters
    ----------
    logq : torch.Tensor, shape (N,)
        Log-probabilities under the proposal distribution q(x).
    logp : torch.Tensor, shape (N,)
        Log of the *unnormalized* target density p(x).
    normalized_ess : bool, default=True
        If True, return the normalized ESS in [1/N, 1].
        If False, return the unnormalized ESS in [1, N].

    Returns
    -------
    ess : torch.Tensor
        Effective sample size.

        If `normalized_ess=True`:
            ESS = mean(w)^2 / mean(w^2) ∈ (0, 1]

        If `normalized_ess=False`:
            ESS = (sum w)^2 / sum(w^2) ∈ (1, N]

    Notes
    -----
    - The ESS quantifies the quality of importance sampling:
        * High ESS → low variance (good overlap between p and q)
        * Low ESS  → high variance (poor overlap)

    - The two definitions are related by:
            ESS_unnormalized = N * ESS_normalized

    - In this implementation, the weights satisfy mean(w) = 1 exactly,
      so the formulas simplify to:

            ESS_normalized = 1 / mean(w^2)
            ESS_unnormalized = N / mean(w^2)
    """
    # Compute importance weights (mean(w) = 1 by construction)
    w = compute_importance_weights(logq, logp)

    # Compute second moment
    w2_mean = (w**2).mean()

    if normalized_ess:
        ess = 1 / w2_mean  # ESS in [1/N, 1]
    else:
        ess = w.shape[0] / w2_mean  # ESS in [1, N]

    return ess


def estimate_expectation_with_importance_sampling(x, logq, logp):
    """
    Estimate the expectation of an observable under a target distribution
    using importance sampling, together with uncertainty diagnostics.

    Both the standard deviation of the target distribution (estimated) and
    the standard error of the mean are returned.

    Parameters
    ----------
    x : torch.Tensor, shape (N,)
        Observable evaluated at sampled points.
    logq : torch.Tensor, shape (N,)
        Log-probabilities under the proposal distribution q(x).
    logp : torch.Tensor, shape (N,)
        Log of the *unnormalized* target density p(x).

    Returns
    -------
    avg : torch.Tensor
        Importance sampling estimate of E_p[x].
    std : torch.Tensor
        Estimated standard deviation of x under the target distribution.
    err : torch.Tensor
        Estimated standard error of E_p[x] based on the estimated standard
        deviation and effective sample size (ESS).

    Notes
    -----
    - Importance weights are defined as:
          w ∝ p(x) / q(x)
      with normalization chosen such that mean(w) = 1 exactly.

    - Expectation estimate:
          E_p[x] ≈ mean(w * x)

    - Variance estimate:
          Var ≈ mean(w * (x - avg)^2)

    - Standard error approximation:
          err ≈ sqrt(Var / (ESS - 1))
    """

    # Importance weights (mean(w) = 1 by construction)
    w = compute_importance_weights(logq, logp)

    # Effective sample size (unnormalized)
    ess = compute_ess(logq, logp, normalized_ess=False)

    # Importance sampling mean estimate
    avg = (w * x).mean()

    # Weighted variance estimate
    var = (w * (x - avg)**2).mean()

    # Standard deviation of the target distribution (estimated)
    std = var ** 0.5

    # Standard error of the mean (ESS-based approximation)
    err = std / (ess - 1)**0.5

    return avg, std, err
