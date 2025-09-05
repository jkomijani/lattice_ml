# Copyright (c) 2025 Javad Komijani

"""
Metropolis-Hastings accept/reject utilities for Markov Chain Monte Carlo.

This module provides functions to perform the Metropolis-Hastings acceptance
step, compute accepted indices, and analyze rejection patterns in batched
MCMC simulations.

Terminology
-----------
- log_proposal_minus_target : torch.Tensor or np.ndarray
    Difference between the log-probability of the proposed sample and the
    target log-probability. For a proposal PDF `q` and target PDF `p`, this
    is `log(q) - log(p)`. It is the quantity used to decide acceptance.

Functions
---------
def apply_accept_reject(y, logq, logp, **kwargs)
    Apply a Metropolis-Hastings accept/reject step to a batch of proposals.

calc_accept_status(log_proposal_minus_target, log_ref=None)
    Compute accept/reject decisions for a sequence of proposals.

calc_accept_indices(accept_seq)
    Return indices mapping rejected samples to their last accepted samples.

calc_accept_count(accept_seq)
    Count the number of times an accepted sample is repeated in the chain.

calc_tau_rejections_prob(accept_seq, max_tau=100)
    Compute the probability of consecutive rejection streaks of length tau.
"""

# pylint: disable=too-many-arguments, too-many-positional-arguments

import numpy as np
import torch

__all__ = [
    "apply_accept_reject",
    "calc_accept_status",
    "calc_accept_count",
    "calc_tau_rejections_prob"
]


def apply_accept_reject(
    y,
    logq,
    logp,
    y_ref=None,
    logq_ref=None,
    logp_ref=None,
    bookkeeping=False
):
    """
    Apply a Metropolis-Hastings accept/reject step to a batch of proposals.

    Parameters
    ----------
    y : torch.Tensor
        Proposed samples, shape (batch_size, *sample_shape).
    logq : torch.Tensor
        Log-probability of the proposal for each sample.
    logp : torch.Tensor
        Log-probability of the target for each sample.
    y_ref : torch.Tensor or None, optional
        Reference sample from the previous step. If None, the first proposal
        is automatically accepted.
    logq_ref : float or None, optional
        Log-probability of the proposal distribution at y_ref.
    logp_ref : float or None, optional
        Log-probability of the target distribution at y_ref.
    bookkeeping : bool, optional
        If True, returns additional acceptance information.

    Returns
    -------
    y_new : torch.Tensor
        Samples after applying accept/reject step.
    logq_new : torch.Tensor
        Proposal log-probabilities aligned with y_new.
    logp_new : torch.Tensor
        Target log-probabilities aligned with y_new.
    accept_seq : np.ndarray, optional
        Boolean array of acceptance status (only if bookkeeping=True).
    accept_indices : np.ndarray, optional
        Indices mapping output samples to accepted/repeated proposals
        (only if bookkeeping=True).

    Notes
    -----
    - If (y_ref, logq_ref, logp_ref) are all None, the first element in y is
      treated as accepted.
    - If provided, none can be None, and they are used as the initial state.
    """
    if (y_ref is None or logq_ref is None or logp_ref is None):
        if not (y_ref is None and logq_ref is None and logp_ref is None):
            raise ValueError("Provide all (y_ref, logq_ref, logp_ref) or none")
        log_ref = None  # First element automatically accepted
    else:
        log_ref = logq_ref - logp_ref  # Use given reference

    # 1) Determine acceptance sequence
    accept_seq = calc_accept_status(logq - logp, log_ref)

    # 2) Handle first element: if first rejected
    if not accept_seq[0]:
        y[0] = y_ref
        logq[0] = logq_ref
        logp[0] = logp_ref

    # 3) Map accepted/rejected samples
    accept_indices = calc_accept_indices(accept_seq)
    accept_indices_torch = torch.LongTensor(accept_indices).to(y.device)

    y_new = y.index_select(0, accept_indices_torch)
    logq_new = logq.index_select(0, accept_indices_torch)
    logp_new = logp.index_select(0, accept_indices_torch)

    if bookkeeping:
        return y_new, logq_new, logp_new, accept_seq, accept_indices
    return y_new, logq_new, logp_new


def calc_accept_status(log_proposal_minus_target, log_ref=None):
    """
    Compute the accept/reject status for a sequence of proposals using
    the Metropolis-Hastings criterion.

    Parameters
    ----------
    log_proposal_minus_target : 1D np.ndarray or torch.Tensor
        A sequnce of difference between proposal and target log-probabilities.
    log_ref : float, optional
        Reference log-probability for the first sample; if None, the first
        sample is accepted unconditionally.

    Returns
    -------
    accept_seq : np.ndarray
        Boolean array indicating which proposals are accepted.
    """
    # Faster if inputs are np.ndarray & python number (NO tensor)
    if isinstance(log_proposal_minus_target, torch.Tensor):
        log_proposal_minus_target = grab(log_proposal_minus_target)

    if log_ref is None:
        log_ref = log_proposal_minus_target[0]

    accept_seq = np.empty(len(log_proposal_minus_target), dtype=bool)
    rand_log = np.log(np.random.rand(len(log_proposal_minus_target)))

    for i, val in enumerate(log_proposal_minus_target):
        accept_seq[i] = rand_log[i] < (log_ref - val)
        if accept_seq[i]:
            log_ref = val

    return accept_seq


def calc_accept_indices(accept_seq):
    """
    Map each element to the most recent accepted index, or 0 if none exists.

    Parameters
    ----------
    accept_seq : np.ndarray
        Boolean array where True means accepted and False means rejected.

    Returns
    -------
    indices : np.ndarray
        For each position i, the index of the latest accepted sample at or
        before i. If no acceptance has occurred yet, the value is 0.
    """
    indices = np.arange(len(accept_seq))
    last_accepted = 0
    for i, accepted in enumerate(accept_seq):
        if accepted:
            last_accepted = i
        else:
            indices[i] = last_accepted
    return indices


def calc_accept_count(accept_seq):
    """
    Count the number of times an accepted sample is repeated in the chain.

    In a Metropolis-Hastings sequence, a rejected proposal means the current
    sample is kept for another iteration. This function returns the number of
    times each accepted sample is effectively repeated until the next sample
    is accepted.

    Parameters
    ----------
    accept_seq : array-like of bool
        Boolean sequence where True indicates an accepted proposal.

    Returns
    -------
    multiplicity : np.ndarray
        Array of counts indicating how many times each accepted sample
        persists due to rejections of subsequent proposals.
    """
    ind = np.where(accept_seq)[0]  # indices of accepted samples
    multiplicity = ind[1:] - ind[:-1]  # counts of rejections between accepts
    return multiplicity


def calc_tau_rejections_prob(accept_seq, max_tau=100):
    """
    Compute the probability distribution of tau consecutive rejections in a
    Metropolis-Hastings accept/reject sequence.

    Parameters
    ----------
    accept_seq : np.ndarray
        Boolean array of accepted (True) and rejected (False) samples.
        Example: [True, False, False, False, False, True, False]
    max_tau : int
        Maximum streak length of consecutive rejections to consider.

    Returns
    -------
    p_tau : np.ndarray
        Array of probabilities for rejection streaks of length 1..max_tau.
        - p_tau[0] gives the fraction of all rejected samples.
        - p_tau[1] gives the fraction of positions where at least 2 consecutive
          rejections occur. In the example above, p_tau[1] = 0.5 because there
          are 3 positions with 2 consecutive rejections out of 6 possible
          overlapping pairs:  [(T,F), (F,F), (F,F), (F,F), (F,T), (T,F)].
        - Similarly, p_tau[tau] counts positions with at least tau + 1
          consecutive rejections.
    """
    # Boolean array where True indicates a rejection
    rej_seq = ~accept_seq

    # Temporary array to track positions where streaks occur
    tau_rej_seq = rej_seq.copy()

    # Probability of a single rejection (streak of length 1)
    p_tau = np.zeros(max_tau)
    p_tau[0] = np.mean(tau_rej_seq)

    # Loop over possible streak lengths tau = 1..max_tau-1
    for tau in range(1, max_tau):
        # Identify positions where a streak of length tau + 1 occurs:
        # logical AND of previous streak (length tau) & ...
        tau_rej_seq = tau_rej_seq[:-1] & rej_seq[tau:]
        # Mean gives the probability of this streak length
        p_tau[tau] = np.mean(tau_rej_seq)

    return p_tau


def grab(x):
    """Convert a PyTorch tensor to a NumPy array."""
    return x.detach().cpu().numpy()
