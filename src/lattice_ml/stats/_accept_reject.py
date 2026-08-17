# Copyright (c) 2026 Javad Komijani

"""Utilities for accept/reject based sampling."""

import torch

__all__ = ["MetropolisHastings"]


class MetropolisHastings:
    """Utilities for independent Metropolis-Hastings sampling.

    The first dimension is the MH sequence dimension.
    All remaining dimensions are treated as independent chains.

    For example:

        (N,)          -> one chain of length N
        (N, B1, ...)  -> (B1, ...) chains of length N
    """

    @staticmethod
    @torch.no_grad()
    def accept_reject_step(y, logq, logp, ref=None):
        """
        Apply a Metropolis-Hastings accept/reject step to a batch of proposed
        samples, optionally chaining from a previous call.

        The MH algorithm is performed along the first dimension of `logq`
        and `logp`; any further dimensions are independent chains. `y` may
        carry additional trailing dimensions beyond those.

        Parameters
        ----------
        y : torch.Tensor
            Proposed samples. Shape (N, *chain_shape, *trailing_shape), where
            (N, *chain_shape) matches logq.shape.

        logq, logp : torch.Tensor
            Log probabilities under q and p. Shape (N, *chain_shape).

        ref : dict, optional
            Reference to the last accepted sample from a previous call,
            with keys 'sample', 'logq', 'logp' (shapes
            (*chain_shape, *trailing_shape), (*chain_shape), (*chain_shape)
            respectively). If None, the first proposed sample of this
            batch is used as the reference.

        Returns
        -------
        y, logq, logp : torch.Tensor
            Batch after the accept/reject step, same shapes as the inputs.

        accept_seq : torch.Tensor
            Boolean accept/reject sequence, shape (N, *chain_shape).
        """
        if ref is not None:
            # Prepend the reference as row 0: calc_accept_status always
            # treats row 0 as accepted, so this seeds the chain exactly
            # like a real previous sample would, with no special-casing.
            y = torch.cat([ref['sample'].unsqueeze(0), y], dim=0)
            logq = torch.cat([ref['logq'].unsqueeze(0), logq], dim=0)
            logp = torch.cat([ref['logp'].unsqueeze(0), logp], dim=0)

        accept_seq = MetropolisHastings.calc_accept_status(logq, logp)
        accept_ind = MetropolisHastings.calc_accept_indices(accept_seq)

        logq = torch.gather(logq, 0, accept_ind)
        logp = torch.gather(logp, 0, accept_ind)

        trailing_ndim = y.dim() - logq.dim()
        y_ind_shape = accept_ind.shape + (1,) * trailing_ndim
        y_ind = accept_ind.reshape(y_ind_shape).expand_as(y)
        y = torch.gather(y, 0, y_ind)

        if ref is not None:
            # Drop the injected reference row again.
            y, logq, logp = y[1:], logq[1:], logp[1:]
            accept_seq = accept_seq[1:]

        return y, logq, logp, accept_seq

    @staticmethod
    @torch.no_grad()
    def calc_accept_status(logq, logp, logq_minus_logp_ref=None):
        """Generate a Metropolis-Hastings accept/reject sequence.

        The MH algorithm is performed along the first dimension.
        All remaining dimensions are treated as independent chains.

        Parameters
        ----------
        logq : torch.Tensor
            Log probability under the proposal distribution q. Shape (N, ...).

        logp : torch.Tensor
            Log probability under the target distribution p. Shape (N, ...).

        logq_minus_logp_ref : torch.Tensor, default=None
            Reference value of log(q) - log(p). Shape (...).
            If None, the first sample is used as the initial reference.

        Returns
        -------
        torch.Tensor
            Boolean accept/reject sequence with the same shape as logq (logp).
        """
        if logq.shape != logp.shape:
            raise ValueError("logq and logp must have the same shape")

        logqp = logq - logp

        if logq_minus_logp_ref is None:
            logqp_ref = logqp[0].clone()
        else:
            assert logq_minus_logp_ref.shape == logqp[0].shape
            logqp_ref = logq_minus_logp_ref.clone()

        status = torch.empty_like(logqp, dtype=torch.bool)
        log_rand = torch.log(torch.rand_like(logqp))

        for i, logqp_i in enumerate(logqp):
            status[i] = log_rand[i] < (logqp_ref - logqp_i)
            logqp_ref = torch.where(status[i], logqp_i, logqp_ref)

        return status  # also called accept_seq

    @staticmethod
    def calc_accept_indices(accept_seq):
        """Return indices of the output configurations.

        The calculation is performed along the first dimension.
        All remaining dimensions are treated as independent chains.

        Parameters
        ----------
        accept_seq : torch.Tensor
            Boolean accept/reject sequence with shape (N, ...).

        Returns
        -------
        torch.Tensor
            Indices of the output configurations with the same shape of input.
        """
        assert accept_seq[0].all(), "The first sample must be accepted."

        indices = torch.zeros(
            accept_seq.shape, dtype=torch.long, device=accept_seq.device,
        )

        for i in range(1, len(accept_seq)):
            indices[i] = torch.where(accept_seq[i], i, indices[i - 1])

        return indices
