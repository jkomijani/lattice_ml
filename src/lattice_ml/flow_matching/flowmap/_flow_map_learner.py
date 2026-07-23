# Created by Javad Komijani, 2026

"""Implements the training framework for flow maps."""

from typing import Callable

import torch

from lattice_ml.diffusion import Trainer


__all__ = ["FlowMapLearner"]


# =============================================================================
class FlowMapLearner(torch.nn.Module):
    """
    Training framework for extracting flow maps from probability-flow models.
    """

    def __init__(
        self,
        flow_map: torch.nn.Module,
        matching_objective: Callable,
        diffuser: Callable | None = None,
        t_leq_s: bool = True,
    ):
        """
        Args:
            flow_map (torch.nn.Module): The `FlowMap` to be trained.
            matching_objective (Callable): Defines the training loss.
            diffuser (Callable | None): If given, used to sample the source
                state `x_s` from data via its forward process. If `None`,
                `x_s` is instead obtained from the flow map's own current
                `Phi_{s,0}(x_0)`.
            t_leq_s (bool): Which triangle `(t, s)` pairs are sampled from
                in `_prepare_t_and_s`: `t <= s` (default) or `t >= s`.
        """
        super().__init__()
        self.flow_map = flow_map
        self.matching_objective = matching_objective
        self.diffuser = diffuser
        self.t_leq_s = t_leq_s
        self.trainer = Trainer(self)

    def training_step(self, batch, batch_idx=None):
        """Perform a training step to be used by Trainer."""
        x_0, = batch

        t, s = self._prepare_t_and_s(bsize=x_0.shape[0], device=x_0.device)

        if self.diffuser is not None:
            x_s, _ = self.diffuser(x_0, t_0=0, t=s)
        else:
            x_s = self.flow_map(t=s, s=0, x_s=x_0)

        x_t = self.flow_map(t, s, x_s)

        return self.matching_objective(self.flow_map, t, s, x_t, x_s)

    def _prepare_t_and_s(self, bsize, device):
        """Sample `(t, s)` uniformly over the triangle `t <= s` or `t >= s`."""
        u1 = torch.rand((bsize,), device=device)
        u2 = torch.rand((bsize,), device=device)
        lo, hi = torch.minimum(u1, u2), torch.maximum(u1, u2)
        return (lo, hi) if self.t_leq_s else (hi, lo)

    def forward(self, t, s, x_s):
        """Evaluate the flow map being trained, `Phi_{t,s}(x_s)`."""
        return self.flow_map(t, s, x_s)
