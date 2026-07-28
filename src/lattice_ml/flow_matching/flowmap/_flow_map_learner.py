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
        ordered: bool = False,
        t_leq_s: bool = True,
    ):
        """
        Args:
            flow_map (torch.nn.Module): The `FlowMap` to be trained.
            matching_objective (Callable): Defines the training loss.
            diffuser (Callable | None): If given, used to sample the source
                state `x_s` from data via its forward process. If `None`,
                `x_s` is instead obtained from the flow map's own current
                `Phi_{0,s}(x_0)`.
            ordered (bool): Whether `(s, t)` pairs are constrained to a
                triangle at all. Default is `False`.
            t_leq_s (bool): Which triangle `(s, t)` pairs are sampled from
                in `_prepare_t_span`: `t <= s` (default) or `t >= s`. Only
                relevant if `ordered` is `True`.
        """
        super().__init__()
        self.flow_map = flow_map
        self.matching_objective = matching_objective
        self.diffuser = diffuser
        self.ordered = ordered
        self.t_leq_s = t_leq_s
        self.trainer = Trainer(self)

    def training_step(self, batch, batch_idx=None):
        """Perform a training step to be used by Trainer."""
        x_0, = batch

        t_span = self._prepare_t_span(bsize=x_0.shape[0], device=x_0.device)
        s, _ = t_span

        if self.diffuser is not None:
            x_s, _ = self.diffuser((0, s), x_0)
        else:
            x_s = self.flow_map((0, s), x_0)

        x_t = self.flow_map(t_span, x_s)

        return self.matching_objective(self.flow_map, t_span, x_s, x_t)

    def _prepare_t_span(self, bsize, device):
        """
        Sample `t_span = (s, t)` uniformly over `t <= s` or `t >= s` triangles,
        depending on `self.t_leq_s`. If `self.ordered` is `False`, `s` and `t`
        are instead sampled independently over the whole unit square.
        """
        u1 = torch.rand((bsize,), device=device)
        u2 = torch.rand((bsize,), device=device)
        if not self.ordered:
            return (u1, u2)
        lo, hi = torch.minimum(u1, u2), torch.maximum(u1, u2)
        return (hi, lo) if self.t_leq_s else (lo, hi)

    def forward(self, t_span, x_s):
        """Evaluate the flow map being trained, `Phi_{s,t}(x_s)`."""
        return self.flow_map(t_span, x_s)
