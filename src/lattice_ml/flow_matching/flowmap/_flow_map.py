# Created by Javad Komijani, 2026

"""Implements the flow map."""

from typing import Callable, Literal, Tuple

import torch
from torch.func import jvp


__all__ = ["FlowMap", "FlowMapMatchingObjective"]


# =============================================================================
class FlowMap(torch.nn.Module):
    """
    Neural approximation of the flow map associated to a flow equation.

    Given the ODE `dx/dt = v_t(x)` and a state `x_s` at source time `s`, the
    flow map `Phi_{s,t}(x_s)` returns the state `x_t` at target time `t`

        x_t = Phi_{s,t}(x_s) ,

    by integrating the ODE from `s` to `t`, without numerically integrating it.

    The flow map satisfies:

    - `Phi_{s,s} = id`
    - `Phi_{r,t} o Phi_{s,r} = Phi_{s,t}`  (semigroup property)
    - `Phi_{t,s}^{-1} = Phi_{s,t}`         (a consequence, not the definition)

    The boundary condition `Phi_{s,s} = id` is enforced exactly by
    parameterizing the map as

        Phi_{s,t}(x_s) = x_s + (t - s) v_s(x_s) + (t - s)^2 f_{s,t}(x_s) / 2.

    This parameterization is deliberate: the first-order (small `|t - s|`)
    behavior of the flow map is exactly a single Euler step along `v`. The map
    then needs only to learn the higher-order corrections (function `f`).
    """

    def __init__(
        self,
        underlying_dynamics_fn: Callable,
        correction_fn: Callable,
    ):
        """
        Args:
            underlying_dynamics_fn (Callable): ODE dynamics function
                `v(t, x) -> dx/dt` that this flow map approximates.
            correction_fn (Callable): Network computing `f(t_span, x_s)`, the
                term in the parameterization above (`t_span = (s, t)`).
        """
        super().__init__()
        self.underlying_dynamics_fn = underlying_dynamics_fn
        self.correction_fn = correction_fn

    def forward(
        self,
        t_span: Tuple,
        x_s: torch.Tensor,
        t_eval: Tuple[float] | torch.Tensor | None = None,
    ):
        """Evaluate the flow map `Phi_{s,t}(x_s)`.

        Args:
            t_span (Tuple): `(s, t)`, the source and target times. Each may
                independently be a plain float or a 0d/1d `torch.Tensor`
                (batched per-example if 1d, matching the batch size of states).
            x_s (torch.Tensor): State at time `s`.
            t_eval (Tuple[float] | torch.Tensor | None): Optional evaluation
                time(s) to evolve through sequentially instead of jumping
                directly to `t_span[1]`. If a `torch.Tensor`, must be 1d.

        Returns:
            torch.Tensor | List[torch.Tensor]:
            - If `t_eval` is `None`, returns a single state, at `t_span[1]`.
            - Otherwise, returns a list of states, one for each time in t_eval.

        Note:
            `underlying_dynamics_fn` is evaluated with no backward graph;
            gradients for its parameters (if any) are only needed by an
            additional loss term in self-distillation methods, which has
            nothing to do with this `forward` method.
        """
        s, t = t_span

        if t_eval is None:
            delta_ts = _prepare_time(t - s, x_s, "t - s")

            with torch.no_grad():
                v_s = self.underlying_dynamics_fn(s, x_s)

            f_ts = self.correction_fn(t_span, x_s)

            return x_s + delta_ts * v_s + (delta_ts**2 / 2) * f_ts

        if isinstance(t_eval, torch.Tensor):
            assert t_eval.ndim == 1, "`t_eval` must be 1d."

        x_eval = []
        for t_i in t_eval:
            # Run the flow map to time t_i
            x_eval.append(self.forward((s, t_i), x_s))

            # Update the state for the next round
            x_s, s = x_eval[-1], t_i

        return x_eval

    def correction_fn_and_partial_t(
        self, t_span: Tuple, x_s: torch.Tensor, eps: float | None = None
    ):
        """Evaluate `correction_fn(t_span, x_s)` and its derivative w.r.t. `t`.

        If `correction_fn` has a `forward_and_partial_t` method, the derivative
        is obtained from it directly; otherwise it falls back to `eval_jvp`
        (automatic differentiation, or finite differences if `eps` is given).

        Args:
            t_span (Tuple): `(s, t)`, the source and target times. Each may
                independently be a plain float or a 0d/1d `torch.Tensor`
                (batched per-example if 1d, matching the batch size of states).
            x_s (torch.Tensor): State at time `s`.
            eps (float | None): Finite-difference step size if not None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                `(correction_fn(.), d correction_fn/dt(.))`.
        """
        if hasattr(self.correction_fn, "forward_and_partial_t"):
            return self.correction_fn.forward_and_partial_t(t_span, x_s)
        return eval_jvp(self.correction_fn, t_span, x_s, eps=eps)


# =============================================================================
class FlowMapMatchingObjective:
    """
    Objective for training a flow map against a known `underlying_dynamics_fn`.

    `method="L"` enforces the Lagrangian condition,
    `d/dt Phi_{s,t}(x_s) = v_t(Phi_{s,t}(x_s))`, i.e. the transported point's
    trajectory, pushed forward in the target time `t`, obeys the known
    drift `v` at wherever it currently is.
    """

    def __init__(self, method: Literal["L"] = "L", eps: float | None = None):
        """
        Args:
            method (str): Which characterization of the flow map to train
                against: "L"` (Lagrangian), "E" (Eulerian), or "S" (Semigroup).
            eps (float | None): Finite-difference step size; if None, uses
                automatic differntiation. Default is None.
        """
        if method != "L":
            raise NotImplementedError(f"method={method!r} is not implemented.")
        self.method = method
        self.eps = eps

    def __call__(self, flow_map, t_span, x_s, x_t):
        """Compute the training objective.

        Args:
            flow_map (Callable): The flow map with signature `(t_span, x_s)`.
            t_span (Tuple): `(s, t)`, the source and target times. Each may
                independently be a plain float or a 0d/1d `torch.Tensor`
                (batched per-example if 1d, matching the batch size of states).
            x_s (torch.Tensor): Batch of states at the source time.
            x_t (torch.Tensor): Batch of states at the target time.

        Returns:
            torch.Tensor: Scalar loss value.

        Note:
            `method="L"` is not a self-distillation method, so
            `underlying_dynamics_fn` is evaluated with no backward graph
            here too, same as in `FlowMap.forward`.
        """
        v = flow_map.underlying_dynamics_fn

        s, t = t_span
        delta_ts = _prepare_time(t - s, x_s, "t - s")

        with torch.no_grad():
            v_t = v(t, x_t)
            v_s = v(s, x_s)

        # Rearranged Lagrangian condition:
        #    (v_t - v_s) / delta_ts = f + (delta_ts / 2 ) * df/dt

        a_ts = (v_t - v_s) / delta_ts

        f_ts, dfdt_ts = flow_map.correction_fn_and_partial_t(
            t_span, x_s, self.eps
        )

        return squared_l2_distance(a_ts - f_ts, (delta_ts / 2) * dfdt_ts)


# =============================================================================
def eval_jvp(f, t_span: Tuple, x_s: torch.Tensor, eps: float | None = None):
    """
    Evaluate JVP of `f` with respect to `t`.

    Args:
        f (Callable): Function `f(t_span, x_s)` to differentiate w.r.t. `t`.
        t_span (Tuple): `(s, t)`, the source and target times; differentiation
            is w.r.t. `t`. Each may independently be a plain float or a
            0d/1d `torch.Tensor` (batched per-example if 1d, matching the
            batch size of states).
        x_s (torch.Tensor): State at time `s`, held fixed.
        eps (float | None): Finite-difference step size; if None, uses AD.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: `(f(t_span, x_s), df/dt)`.
    """
    s, t = t_span

    if eps is not None:
        f_ts = f((s, t), x_s)
        dfdt_ts = (f((s, t + eps), x_s) - f_ts) / eps
        return f_ts, dfdt_ts

    return jvp(lambda t_: f((s, t_), x_s), (t,), (torch.ones_like(t),))


# =============================================================================
def squared_l2_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Default distance function: mean squared error (real part)."""
    res = a - b
    return torch.mean((res * res.conj()).real)


# =============================================================================
def _prepare_time(t, x: torch.Tensor, name: str = "t") -> torch.Tensor:
    """Convert `t` to a tensor if needed, and align it with `x`'s batch."""
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t, device=x.device)
    assert t.ndim <= 1, f"`{name}` must be 0d or 1d."
    if t.numel() > 1:
        t = t.view(-1, *[1] * (x.ndim - 1))
    return t
