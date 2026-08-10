# Created by Javad Komijani, 2026

"""Implements the flow map for Lie groups."""

# pylint: disable=invalid-name

from typing import Callable, Literal, Tuple

import torch
from torch import matrix_exp
from torch.func import jvp


__all__ = ["LieFlowMap", "LieFlowMapMatchingObjective"]


# =============================================================================
class LieFlowMap(torch.nn.Module):
    """
    Neural approximation of the flow map associated to a flow equation of
    Lie-group states.

    Given the ODE `dU/dt = v_t(U)` and a state `U_s` at source time `s`, the
    flow map `Phi_{s,t}(x_s)` returns the state `U_t` at target time `t`

        U_t = Phi_{s,t}(U_s) ,

    by integrating the ODE from `s` to `t`, without numerically integrating it.

    The flow map satisfies:

    - `Phi_{s,s} = id`
    - `Phi_{r,t} o Phi_{s,r} = Phi_{s,t}`  (semigroup property)
    - `Phi_{t,s}^{-1} = Phi_{s,t}`         (a consequence, not the definition)

    The boundary condition `Phi_{s,s} = id` is enforced exactly by
    parameterizing the map as

        Phi_{s,t}(U_s) = e^{(t - s) v_s(U_s) + (t - s)^2 f_{s,t}(U_s) / 2.} U_s

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

        Note: It is the user's responsibility to ensure both input functions
            returns algebra-valued tensors.
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

            return matrix_exp(delta_ts * v_s + (delta_ts**2 / 2) * f_ts) @ x_s

        if isinstance(t_eval, torch.Tensor):
            assert t_eval.ndim == 1, "`t_eval` must be 1d."

        x_eval = []
        for t_i in t_eval:
            # Run the flow map to time t_i
            x_eval.append(self.forward((s, t_i), x_s))

            # Update the state for the next round
            x_s, s = x_eval[-1], t_i

        return x_eval


# =============================================================================
class LieFlowMapMatchingObjective:
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

        # Rearranged Lagrangian condition (A, B, and C defined below):
        #    (v_t - v_s) / delta_ts = (f_ts / 2 +  C)
        #    C = \int_0^1 dz e^{zA} B e^{-zA})

        a_ts = torch.nan_to_num((v_t - v_s) / delta_ts, nan=0.0)

        f_ts, dfdt_ts = eval_jvp(flow_map.correction_fn, t_span, x_s, self.eps)

        A = delta_ts * v_s + (delta_ts**2 / 2) * f_ts
        B = 0.5 * (f_ts + delta_ts * dfdt_ts)
        C = dexp_integral(A, B)

        # Both LHS & HRS are of order delta_ts
        return squared_l2_distance(a_ts - f_ts, C - 0.5 * f_ts)


# =============================================================================
def eval_jvp(f, t_span: Tuple, x_s: torch.Tensor, eps: float | None = None):
    """
    Evaluate JVP of `f(t_span, x_s)` with respect to `t`.

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
def dexp_integral(
    A: torch.Tensor,
    B: torch.Tensor,
    num_samples: int | None = None,
) -> torch.Tensor:
    r"""
    Evaluates the integral

        I(A, B) = \int_0^1 e^{sA} B e^{-sA} ds,

    where `A` and `B` belong to a Lie algebra.

    This integral is exactly the correction term appearing in the derivative
    of a matrix exponential along a path: for `X = X(t)` with `Xdot = dX/dt`,

        d/dt e^{X(t)} = I(X, Xdot) @ e^{X(t)}.

    Args:
        A (torch.Tensor): Lie-algebra-valued tensor, shape `(..., n, n)`.
        B (torch.Tensor): Lie-algebra-valued tensor, same shape as `A`.
        num_samples (int | None): Number of Monte Carlo samples to average
            over, each drawing `s ~ Uniform(0, 1)` independently per batch
            element. If `None` (default), the integral is instead evaluated
            exactly via a JVP through `torch.matrix_exp`.

    Returns:
        torch.Tensor: `I(A, B)`, same shape as `A` and `B`.
    """
    if num_samples is None:
        _, dexp_val = jvp(torch.matrix_exp, (A,), (B,))
        return dexp_val @ torch.matrix_exp(-A)

    s_shape = (num_samples,) + A.shape[:-2] + (1, 1)
    s = torch.rand(s_shape, device=A.device, dtype=A.real.dtype)

    exp_pos = torch.matrix_exp(s * A)
    exp_neg = torch.matrix_exp(-s * A)

    return (exp_pos @ B @ exp_neg).mean(dim=0)


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
