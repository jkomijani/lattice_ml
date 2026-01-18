# Created by Javad Komijani, 2025-2026

"""Implements the flow-matching procedure."""

import torch

from lattice_ml.diffusion import Trainer
from lattice_ml.functions import pow_special_unitary_group_

from normflow.nn import RQSplineWithGrad


__all__ = ["FlowMatchingModel", "SUnFlowMatchingModel"]


# =============================================================================
class FlowMatchingModel(torch.nn.Module):
    """
    Flow matching model implementing an interpolation that is linear in X_0 and
    X_1, with a (possibly nonlinear) time reparameterization τ(t).

    The interpolation is defined as

        X_t = τ(t) X_0 + (1 - τ(t)) X_1,

    where t ∈ [0, 1] is a scalar (or batched) flow time. The function τ(t)
    parameterizes time and may be the identity or a nonlinear map.
    """

    def __init__(self, dynamics_fn, num_rqs_knots=None):
        """
        Initialize the flow matching model.

        Parameters
        ----------
        dynamics_fn : callable
            The dynamics function of the flow. Must have a `.parameters()`
            method for use with PyTorch optimizers.
        num_rqs_knots : int, optional
            Number of knots in the Rational Quadratic Spline used to model τ(t)
            in the interpolation formula. If provided it must be larger than 2.
            If None (default), a linear map is used.
        """
        super().__init__()
        self.dynamics_fn = dynamics_fn

        if num_rqs_knots is None:
            self.tau_func = LinearMapWithGrad()
        elif num_rqs_knots > 2:
            self.tau_func = RQSplineWithGrad(num_rqs_knots, smooth=True)
        else:
            raise ValueError("num_rqs_knots must be None or larger than 2.")

        # Components for training
        self.trainer = Trainer(self)

    def training_step(self, batch, batch_idx=None):
        """
        Perform a single training step to be used by Trainer.

        Samples a random flow time t ∈ [0, 1], computes the corresponding
        interpolated state x_t, and minimizes the squared error between
        the predicted and true time derivatives at (t, x_t).

        Parameters
        ----------
        batch : tuple
            Tuple (x_0, x_1) containing pairs of data points to interpolate.
        batch_idx : int, optional
            Index of the batch (unused).

        Returns
        -------
        torch.Tensor
            Scalar loss value for the batch.
        """
        x_0, x_1 = batch

        bsize = x_0.shape[0]
        shape = (bsize, *(1,) * (x_0.ndim - 1))  # for reshaping dtau/dt

        # Choose a random flow time per sample, uniformly in [0, 1].
        t = torch.rand((bsize,), device=x_0.device)
        tau, dtau_dt = self.tau_func(t)

        # τ(t) and dτ/dt are broadcast over spatial.
        tau = tau.reshape(shape)
        dtau_dt = dtau_dt.reshape(shape)

        # Flow the data to time t
        x_t = tau * x_0 + (1 - tau) * x_1
        x_dot = (x_0 - x_1) * dtau_dt

        # Predict flow using the learned dynamics at (t, x_t)
        deterministic_flow = self.dynamics_fn(t, x_t)

        # Compute flow-matching loss
        res = deterministic_flow - x_dot
        loss = torch.mean(res ** 2)

        return loss


# =============================================================================
class SUnFlowMatchingModel(torch.nn.Module):
    """
    Flow matching model implementing a group-aware interpolation between two
    SU(n) elements X_0 and X_1.

    The group-valued interpolation is defined as

        X_t = (X_1 X_0†)^{τ(t)} X_0,

    where t ∈ [0, 1] is a scalar (or batched) flow time. The function τ(t)
    parameterizes time and may be the identity or a nonlinear map.
    """
    def __init__(self, algebra_dynamics_fn, num_rqs_knots=None):
        """
        Initialize the flow matching model.

        Parameters
        ----------
        algebra_dynamics_fn : callable
            The Lie algebra dynamics function of the flow. Must have a
            `.parameters()` method for use with PyTorch optimizers.
        num_rqs_knots : int, optional
            Number of knots in the Rational Quadratic Spline used to model τ(t)
            in the interpolation formula. If provided it must be larger than 2.
            If None (default), a linear map is used.
        """
        super().__init__()
        self.algebra_dynamics_fn = algebra_dynamics_fn

        if num_rqs_knots is None:
            self.tau_func = LinearMapWithGrad()
        elif num_rqs_knots > 2:
            self.tau_func = RQSplineWithGrad(num_rqs_knots, smooth=True)
        else:
            raise ValueError("num_rqs_knots must be None or larger than 2.")

        # Components for training
        self.trainer = Trainer(self)

    def training_step(self, batch, batch_idx=None):
        """
        Perform a single training step to be used by Trainer.

        Samples a random flow time t ∈ [0, 1], computes the corresponding
        interpolated state x_t, and minimizes the squared error between
        the predicted and true time derivatives at (t, x_t).

        Parameters
        ----------
        batch : tuple
            Tuple (x_0, x_1) containing pairs of data points to interpolate.
        batch_idx : int, optional
            Index of the batch (unused).

        Returns
        -------
        torch.Tensor
            Scalar loss value for the batch.
        """
        x_0, x_1 = batch

        bsize = x_0.shape[0]
        shape0 = (bsize, *(1,) * (x_0.ndim - 2))  # for reshaping tau
        shape1 = (bsize, *(1,) * (x_0.ndim - 1))  # for reshaping dtau/dt

        # Choose a random flow time per sample, uniformly in [0, 1].
        t = torch.rand((bsize,), device=x_0.device)
        tau, dtau_dt = self.tau_func(t)

        # τ(t) and dτ/dt are broadcast over spatial and eigenvalues & group dim
        tau = tau.reshape(shape0)
        dtau_dt = dtau_dt.reshape(shape1)

        # Flow the data to time t
        #   y = x_1 @ x_0†  (relative transformation)
        #   pow_y = y^{tau}  (group exponential interpolation)
        #   log_y = log(y)  (algebra direction of the flow)
        y = x_1 @ x_0.adjoint()
        pow_y, log_y = pow_special_unitary_group_(y, tau.reshape(shape0))
        x_t = pow_y @ x_0
        alg_x_dot = log_y * dtau_dt

        # Predict flow using the learned (algebra) dynamics at (t, x_t)
        deterministic_alg_flow = self.algebra_dynamics_fn(t, x_t)

        # Compute flow-matching loss
        res = deterministic_alg_flow - alg_x_dot
        loss = torch.mean(res * res.conj()).real

        return loss


# =============================================================================
class LinearMapWithGrad(torch.nn.Module):
    """Linear map returning input tensor and its gradient of ones."""

    def forward(self, t: torch.Tensor):
        """Forward pass of the linear map returning input and its gradient."""
        return t, torch.ones_like(t)
