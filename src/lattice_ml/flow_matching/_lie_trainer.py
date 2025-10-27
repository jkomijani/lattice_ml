# Created by Javad Komijani, 2025

"""
This module provides the `Trainer` class for handling optimization,
training loops, loss computation, logging, and checkpointing.
"""

# pylint: disable=too-many-arguments, too-many-locals
# pylint: disable=too-many-positional-arguments

import itertools
import logging
import time
import torch

from normflow.nn import RQSplineWithGrad
from lattice_ml.functions import pow_special_unitary_group_


IS_MAIN_PROCESS = True  # TODO: Automate detection in distributed training

__all__ = ["LieTrainer"]


# =============================================================================
class LieTrainer:
    """
    Trainer class for learning Lie group dynamics via flow matching.

    This class implements the optimization and training routines for
    flow matching on Lie groups, where the group-valued flow is defined as

        X_t = (X_1 X_0†)^{τ(t)} X_0,

    with τ(t) being a scalar (or batched) exponent parameterization that
    governs the interpolation between points X_0 and X_1 on the group.
    The exponent τ(t) can be modeled using a Rational Quadratic Spline (RQS)
    to model nonlinear temporal deformations.
    """

    optimizer = None
    scheduler = None

    def __init__(self, algebra_dynamics_fn, num_rqs_knots=None):
        """
        Initialize the LieTrainer instance for flow-matching dynamics.

        Parameters
        ----------
        algebra_dynamics_fn : callable
            The Lie algebra dynamics function to be trained. Must expose a
            `.parameters()` method for use with PyTorch optimizers.
        num_rqs_knots : int, optional
            Number of knots in the Rational Quadratic Spline used to model τ(t)
            in the group interpolation formula (x y†)^{τ(t)} y. If provided it
            must be larger than 2. If None (default), a linear map is used.
        """
        self.algebra_dynamics_fn = algebra_dynamics_fn

        if num_rqs_knots is None:
            self.tau_func = LinearMapWithGrad()
        elif num_rqs_knots > 2:
            self.tau_func = RQSplineWithGrad(num_rqs_knots, smooth=True)
        else:
            raise ValueError("num_rqs_knots must be None or larger than 2.")

        # Initialize training history tracking
        self.train_history = {'epoch': 0, 'loss': []}

        # Default hyperparameters
        self.hyperparam = {'fused': torch.cuda.is_available()}

        # Default checkpoint configuration
        self.checkpoint_dict = {'print_every': None}

    def execute(
        self,
        data_loader0,
        data_loader1,
        n_epochs: int = 100,
        optimizer_class=None,
        scheduler=None,
        hyperparam=None,
        checkpoint_dict=None,
    ):
        """
        Train the model using flow matching.

        Parameters
        ----------
        data_loader0 : iterable
            Loader for samples from the initial-time distribution.
        data_loader1 : iterable
            Loader for samples from the terminal-time distribution.
        n_epochs : int, optional
            Number of training epochs (default: 100).
        optimizer_class : type, optional
            Optimizer class to use (default: AdamW).
        scheduler : callable, optional
            Learning-rate scheduler constructor (default: None).
        hyperparam : dict, optional
            Additional optimizer hyperparameters such as learning rate.
        checkpoint_dict : dict, optional
            Checkpoint and logging configuration. Keys include:
              - ``print_every`` : int or None, print progress every N epochs.
        """
        # Update hyperparameters and checkpoint configuration
        if hyperparam is not None:
            self.hyperparam.update(hyperparam)

        if checkpoint_dict is not None:
            self.checkpoint_dict.update(checkpoint_dict)

        if optimizer_class is None:
            optimizer_class = torch.optim.AdamW

        # Initialize optimizer
        parameters = itertools.chain(
            self.algebra_dynamics_fn.parameters(),
            self.tau_func.parameters()
        )
        self.optimizer = optimizer_class(parameters, **self.hyperparam)

        # Initialize scheduler (if provided)
        if scheduler is not None:
            self.scheduler = scheduler(self.optimizer)

        # Start training loop
        if n_epochs > 0:
            self._train(data_loader0, data_loader1, n_epochs)

    def _train(self, data_loader0, data_loader1, n_epochs):
        """
        Internal training loop over multiple epochs.

        Parameters
        ----------
        data_loader0 : iterable
            Loader for samples from the initial-time distribution.
        data_loader1 : iterable
            Loader for samples from the terminal-time distribution.
        n_epochs : int
            Number of training epochs.
        """
        self.train_history['loss'].extend([None] * n_epochs)

        is_main_process = IS_MAIN_PROCESS
        last_epoch = self.train_history['epoch']
        report_progress = self.checkpoint_dict['print_every'] is not None

        if is_main_process and report_progress:
            logging.info("Training started for %d epochs", n_epochs)

        t_1 = time.time()

        for epoch in range(last_epoch + 1, last_epoch + 1 + n_epochs):
            loss = self.step(data_loader0, data_loader1)
            self._checkpoint(epoch, loss)

            if self.scheduler is not None:
                self.scheduler.step()

        t_2 = time.time()

        if is_main_process and report_progress:
            logging.info(
                "Training finished (%s); TIME = %.3g s", loss.device, t_2 - t_1
            )

    def step(self, data_loader0, data_loader1):
        """
        Perform one training step (one pass over paired batches).

        Parameters
        ----------
        data_loader0 : iterable
            Loader for samples at initial time.
        data_loader1 : iterable
            Loader for samples at terminal time.

        Returns
        -------
        torch.Tensor
            Average loss over all processed samples in the step.
        """

        loss_sum = 0
        time_weight_sum = 0

        for (x_0,), (x_1,) in zip(data_loader0, data_loader1):
            bsize = x_0.shape[0]
            shape0 = (bsize, *(1,) * (x_0.ndim - 2))  # for reshaping tau
            shape1 = (bsize, *(1,) * (x_0.ndim - 1))  # for reshaping dtau/dt

            # Sample a random diffusion time per example, uniformly in [0, 1].
            t = torch.rand((bsize,), device=x_0.device)
            tau, dtau_dt = self.tau_func(t)

            # Flow the data to time t
            # Compute the "true" flow from x_0 → x_1 at intermediate time t
            #   y = x_1 @ x_0†  (relative transformation)
            #   pow_y = y^{tau}  (group exponential interpolation)
            #   log_y = log(y)  (algebra direction of the flow)
            y = x_1 @ x_0.adjoint()
            pow_y, log_y = pow_special_unitary_group_(y, tau.reshape(shape0))
            x_t = pow_y @ x_0
            alg_x_dot = log_y

            # Predict flow using the learned (algebra) dynamics at (t, x_t)
            alg_flow = self.algebra_dynamics_fn(t, x_t)

            # Compute flow-matching loss
            res = alg_flow / dtau_dt.reshape(shape1) - alg_x_dot
            loss = torch.mean(res * res.conj()).real

            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Track accumulated loss
            with torch.no_grad():
                loss_sum += bsize * loss
                time_weight_sum += torch.sum(1 / dtau_dt ** 2)

        return loss_sum / time_weight_sum

    @torch.no_grad()
    def _checkpoint(self, epoch, loss):
        """
        Save training progress into history and log if required.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        loss : torch.Tensor
            Loss value from this epoch.
        """
        is_main_process = IS_MAIN_PROCESS
        every = self.checkpoint_dict['print_every']

        # Convert tensor to float
        loss = loss.item()

        if is_main_process:
            # Update training history
            self.train_history['epoch'] = epoch
            self.train_history['loss'][epoch - 1] = loss

            # Print/log progress if requested
            if every is not None and epoch % every == 0:
                logging.info("Epoch: %d | loss: %.4f", epoch, loss)


class LinearMapWithGrad(torch.nn.Module):
    """Linear map returning input tensor and its gradient of ones."""

    def forward(self, t: torch.Tensor):
        """Forward pass of the linear map returning input and its gradient."""
        return t, torch.ones_like(t)
