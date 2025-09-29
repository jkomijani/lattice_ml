# Created by Javad Komijani, 2025

"""
This module provides the `Trainer` class for handling optimization,
training loops, loss computation, logging, and checkpointing.
"""

# pylint: disable=too-many-arguments, too-many-locals
# pylint: disable=too-many-positional-arguments

import logging
import time
import torch

from lattice_ml.functions import pow_special_unitary_group_


IS_MAIN_PROCESS = True  # TODO: Automate detection in distributed training

__all__ = ["Trainer"]


# =============================================================================
class Trainer:
    """
    Trainer class for learning dynamics using flow matching.
    """

    optimizer = None
    scheduler = None

    def __init__(self, dynamics_fn):
        """
        Initialize the trainer.

        Parameters
        ----------
        dynamics_fn : callable
            Dynamics function to be trained. Must expose `.parameters()`.
        """
        self.dynamics_fn = dynamics_fn

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
        parameters = self.dynamics_fn.parameters()
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
        n_samples = 0

        for (x_0,), (x_1,) in zip(data_loader0, data_loader1):
            bsize = x_0.shape[0]

            # Sample a random diffusion time per example, uniformly in [0, 1].
            # Shape: (batch_size,), reshaped for broadcasting
            t = torch.rand((bsize,), device=x_0.device)
            t_ = t.reshape((-1, *(1,) * (x_0.ndim - 2)))

            # Flow the data to time t
            # Compute the "true" flow from x_0 → x_1 at intermediate time t
            x_t = x_0 * (1 - t) + x_1 * t
            x_dot = x_1 - x_0

            # Predict flow using the learned (algebra) dynamics at (t, x_t)
            flow = self.dynamics_fn(t, x_t)

            # Compute flow-matching loss
            res = flow - x_dot
            loss = torch.mean(res * res)

            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Track accumulated loss
            with torch.no_grad():
                loss_sum += bsize * loss
                n_samples += bsize

        return loss_sum / n_samples

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
