# Created by Javad Komijani, 2024

"""
Trainer for diffusion models using implicit score matching.

Provides the `Trainer` class to handle optimization, training loops,
loss computation, logging, and checkpointing. Uses AdamW by default
and samples diffusion times uniformly for each batch. Logging includes
timestamps and optional progress reporting.
"""

import logging

import time
import torch


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S"
)

IS_MAIN_PROCESS = True  # will be automatic in future


__all__ = ["Trainer", "implicit_score_matching"]


# =============================================================================
class Trainer:
    """
    Trainer for diffusion models using implicit score matching.

    This class manages optimization, training history, and checkpoints.
    By default, it trains with implicit score matching loss, where the weight
    over time is given by the variance of the effective noise.

    Attributes
    ----------
    diffusion_process : object
        The diffusion process defining forward corruption and score model.
    optimizer : torch.optim.Optimizer or None
        Instantiated optimizer after calling ``execute``. Default is AdamW.
    scheduler : torch.optim.lr_scheduler._LRScheduler or None
        Learning rate scheduler (optional).
    train_history : dict
        Tracks 'epoch' (int) and 'loss' (list of float).
    hyperparam : dict
        Stores optimizer hyperparameters, e.g., learning rate.
    checkpoint_dict : dict
        Configuration for checkpointing and logging.
    loss_fn : callable
        Training loss function. Defaults to ``implicit_score_matching``.
    """

    optimizer = None
    scheduler = None

    loss_c0 = 0  # c_0 in the augmented loss functions
    action = None  # needed if loss_c0 > 0

    def __init__(self, diffusion_process):

        self._diffusion_process = diffusion_process

        # Initialize training history tracking
        self.train_history = {'epoch': 0, 'loss': []}

        # Default hyperparameters
        self.hyperparam = {'fused': torch.cuda.is_available()}

        # Checkpoint configuration
        self.checkpoint_dict = {'print_every': None}

        self.loss_fn = implicit_score_matching

    def execute(
        self,
        data_loader,
        n_epochs: int = 100,
        loss_c0: float = 0,
        action=None,
        optimizer_class=None,
        scheduler=None,
        hyperparam=None,
        checkpoint_dict=None,
    ):
        """
        Train the score model with implicit score matching.

        Parameters
        ----------
        data_loader : iterable
            Loads true samples for training.
        n_epochs : int, default=100
            Number of training epochs.
        loss_c0 : float, default=0.0
            If greater than zero, enables force regularization at t = 0.
        action : callable or None, optional
            The action to be used when `loss_c0` > 0. Ignored otherwise.
        loss_c0 : float, default=0,
            If larger than zero, it executed the force regularization at t=0.
        action : callable | None, optional
            Relavant only if loss_c0 is larger than 0 (default: None).
        optimizer_class : type, optional
            Optimizer class (default: AdamW).
        scheduler : callable, optional
            Learning rate scheduler constructor (default: None).
        hyperparam : dict, optional
            Extra optimizer hyperparameters such as learning rate.
        checkpoint_dict : dict, optional
            Checkpoint and logging configuration. Keys include:
              - ``print_every`` : int or None, print training progress
                every given number of epochs.
        """
        # Update the attributes of the instance
        if hyperparam is not None:
            self.hyperparam.update(hyperparam)

        if checkpoint_dict is not None:
            self.checkpoint_dict.update(checkpoint_dict)

        if optimizer_class is None:
            optimizer_class = torch.optim.AdamW

        self.loss_c0 = loss_c0
        self.action = action

        parameters = self._diffusion_process.score_fn.parameters()
        self.optimizer = optimizer_class(parameters, **self.hyperparam)

        if scheduler is not None:
            self.scheduler = scheduler(self.optimizer)

        if n_epochs > 0:
            self._train(data_loader, n_epochs)

    def _train(self, data_loader, n_epochs):

        self.train_history['loss'].extend([None] * n_epochs)

        is_main_process = IS_MAIN_PROCESS

        last_epoch = self.train_history['epoch']
        report_progress = self.checkpoint_dict['print_every'] is not None

        if is_main_process and report_progress:
            logging.info("Training started for %d epochs", n_epochs)

        t_1 = time.time()

        for epoch in range(last_epoch + 1, last_epoch + 1 + n_epochs):

            loss = self.step(data_loader)

            self._checkpoint(epoch, loss)

            if self.scheduler is not None:
                self.scheduler.step()

        t_2 = time.time()

        if is_main_process and report_progress:
            logging.info(
                "Training finished (%s); TIME = %.3g s", loss.device, t_2 - t_1
            )

    def step(self, data_loader):
        """Perform a train step."""

        process = self._diffusion_process

        loss_sum = 0
        n_samples = 0

        for x_0, in data_loader:
            bsize = x_0.shape[0]

            # Sample a random diffusion time per example, uniformly in [0, 1].
            diffusion_time = torch.rand((bsize,), device=x_0.device)

            # Diffuse the data at time t, returning noisy sample, injected
            # `noise/std`, and the effective noise standard deviation.
            x_t, eps, noise_std = process.run_for_training(x_0, diffusion_time)

            # Predict the score (gradient of log density) at (t, x_t).
            score = process.score_fn(diffusion_time, x_t)

            # Compute loss: implicit score matching weighted by noise variance.
            loss = self.loss_fn(score, eps, noise_std)

            # contribution from t = 0 if loss_c0 > 0
            if self.loss_c0 > 0:
                score0 = process.score_fn(0 * diffusion_time, x_0)
                df0 = score0 - self.action.algebra_force(x_0)
                loss = loss + self.loss_c0 * torch.mean(df0 * df0.conj()).real

            self.optimizer.zero_grad()  # clears old gradients from last steps
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                loss_sum += bsize * loss
                n_samples += bsize

        return loss_sum / n_samples

    @torch.no_grad()
    def _checkpoint(self, epoch, loss):

        is_main_process = IS_MAIN_PROCESS

        every = self.checkpoint_dict['print_every']

        # loss = self._model.device_handler.all_gather_into_tensor(loss).item()
        loss = loss.item()

        if is_main_process:

            self.train_history['epoch'] = epoch
            self.train_history['loss'][epoch - 1] = loss

            if every is not None and epoch % every == 0:
                logging.info("Epoch: %d | loss: %.4f", epoch, loss)


# =============================================================================
def implicit_score_matching(
    score: torch.Tensor,
    eps: torch.Tensor,
    noise_std: torch.Tensor
) -> torch.Tensor:
    r"""
    Implicit score matching loss.

    Computes the weighted MSE between the predicted and empirical scores,
    weighted by the effective noise variance at time `t`.


    Parameters
    ----------
    score : torch.Tensor
        Predicted score, shape (batch_size, ...).
    eps : torch.Tensor
        Sampled Gaussian noise added during forward diffusion process.
    noise_std : torch.Tensor
        Standard deviation of the effective noise at time ``t``.

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    n_c = score.shape[-1]

    # Residual between predicted and conditional scores (variance-weighted)
    res = score * noise_std + eps
    loss = torch.mean(res * res.conj()).real

    # Correcting for noise fluctuation
    fluctuation = torch.mean(eps * eps.conj()).real - (n_c ** 2 - 1) / n_c ** 2

    return (loss - fluctuation) * n_c ** 2
