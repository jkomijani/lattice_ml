# Created by Javad Komijani, 2025

"""
Dataset and DataLoader utilities for flow-matching training.
"""

from torch.utils.data import DataLoader, IterableDataset


__all__ = [
    'make_random_dataloader',
    'make_process_dataloader',
    'make_hmc_dataloader'
]


class IIDPriorDataset(IterableDataset):
    """
    IterableDataset that generates batches of i.i.d. samples from a prior
    distribution.

    Parameters
    ----------
    prior : object
        Distribution-like object with a `.sample(batch_size)` method that
        returns a batch of random samples (e.g., SU(n) matrices).
    batch_size : int
        Number of samples per batch.
    num_batches : int, optional (default=1)
        Total number of batches to generate when iterating over the dataset.

    Yields
    ------
    tuple of torch.Tensor
        A tuple containing one batch of random samples.
    """
    def __init__(self, prior, batch_size: int, num_batches: int = 1):
        self.prior = prior
        self.batch_size = batch_size
        self.num_batches = num_batches

    def __iter__(self):
        # Generate num_batches independent sample batches
        for _ in range(self.num_batches):
            samples = self.prior.sample(self.batch_size)
            yield (samples,)  # wrap in tuple to match DataLoader conventions


class ProcessDataset(IterableDataset):
    """
    IterableDataset that wraps a source DataLoader and applies a processing
    function to each batch.

    This class is ideal for cascading datasets, where the output of one stage
    serves as the input for the next stage.

    Parameters
    ----------
    processor : Callable
        Callable taking a tensor batch and returning a processed batch.
    source_dataloader : DataLoader or Iterable
        A DataLoader yielding batches as single-tensor tuples, e.g. `(x,)`.

    Yields
    ------
    tuple of torch.Tensor
        A tuple containing the processed batch.
    """

    def __init__(self, processor, source_dataloader):
        self.processor = processor
        self.source_dataloader = source_dataloader

    def __iter__(self):
        """
        Iterate over the source dataloader, apply `processor` to each batch,
        and yield the processed batch.
        """
        # Each batch from the source dataloader must be a single-tensor tuple
        for x, in self.source_dataloader:
            y = self.processor(x)
            yield (y,)  # wrap in tuple to match DataLoader conventions


class HMCDataset(IterableDataset):
    """
    IterableDataset that generates a Markov chain via HMC updates.

    Each iteration applies one HMC step and yields the updated batch.

    Parameters
    ----------
    hmc : object
        HMC sampler instance with a `.step(x)` method returning (new_x, info).
    initial_cfgs : torch.Tensor
        Initial batch of configurations to start the Markov chain.
    num_batches : int, optional (default=1)
        Total number of batches to generate when iterating over the dataset.

    Yields
    ------
    tuple of torch.Tensor
        A tuple containing the current batch of updated configurations.
    """
    def __init__(self, hmc, initial_cfgs, num_batches: int = 1):
        self.hmc = hmc
        self.cfgs = initial_cfgs  # current state of the Markov chain
        self.num_batches = num_batches

    def __iter__(self):
        # Advance the chain num_batches times
        for _ in range(self.num_batches):
            self.cfgs, _ = self.hmc.step(self.cfgs)
            yield (self.cfgs,)  # yield current state as one batch


def make_random_dataloader(prior, batch_size, num_batches):
    """
    Create a DataLoader yielding independent batches from a prior distribution.

    Parameters
    ----------
    prior : object
        Distribution-like object with a `.sample(batch_size)` method that
        returns a batch of random samples (e.g., SU(n) matrices).
    batch_size : int
        Number of samples per batch.
    num_batches : int, optional (default=1)
        Total number of batches to generate when iterating over the dataset.

    Returns
    -------
    DataLoader
        A DataLoader instance yielding batches of random samples.
    """
    dataset = IIDPriorDataset(prior, batch_size, num_batches)

    # Note: `batch_size=None` means dataset items are returned directly.
    return DataLoader(dataset, batch_size=None, shuffle=False)


def make_process_dataloader(processor, source_dataloader):
    """
    Create a DataLoader that yields batches processed by a callable.

    Each batch from the source_dataloader is passed through the processor
    callable, and the resulting processed batch is yielded. Designed for
    multi-stage pipelines or iterative transformations.

    Parameters
    ----------
    processor : callable
        Callable taking a tensor batch and returning a processed batch.
    source_dataloader : DataLoader or Iterable
        A DataLoader yielding batches as single-tensor tuples, e.g., `(x,)`.

    Returns
    -------
    DataLoader
        A DataLoader instance yielding batches of processed data.
    """
    dataset = ProcessDataset(processor, source_dataloader)

    # batch_size=None ensures each dataset item is returned directly.
    return DataLoader(dataset, batch_size=None, shuffle=False)


def make_hmc_dataloader(hmc, initial_cfgs, num_batches):
    """
    Create a DataLoader yielding batches of configurations generated via HMC.

    Each iteration applies one HMC step and yields the updated batch.

    Parameters
    ----------
    hmc : object
        HMC sampler instance with a `.step(x)` method returning (new_x, info).
    initial_cfgs : torch.Tensor
        Initial batch of configurations to start the Markov chain.
    num_batches : int
        Total number of batches to generate when iterating over the dataset.

    Returns
    -------
    DataLoader
        A DataLoader instance yielding batches of updated configurations.
    """
    dataset = HMCDataset(hmc, initial_cfgs, num_batches)

    # Note: `batch_size=None` means dataset items are returned directly.
    return DataLoader(dataset, batch_size=None, shuffle=False)
