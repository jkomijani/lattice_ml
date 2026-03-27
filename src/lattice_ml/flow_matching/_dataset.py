# Created by Javad Komijani, 2025

"""
Dataset and DataLoader utilities for flow-matching training.
"""

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset


__all__ = [
    'IIDPriorDataset',
    'make_paired_dataloader',
    'make_random_dataloader',
    'make_process_dataloader',
    'make_hmc_dataloader'
]


class PairedDataset(Dataset):
    """
    Dataset returning a pair of samples (x0, x1) for paired-data tasks.

    Can be used in Flow Matching, where training requires data at endpoints.

    Args:
        dataset0 (Dataset): Dataset for the first endpoint (x0).
        dataset1 (Dataset): Dataset for the second endpoint (x1).
        fixed_pairing (bool): If True, use a deterministic pairing.
            If False (default), sample x1 independently at random for each x0.
    """
    def __init__(self, dataset0, dataset1, fixed_pairing: bool = False):

        if not (hasattr(dataset0, "__len__") and hasattr(dataset1, "__len__")):
            raise TypeError(
                "PairedDataset requires both datasets to have __len__. "
                "For IterableDataset, consider PairedIterableDataset instead."
            )

        self.dataset0 = dataset0
        self.dataset1 = dataset1
        self.fixed_pairing = fixed_pairing

    def __len__(self):
        return min(len(self.dataset0), len(self.dataset1))

    def __getitem__(self, idx):
        x0 = self.dataset0[idx]
        if self.fixed_pairing:
            x1 = self.dataset1[idx]
        else:
            j = torch.randint(0, len(self.dataset1), (1,)).item()
            x1 = self.dataset1[j]
        return *x0, *x1


class PairedIterableDataset(IterableDataset):
    """
    IterableDataset yielding paired batches (x0_batch, x1_batch).

    Can be used in Flow Matching, where training requires data at endpoints.

    Both dataset0 and dataset1 can be either:
        - Dataset (finite, will be wrapped to iterable)
        - IterableDataset (stochastic)

    If the inputs are already IterableDatasets, batch_size is ignored.

    Each iteration returns a tuple of batches:
        x0_batch, x1_batch

    Warning:
        If both inputs are finite Datasets, internal batching is sequential
        and no shuffling occurs. Use `PairedDataset` instead if shuffling is
        required.
    """

    def __init__(self, dataset0, dataset1, batch_size: int):
        n_non_iterables = 0

        # Wrap non-iterable datasets into DatasetToIterable
        if not isinstance(dataset0, IterableDataset):
            dataset0 = DatasetToIterable(dataset0, batch_size)
            n_non_iterables += 1
        if not isinstance(dataset1, IterableDataset):
            dataset1 = DatasetToIterable(dataset1, batch_size)
            n_non_iterables += 1

        # Warn if both are finite Datasets
        if n_non_iterables == 2:
            print(
                "Warning: both inputs are finite Datasets. "
                "Internal batching is sequential, so shuffling is disabled. "
                "Use PairedDataset if you need shuffling."
            )

        self.dataset0 = dataset0
        self.dataset1 = dataset1

    def __len__(self):
        return min(len(self.dataset0), len(self.dataset1))

    def __iter__(self):
        iter_x0 = iter(self.dataset0)
        iter_x1 = iter(self.dataset1)

        # Yield paired batches
        for x0_batch, x1_batch in zip(iter_x0, iter_x1):
            yield *x0_batch, *x1_batch


class DatasetToIterable(IterableDataset):
    """Wraps a finite Dataset to yield sequential batches without shuffling."""

    def __init__(self, dataset: Dataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            batch = self.dataset[i:i + self.batch_size]  # may be a tuple!
            yield batch  # already a tuple


class IIDPriorDataset(Dataset):
    """
    Dataset that generates batches of i.i.d. samples from a prior distribution.

    Parameters
    ----------
    prior : object
        Distribution-like object with a `.sample(batch_size)` method that
        returns a batch of random samples (e.g., SU(n) matrices).
    num_samples : int
        The total size of dataset in one epoch.

    Yields
    ------
    tuple of torch.Tensor
        A tuple containing one random sample.
    """
    def __init__(self, prior, num_samples: int):
        self.prior = prior
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        samples = self.prior.sample(1)
        return (samples[0],)


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


def make_paired_dataloader(dataset0, dataset1, batch_size: int):
    """
    Create a DataLoader yielding paired batches (x0_batch, x1_batch).

    Chooses between PairedDataset and PairedIterableDataset depending
    on whether the inputs are Dataset or IterableDataset.

    Parameters
    ----------
    dataset0 : Dataset or IterableDataset
        Dataset for the first endpoint (x0).
    dataset1 : Dataset or IterableDataset
        Dataset for the second endpoint (x1).
    batch_size : int
        Batch size for training. Used only when wrapping finite Datasets.

    Returns
    -------
    DataLoader
        A DataLoader instance yielding tuples of batches:
        (x0_batch, x1_batch)
    """
    # Determine if either input is iterable
    is_iterable0 = isinstance(dataset0, IterableDataset)
    is_iterable1 = isinstance(dataset1, IterableDataset)

    # Choose dataset class
    if is_iterable0 or is_iterable1:
        # Use PairedIterableDataset
        dataset = PairedIterableDataset(
            dataset0, dataset1, batch_size=batch_size
        )
        return DataLoader(dataset, batch_size=None, shuffle=False)
    else:
        # Use PairedDataset
        dataset = PairedDataset(dataset0, dataset1)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)


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
