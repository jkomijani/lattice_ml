# Created by Javad Komijani, 2025

"""
Random SU(n) Dataset and DataLoader utilities for flow-matching training.

Classes
-------
SUnRandomDataset
    Generates batches of random SU(n) matrices on-the-fly.

Functions
---------
make_uniform_sun_dataloader
    Returns a DataLoader yielding batches from SUnRandomDataset.
"""

from torch.utils.data import Dataset, DataLoader
from normflow.prior import SUnPrior


__all__ = ["make_uniform_sun_dataloader"]


# -----------------------------------------------------------------------------
# Random SU(n) Dataset (batch-wise)
# -----------------------------------------------------------------------------
class SUnRandomDataset(Dataset):
    """
    Dataset that generates batches of random SU(n) matrices on-the-fly.

    The SU(n) matrices are drawn uniformly using `SUnPrior` class.

    Each __getitem__ returns a batch of size `batch_size`.

    Parameters
    ----------
    n : int
        Dimension of the SU(n) matrices.
    sample_shape : tuple of ints
        Shape of a single sample (excluding batch dimension). Each sample
        will have shape (*sample_shape, n, n).
    num_samples : int
        Total number of samples in the dataset.
    batch_size : int
        Number of samples per batch.
    """
    def __init__(self, n, sample_shape, num_samples, batch_size):
        self.prior = SUnPrior(n, sample_shape)
        self.num_samples = num_samples
        self.sample_shape = sample_shape
        self.batch_size = batch_size
        self.num_batches = 1 + (num_samples - 1) // batch_size

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        # Compute batch size for last batch
        bsize = min(self.batch_size, self.num_samples - idx * self.batch_size)
        batch = self.prior.sample(bsize)  # shape: (bsize, *sample_shape, n, n)
        return (batch,)

    def to(self, *args, **kwargs):
        """Move the prior distribution to a device."""
        self.prior.to(*args, **kwargs)
        return self


# -----------------------------------------------------------------------------
# Function to make a DataLoader
# -----------------------------------------------------------------------------
def make_uniform_sun_dataloader(n, sample_shape, num_samples, batch_size):
    """
    Create a DataLoader yielding random SU(n) batches each epoch.

    Parameters
    ----------
    n : int
        Dimension of the SU(n) matrices.
    sample_shape : tuple of ints
        Shape of a single sample (excluding batch dimension).
    num_samples : int
        Total number of samples per epoch.
    batch_size : int
        Number of samples per batch.

    Returns
    -------
    DataLoader
        A DataLoader instance yielding batches of random SU(n) samples.
    """
    dataset = SUnRandomDataset(n, sample_shape, num_samples, batch_size)
    # We set `batch_size=None` so that dataset items are returned directly.
    return DataLoader(dataset, batch_size=None, shuffle=False)
