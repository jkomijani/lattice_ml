# Copyright (c) 2025 Javad Komijani


from functools import partial
import torch
import numpy as np


__all__ = [
    "Resampler",
    "bootstrap_resampler",
    "jackknife_resampler",
    "shuffling_resampler"
]


class Resampler:
    """
    A flexible class for resampling datasets using different methods.

    Parameters
    ----------
    method : str, optional
        The resampling method to use. Supported options are:

        - "bootstrap": Random resampling with replacement.
        - "jackknife": Leave-one-out resampling.
        - "shuffling": Random permutation of the data.

        Default is "bootstrap".

    Notes
    -----
    - The bootstrap and jackknife methods are standard statistical resampling
      techniques.

    - Shuffling is not a proper resampling method in the statistical sense.
      However, this can be useful, for instance, when applying an accept-reject
      procedure to a set of data to convert it into MCMC samples.
    """
    def __init__(self, method='bootstrap'):
        self.method = method

    def __call__(self, samples, n_resamples=100, binsize=1, batch_size=None):
        """
        Generate resampled versions of the input data.

        Parameters
        ----------
        samples : torch.Tensor or ndarray
            The dataset to be resampled. The first dimension corresponds to
            the number of samples.

        n_resamples : int, optional
            The number of resamples to generate. This is relevant only for the
            "bootstrap" and "shuffling" methods. Default is 100.

        binsize : int, optional
            Size of bins used to group consecutive samples before resampling.
            Binning can help reduce autocorrelation in sequential data.
            Default is 1 (no binning).

        batch_size : int or None, optional
            Number of bins to include in each bootstrap resample. This argument
            is ignored for the "jackknife" method. If None, the batch size is
            set equal to the number of bins (i.e., the number of original
            samples divided by the binsize).

        Yields
        ------
        resampled : torch.Tensor or ndarray
            A resampled version of the input data. The shape matches the input
            except for the first dimension, which depends on the resampling
            method.

        Notes
        -----
        - In the "jackknife" method, each resample leaves out one bin.
        - In the "bootstrap" method, bins are sampled with replacement.
        - In the "shuffling" method, the order of bins is randomly permuted.
        """
        l_b = samples.shape[0] // binsize  # lenght of binned samples
        binned_samples = samples[:(l_b * binsize)].reshape(l_b, binsize, -1)

        if isinstance(samples, torch.Tensor):
            arange = partial(torch.arange, device=samples.device)
            randint = partial(torch.randint, device=samples.device)
            randperm = partial(torch.randperm, device=samples.device)
        else:
            arange = np.arange
            randint = np.random.randint
            randperm = np.random.permutation

        match self.method:
            case 'jackknife':
                n_resamples = l_b
                get_indices = lambda i: arange(l_b)[arange(l_b) != i]
                resample_shape = ((l_b - 1) * binsize, *samples.shape[1:])
            case 'bootstrap':
                if batch_size is None:
                    batch_size = l_b  # useful if method is not 'jackknife'
                get_indices = lambda i: randint(l_b, size=(batch_size,))
                resample_shape = (l_b * binsize, *samples.shape[1:])
            case 'shuffling':
                get_indices = lambda i: randperm(l_b)
                resample_shape = (l_b * binsize, *samples.shape[1:])

        for i in range(n_resamples):
            yield binned_samples[get_indices(i)].reshape(*resample_shape)

    def resample_and_evaluate(self, samples, func, **kwargs):
        """
        Resample the data, evaluate a function on each resample, and compute
        the mean and standard error of the mean (STE) of the results.

        Parameters
        ----------
        samples : tensor or ndarray
            The main samples to be resampled.

        func : Callable
            The function to be evaluated on each resampled dataset.

        **kwargs :
            Additional parameters to be passed to the resampling process.

        Returns
        -------
        mean : tensor or ndarray
            Mean of the function outputs over all resamples. Shape matches
            the output of func.

        ste : tensor or ndarray
            Standard error of the mean of the function outputs over all
            resamples. Shape matches the output of func.

        Notes
        -----
        - For bootstrap, the standard error is simply the sample standard
          deviation of the resampled outputs. We use unbiased=True (ddof=1)
          to account for finite number of resamples.

        - For jackknife, the standard error formula is the square root of
              var_jack = (n-1)/n * sum((theta_i - theta_bar)^2).
         To compute the standard error in practice, we multiply the unbiased
         standard deviation of the estimates by `(n-1)/sqrt(n)`. This factor
         accounts for both the scaling of the leave-one-out variance and the
         fact that std() already uses ddof=1.

        - Shuffling is not a proper resampling method in the statistical sense.
          This can be useful, for instance, when applying an accept-reject
          procedure to a set of data to convert it into MCMC samples.
          Nevertherless, the error estimates obtained from shuffling should be
          treated with caution. Here, we calculate the standard error as for
          bootstrap method. It is the responsibility of the user to ensure that
          the error analysis is correct.
        """
        x = [func(q) for q in self(samples, **kwargs)]

        # Detect whether we're working with torch or numpy
        if isinstance(x[0], torch.Tensor):
            x = torch.stack(x)
            mean = torch.mean(x, dim=0)
            std = torch.std(x, dim=0, unbiased=True)
        else:
            x = np.array(x)
            mean = np.mean(x, axis=0)
            std = np.std(x, axis=0, ddof=1)

        std_2_sde_factor = 1
        if self.method == 'jackknife':
            n = len(x)
            std_2_sde_factor = (n - 1) / n ** 0.5

        return mean, std * std_2_sde_factor


bootstrap_resampler = Resampler(method='bootstrap')
jackknife_resampler = Resampler(method='jackknife')
shuffling_resampler = Resampler(method='shuffling')
