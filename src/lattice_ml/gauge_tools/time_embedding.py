# Created by Javad Komijan, 2025

"""
Module for generating time-embedded weight tensors and modules.
"""


from typing import Tuple
import torch


__all__ = [
    "TimeEmbeddedWeight",
    "TimeModulatedWeight",  # alis -> TimeEmbeddedWeight; kept for legacy
    "SinusoidalTimeEncoder",
]


# =============================================================================
class TimeEmbeddedWeight(torch.nn.Module):
    """
    Constructs time-emebedded weight tensors.

    Args:
        weight_shape (tuple of int): Shape of the output weight tensor,
            excluding batch dimension.
        hidden_dim (int): Hidden dimension of the MLP (default 32).
        max_freq (int): Maximum frequencey in the time encoder (default 32).
            Overlooked if `time_encoder` is provided.
        time_encoder (nn.Module, optional): Module that encodes time.
            Defaults to `SinusoidalTimeEncoder(hidden_dim, max_freq=max_freq)`.
    """

    def __init__(
        self,
        weight_shape: Tuple[int],
        hidden_dim: int = 32,
        max_freq: int = 32,
        time_encoder=None
    ):
        super().__init__()

        self.weight_shape = weight_shape
        n_weight = int(torch.tensor(weight_shape).prod())

        if time_encoder is None:
            time_encoder = SinusoidalTimeEncoder(hidden_dim, max_freq=max_freq)

        self.time_encoder = time_encoder
        time_n_embed = self.time_encoder.n_embed

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(time_n_embed, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, n_weight),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute time-dependent weight tensor for given time.

        Args:
            t (Tensor): A 1D tensor representing time (batch dimension).

        Returns:
            Tensor: Weight tensor of shape `(len(t), *self.weight_shape)`.
        """
        t_emb = self.time_encoder(t)
        weight_t = self.mlp(t_emb).view(-1, *self.weight_shape)
        return weight_t

    def set_param2zero(self):
        """Set all trainable parameters to zero."""
        for param in self.mlp.parameters():
            torch.nn.init.zeros_(param)

    def set_param2normal(self, mean: float = 0.0, std: float = 1.0):
        """Set all trainable parameters to Gaussian with given mean and std."""
        for param in self.mlp.parameters():
            torch.nn.init.normal_(param, mean=mean, std=std)


TimeModulatedWeight = TimeEmbeddedWeight  # for legacy


# =============================================================================
class SinusoidalTimeEncoder(torch.nn.Module):
    """
    Implements a sinusoidal time encoding inspired by "Attention Is All You
    Need," where the frequencies change geometrically.

    Unlike the original paper where positions are integers, this class supports
    non-integer time values, typically within [0, 1]. The frequency spectrum
    can be adjusted using `max_freq` and `max_freq`.

    Args:
        n_embed (int): Length of the code vector (must be even).
        min_freq (float, int): Minimum angular frequency (default is 1).
        max_freq (float, int): Maximum angular frequency (default is 1000).
        inner_ndim (int): for reshaping the output (default is 0).
        trainable_freq (bool): Frequencies are trainable (defaults to False).
        trainable_ampl (bool): Amplitudes are trainable (defaults to False).
    """

    def __init__(
        self,
        n_embed: int,
        min_freq: float = 1.,
        max_freq: float = 1000.,
        inner_ndim: int = 0,
        trainable_freq: bool = False,
        trainable_ampl: bool = False
    ):

        assert n_embed % 2 == 0, "Embedding length must be even."

        super().__init__()

        self.n_embed = n_embed
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.inner_ndim = inner_ndim
        self.trainable_freq = trainable_freq
        self.trainable_ampl = trainable_ampl

        if trainable_freq:
            self.freq_ratio = torch.nn.Parameter(torch.rand(n_embed // 2))
        else:
            power = torch.arange(n_embed // 2) * (2 / n_embed)  # \in [0, 1)
            freq = max_freq / (max_freq / min_freq)**power
            self.register_buffer('freq', freq)

        if trainable_ampl:
            self.ampl = torch.nn.Parameter(torch.randn(n_embed))
        else:
            self.ampl = None

    def forward(self, time):
        """
        Computes the sinusoidal time encodingdding.

        Args:
            time (Tensor): A 1D tensor representing time (batch dimension).

        Returns:
            Tensor: A tensor of original shape (batch_size, n_embed) with
                    sinusoidal encoding. The tensor then reshaped to have
                    `inner_ndim` additional inner dimensions with unit lenght.
        """
        if self.trainable_freq:
            angle = time[:, None] * self.freq_ratio[None, :] * self.max_freq
        else:
            angle = time[:, None] * self.freq[None, :]

        shape = (len(time), self.n_embed, *(1,) * self.inner_ndim)
        encoded_time = torch.zeros(shape[:2], device=time.device)

        encoded_time[:, 0::2] = torch.sin(angle)
        encoded_time[:, 1::2] = torch.cos(angle)

        if self.trainable_ampl:
            encoded_time = self.ampl[None, :] * encoded_time

        return encoded_time.reshape(*shape)
