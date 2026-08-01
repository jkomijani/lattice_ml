# Created by Javad Komijan, 2025

"""Module for generating time-embedded weight tensors and modules."""


from typing import Tuple
import torch


__all__ = [
    "TimeEmbedding",
    "TimeEmbeddedWeight",
    "SinusoidalEncoder",
    "TimeModulatedWeight",  # alias -> TimeEmbeddedWeight; for legacy
    "SinusoidalTimeEncoder"  # alias -> SinusoidalEncoder; for legacy
]


# =============================================================================
class TimeEmbedding(torch.nn.Module):
    """Computes an embedding of time.

    Args:
        emb_dim (int): Size of the embedding.
        encoder_dim (int | None): Size of the time encoder's own output.
            Default `None`: falls back to `emb_dim` (no separate encoder
            size). Ignored if `time_encoder` is provided.
        hidden_dim (int | None): Hidden width of the MLP. Default `None`:
            falls back to `emb_dim` (no separate hidden width).
        max_freq (float | None): Maximum frequencey in the sinusoidal encoder
            if not None (default is None). Otherwise, a dense econder is used.
            Overlooked if `time_encoder` is provided.
        time_encoder (torch.nn.Module): Module that encodes time if provided;
            overrides `encoder_dim`/`max_freq`.
        n_coords (int | None): Number of scalar coordinates jointly encoded
            (e.g. `2` for a pair of times `(s, t)`). Default `None`: `t` has
            no coordinate axis at all -- `t.shape` is exactly the batch shape.
    """
    def __init__(
        self,
        emb_dim: int,
        encoder_dim: int | None = None,
        hidden_dim: int | None = None,
        max_freq: float | None = None,
        time_encoder: torch.nn.Module = None,
        n_coords: int | None = None,
    ):
        super().__init__()

        self.emb_dim = emb_dim
        self.n_coords = n_coords

        if time_encoder is None:
            encoder_dim = emb_dim if encoder_dim is None else encoder_dim
            if max_freq is None:
                time_encoder = DenseEncoder(encoder_dim)
            else:
                time_encoder = SinusoidalEncoder(
                    encoder_dim, max_freq=max_freq
                )

        self.time_encoder = time_encoder
        extended_encoder_dim = self.time_encoder.encoder_dim * (n_coords or 1)
        hidden_dim = emb_dim if hidden_dim is None else hidden_dim

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(extended_encoder_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the shared time embedding.

        Args:
            t (torch.Tensor): A tensor representing time, of shape `(*batch,)`
                if `n_coords is None`, or `(*batch, n_coords)` otherwise.

        Returns:
            Tensor: Embedding of shape `(*batch, emb_dim)`.
        """
        emb = self.time_encoder(t)
        if self.n_coords is not None:
            emb = emb.flatten(start_dim=-2)  # merge n_coords & encoder_dim

        return self.mlp(emb)


# =============================================================================
class TimeEmbeddedWeight(TimeEmbedding):
    """Constructs time-embedded weight tensors.

    A `TimeEmbedding` whose output is reshaped into an arbitrary `weight_shape`
    instead of returned as a flat embedding vector.

    Args:
        weight_shape (tuple of int): Shape of the output weight tensor,
            excluding batch dimension.
        encoder_dim (int): The time encoder's own output size (default 32).
            Overlooked if `time_encoder` is provided.
        hidden_dim (int): Hidden width of the MLP (default 32).
        max_freq (float | None): Maximum frequencey in the sinusoidal encoder
            if not None (default is None). Otherwise, a dense econder is used.
            Overlooked if `time_encoder` is provided.
        time_encoder (torch.nn.Module): Module that encodes time if provided;
            overrides `encoder_dim`/`max_freq`.
        n_coords (int | None): Number of scalar coordinates jointly encoded
            (e.g. `2` for a pair of times `(s, t)`. Default `None`: `t` has no
            coordinate axis at all -- `t.shape` is exactly the batch shape.
    """
    def __init__(
        self,
        weight_shape: Tuple[int],
        encoder_dim: int = 32,
        hidden_dim: int = 32,
        max_freq: float | None = None,
        time_encoder: torch.nn.Module = None,
        n_coords: int | None = None
    ):
        n_weight = int(torch.tensor(weight_shape).prod())
        super().__init__(
            emb_dim=n_weight,
            encoder_dim=encoder_dim,
            hidden_dim=hidden_dim,
            max_freq=max_freq,
            time_encoder=time_encoder,
            n_coords=n_coords,
        )
        self.weight_shape = weight_shape

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute time-dependent weight tensor for given time(s).

        Args:
            t (torch.Tensor): A tensor representing time, of shape `(*batch,)`
                if `n_coords is None`, or `(*batch, n_coords)` otherwise.

        Returns:
            Tensor: Weight tensor of shape `(*batch, *self.weight_shape)`.
        """
        batch_shape = t.shape if self.n_coords is None else t.shape[:-1]
        emb = super().forward(t)
        return emb.reshape(*batch_shape, *self.weight_shape)

    def set_param2zero(self):
        """Set all trainable parameters to zero."""
        for param in self.mlp.parameters():
            torch.nn.init.zeros_(param)

    def set_param2normal(self, mean: float = 0.0, std: float = 1.0):
        """Set all trainable parameters to Gaussian with given mean and std."""
        for param in self.mlp.parameters():
            torch.nn.init.normal_(param, mean=mean, std=std)


# =============================================================================
class SinusoidalEncoder(torch.nn.Module):
    """
    Implements a sinusoidal encoding inspired by "Attention Is All You Need,"
    where the frequencies change geometrically.

    Unlike the original paper where positions are integers, this class supports
    non-integer values, typically within [0, 1]. The frequency spectrum can be
    adjusted using `min_freq` and `max_freq`.

    Args:
        encoder_dim (int): Length of the code vector (must be even).
        min_freq (float): Minimum angular frequency (default is 1).
        max_freq (float): Maximum angular frequency (default is 1000).
        inner_ndim (int): For reshaping the output (default is 0).
        trainable_freq (bool): Frequencies are trainable (defaults to False).
        trainable_ampl (bool): Amplitudes are trainable (defaults to False).
    """

    def __init__(
        self,
        encoder_dim: int,
        min_freq: float = 1.,
        max_freq: float = 1000.,
        inner_ndim: int = 0,
        trainable_freq: bool = False,
        trainable_ampl: bool = False
    ):

        assert encoder_dim % 2 == 0, "Embedding length must be even."

        super().__init__()

        self.encoder_dim = encoder_dim
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.inner_ndim = inner_ndim
        self.trainable_freq = trainable_freq
        self.trainable_ampl = trainable_ampl

        if trainable_freq:
            self.freq_ratio = torch.nn.Parameter(torch.rand(encoder_dim // 2))
        else:
            power = torch.arange(encoder_dim // 2) * (2 / encoder_dim)
            freq = max_freq / (max_freq / min_freq)**power  # power \in [0, 1)
            self.register_buffer('freq', freq)

        if trainable_ampl:
            self.ampl = torch.nn.Parameter(torch.randn(encoder_dim))
        else:
            self.ampl = None

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Computes the sinusoidal encoding of t.

        Args:
            t (torch.Tensor): The input tensor, e.g., representing time.

        Returns:
            torch.Tensor: A tensor of original shape `(*t.shape, encoder_dim)`.
                It is then reshaped to have `inner_ndim` additional inner
                dimensions with unit lenght.
        """
        if self.trainable_freq:
            angle = t.unsqueeze(-1) * (self.freq_ratio * self.max_freq)
        else:
            angle = t.unsqueeze(-1) * self.freq

        encoded_t = torch.zeros((*t.shape, self.encoder_dim), device=t.device)

        encoded_t[..., 0::2] = torch.sin(angle)
        encoded_t[..., 1::2] = torch.cos(angle)

        if self.trainable_ampl:
            encoded_t = self.ampl * encoded_t

        out_shape = (*t.shape, self.encoder_dim, *(1,) * self.inner_ndim)
        return encoded_t.reshape(*out_shape)


class DenseEncoder(torch.nn.Module):
    """Implements a dense encoding.

    Args:
        encoder_dim (int): Length of the code vector.
    """

    def __init__(self, encoder_dim: int):

        super().__init__()

        self.encoder_dim = encoder_dim

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(1, 4), torch.nn.SiLU(),
            torch.nn.Linear(4, encoder_dim), torch.nn.SiLU()
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Computes the dense encoding of t.

        Args:
            t (torch.Tensor): The input tensor, e.g., representing time.

        Returns:
            torch.Tensor: A tensor of original shape `(*t.shape, encoder_dim)`.
        """
        return self.mlp(3.14 * t.unsqueeze(-1))


# Keep for legacy
TimeModulatedWeight = TimeEmbeddedWeight
SinusoidalTimeEncoder = SinusoidalEncoder
