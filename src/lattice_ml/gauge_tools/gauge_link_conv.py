# Copyright (c) 2025 Javad Komijani

"""
Wilson line convolutional layers for lattice gauge theory.

This module provides gauge-equivariant layers that update lattice gauge links
using short Wilson lines starting at the tail of a link and ending at its head.
"""

import torch

from .wilson_staples import compute_staples
from .time_embedding import TimeEmbeddedWeight


__all__ = ["GaugeLinkConv", "TimeEmbeddedStapleLayer"]

matmul = torch.matmul


# =============================================================================
class GaugeLinkConv(torch.nn.Module):
    """Gauge-equivariant link convolution layer for lattice gauge fields.

    GaugeLinkConv updates gauge links using gauge-covariant linear combinations
    of Wilson staples. Only nearby links are mixed, similar to a convolution in
    standard ML layers. The linear weights are time-dependent, allowing dynamic
    evolution of the gauge links.

    The name reflects that it is:
        - Gauge-covariant (preserves link transformations)
        - Operates on links (not plaquettes or other lattice objects)
        - Convolution-like (local aggregation over nearby staples)

    The output is not unitary, but is scaled so its Frobenius norm equals
    sqrt(n_c), as in unitary matrices.

    Note:
        Tensors are expected by default to have spatial lattice axes before
        the link direction axis (sites_before_link=True). Set to False if your
        tensor uses link axis before lattice sites.

        It is possible to set the input and ouput channels to None. If set to
        None, a singleton channel axis is automatically added to inputs before
        processing and/or removed afterwards. This allows layers to operate in
        both channel-free and channel-based architectures.

        This cannot be used for U(1).
    """

    def __init__(
        self,
        in_channels: int | None,
        out_channels: int | None,
        spatial_ndim: int,
        sites_before_link: bool = True,
        sum_over_staples: bool = False,
        normalize_output: bool = True,
        **time_embed_kwargs
    ):
        """Initialize the GaugeLinkConv module.

        Parameters
        ----------
        in_channels: int | None
            Number of input channels. If None, a singleton channel is added.
        out_channels: int | None
            Number of output channels. If None, a singleton channel is removed.
        spatial_ndim: int
            Number of spatial dimensions of the lattice.
        sites_before_link: bool, default=True
            Whether spatial lattice axes come before the link axis.
        sum_over_staples: bool, default=False
            Whether to sum over all staples instead of keeping them separate.
        normalize_output: bool, default=True
            Whether to normalize the output to have Frobenius norm sqrt(n_c).
        **time_embed_kwargs:
            Additional options to pass to `TimeEmbeddedWeight`.
        """
        super().__init__()

        self.true_in_channels = in_channels
        self.true_out_channels = out_channels
        self.in_channels = 1 if in_channels is None else in_channels
        self.out_channels = 1 if out_channels is None else out_channels
        self.normalize_output = normalize_output

        self.wilson_staple_linear = TimeEmbeddedStapleLayer(
            self.in_channels,
            4 * self.out_channels,
            spatial_ndim,
            sites_before_link,
            sum_over_staples,
            **time_embed_kwargs
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
           x (torch.Tensor): Tensor containing the gauge links.

        Returns:
            torch.Tensor: Updated gauge field tensor with the same shape as x.
        """
        # Add a channel dimension if input channels are None
        if self.true_in_channels is None:
            x = x.unsqueeze(1)

        staples = self.wilson_staple_linear(t, x)
        # Note that the number of channels in staples is twice of out_channels

        s_1, s_2, s_3, s_4 = torch.tensor_split(staples, 4, dim=1)

        if self.in_channels > 1:
            x = x.mean(dim=1, keepdim=True)

        trace_plaqs = torch.einsum('...ii->...', x @ s_3 - (x @ s_4).adjoint())

        x = x + trace_plaqs[..., None, None] / 3 * x

        # Covariant update of the link
        if self.normalize_output:
            x = normalize_matrix(x + s_1.adjoint() - x @ s_2 @ x)
        else:
            x = x + s_1.adjoint() - x @ s_2 @ x

        # Remove the added channel dimension if necessary
        if self.true_out_channels is None:
            x = x.squeeze(1)

        return x


# =============================================================================
class TimeEmbeddedStapleLayer(torch.nn.Module):
    """
    Computes Wilson staples from gauge links and mixes them using a
    time-dependent linear map.

    Note:
        By default, lattice-site axes are assumed to come before the link axis
        (`sites_before_link=True`). Set to False if the link axis comes first.

        Inputs `in_channels` and/or `out_channels` may be None; in that case
        a singleton channel is added or removed automatically.

        Cannot be used for U(1). If needed, use appropriate 'compute_staples'.
    """
    def __init__(
        self,
        in_channels: int | None,
        out_channels: int | None,
        spatial_ndim: int,
        sites_before_link: bool = True,
        sum_over_staples: bool = False,
        **time_embed_kwargs
    ):
        """Initialize the GaugeLinkConv module.

        Parameters
        ----------
        in_channels: int | None
            Number of input channels. If None, a singleton channel is added.
        out_channels: int | None
            Number of output channels. If None, a singleton channel is removed.
        spatial_ndim : int
            Number of spatial lattice dimensions.
        sites_before_link : bool, default=True
            Whether spatial lattice axes come before the link axis.
        sum_over_staples: bool, default=False
            Whether to sum over all staples instead of keeping them separate.
        **time_embed_kwargs:
            Additional options to pass to `TimeEmbeddedWeight`.
        """
        super().__init__()

        self.spatial_ndim = spatial_ndim
        self.sites_before_link = sites_before_link
        self.sum_over_staples = sum_over_staples

        # Remember user-specified channels
        self.true_in_channels = in_channels
        self.true_out_channels = out_channels

        # Actual channel dimensions used in computation
        self.in_channels = 1 if in_channels is None else in_channels
        self.out_channels = 1 if out_channels is None else out_channels

        # Number of staples per link: 2 staples for each transverse direction
        if self.sum_over_staples:
            num_staples = 1
        else:
            num_staples = 2 * (spatial_ndim - 1)

        # Learnable time-dependent weight tensor
        weight_shape = (self.out_channels, self.in_channels * num_staples)
        self.weight_fn = TimeEmbeddedWeight(
            weight_shape=weight_shape,
            **time_embed_kwargs
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Apply the time-dependent linear staple map to gauge links.

        Args:
           x (torch.Tensor): Tensor containing the gauge links.

        Returns:
            torch.Tensor: Computed link-like tensor with the same shape as x.
        """
        # Insert channel dimension if needed
        if self.true_in_channels is None:
            x = x.unsqueeze(1)

        # Compute staples
        staples = compute_staples(
            x,
            prefix_dims=2,
            sites_before_link=self.sites_before_link,
            sum_over_staples=self.sum_over_staples,
        )

        if not self.sum_over_staples:
            # Flatten staple and channel dims
            # staples: (B, C_in, n_staples, ...) -> (B, C_in*n_staples, ...)
            staples = staples.flatten(start_dim=1, end_dim=2)

        # Linear map: (B, F, ...) -> (B_out, F_out, ...)
        weight = self.weight_fn(t) + 0j
        staples = torch.einsum('bi...,boi->bo...', staples, weight)

        # Remove channel if out_channels=None
        if self.true_out_channels is None:
            staples = staples.squeeze(1)

        return staples


# =============================================================================
def normalize_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize matrices by their Frobenius norm scaled by sqrt(n_c) and averaged
    over channels.

    For unitary matrices this scale is one, so the output is unchanged.
    """
    n_c = x.shape[-1]
    norm = torch.linalg.matrix_norm(x, keepdim=True) / n_c ** 0.5
    norm = torch.mean(norm, dim=1, keepdim=True)
    norm = norm.clamp_min(1e-12)  # avoid division by accidental zero
    return x / norm


# =============================================================================
def _test_gauge_equivaraince():
    """Shows the gauge equivariance of the transformation in GaugeLinkConv."""

    saved_dtype = torch.get_default_dtype()
    # pylint: disable=import-outside-toplevel
    from normflow.prior import SUnPrior
    torch.set_default_dtype(saved_dtype)  # importing normflow may change dtype

    shape = (2, 2, 2, 2, 4)  # 2^4 lattice; the last axis is the "mu" axis.
    prior = SUnPrior(3, shape=shape)

    # Define `x` and transform it with instances of GaugeLinkConv
    gauge_link_conv1 = GaugeLinkConv(None, 5, spatial_ndim=4)
    gauge_link_conv2 = GaugeLinkConv(5, None, spatial_ndim=4)
    x = prior.sample(2)
    y = gauge_link_conv2(gauge_link_conv1(x))

    # Now gauge transform `x`; only the links connected to the origin
    q = prior.sample(1)[0, 0, 0, 0, 0, 0]
    for i in range(4):
        x[0, 0, 0, 0, 0, i] = q @ x[0, 0, 0, 0, 0, i]
    x[0, -1, 0, 0, 0, 0] = x[0, -1, 0, 0, 0, 0] @ q.adjoint()
    x[0, 0, -1, 0, 0, 1] = x[0, 0, -1, 0, 0, 1] @ q.adjoint()
    x[0, 0, 0, -1, 0, 2] = x[0, 0, 0, -1, 0, 2] @ q.adjoint()
    x[0, 0, 0, 0, -1, 3] = x[0, 0, 0, 0, -1, 3] @ q.adjoint()

    # Use the gauge transformed x & transform it w/ instances of GaugeLinkConv
    z = gauge_link_conv2(gauge_link_conv1(x))

    # Undo the gauge transformation on `z` to check the gauge equivarience.
    for i in range(4):
        z[0, 0, 0, 0, 0, i] = q.adjoint() @ z[0, 0, 0, 0, 0, i]
    z[0, -1, 0, 0, 0, 0] = z[0, -1, 0, 0, 0, 0] @ q
    z[0, 0, -1, 0, 0, 1] = z[0, 0, -1, 0, 0, 1] @ q
    z[0, 0, 0, -1, 0, 2] = z[0, 0, 0, -1, 0, 2] @ q
    z[0, 0, 0, 0, -1, 3] = z[0, 0, 0, 0, -1, 3] @ q

    print(f"Gauge Equivariant if {(z - y).abs().mean()} is approximately 0")
