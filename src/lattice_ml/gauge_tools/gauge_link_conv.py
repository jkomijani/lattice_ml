# Copyright (c) 2025 Javad Komijani

"""
Wilson line convolutional layers for lattice gauge theory.

This module provides gauge-equivariant layers that update lattice gauge links
using short Wilson lines starting at the tail of a link and ending at its head.
"""

from typing import List
import torch

from .wilson_staples import compute_planar_staples


__all__ = ["GaugeLinkConv"]

matmul = torch.matmul


# =============================================================================
class GaugeLinkConv(torch.nn.Module):
    """
    Gauge-equivariant link convolution layer for lattice gauge fields.

    GaugeLinkConv updates gauge links using gauge-covariant combinations of
    short Wilson lines (paths along links) starting at the tail of each link
    and ending at the head. The layer maintains gauge covariance by combining
    contributions from allowed paths containing the staples and of length <= 5,
    weighted by learnable parameters.

    Unlike the original gauge link variables, the output of this layer is not
    necessarily unitary, but it transforms under gauge transformations in the
    same way as the original link variables.

    Note:
        Tensors are expected by default to have spatial lattice axes before
        the link direction axis (sites_before_link=True). Set to False if your
        tensor uses link axis before lattice sites.

        It is possible to set the input and ouput channels to None. If set to
        None, a singleton channel axis is automatically added to inputs before
        processing and/or removed afterwards. This allows layers to operate in
        both channel-free and channel-based architectures.
    """

    def __init__(
        self,
        in_channels: int | None,
        out_channels: int | None,
        spatial_ndim: int,
        sites_before_link: bool = True
    ):
        """
        Initialize the GaugeLinkConv module.

        Parameters
        ----------
        in_channels : int | None
            Number of input feature channels per link. If None, a singleton
            channel axis is automatically added to the input.
        out_channels : int | None
            Number of output feature channels per link. If None, a singleton
            channel axis of output is automatically removed.
        spatial_ndim : int
            Number of spatial dimensions of the lattice.
        sites_before_link : bool, default=True
            Whether spatial lattice axes come before the link axis.
        """
        super().__init__()
        self.spatial_ndim = spatial_ndim

        self.true_in_channels = in_channels
        self.true_out_channels = out_channels
        self.in_channels = 1 if in_channels is None else in_channels
        self.out_channels = 1 if out_channels is None else out_channels

        self.sites_before_link = sites_before_link
        self._link_axis = -3 if sites_before_link else 2  # 2: batch & channel

        # Learnable weight tensor for each valid (mu, nu) pair
        ndim = spatial_ndim
        shape = (ndim, 2*(ndim-1), self.out_channels, self.in_channels, 2)
        scale = 0.01
        self.weight = torch.nn.Parameter(torch.randn(*shape) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tensor containing the gauge links.

        Returns
        -------
        torch.Tensor
            Updated gauge field tensor with the same shape as `x`.
        """
        # Add a channel dimension if input channels are None
        if self.true_in_channels is None:
            x = x.unsqueeze(1)

        # Allocate output tensor
        spatial_ndim = self.spatial_ndim
        link_axis = self._link_axis

        x_unbound = torch.unbind(x, dim=link_axis)

        output_stack: List[torch.Tensor] = [None] * spatial_ndim

        # Loop over lattice directions
        for mu in range(spatial_ndim):
            x_mu = x_unbound[mu]

            shape = (x_mu.shape[0], 2 * self.out_channels, *x_mu.shape[2:])
            staples = torch.zeros(shape, dtype=x.dtype, device=x.device)

            ind = 0  # index for weight slices (ind+=2 for each valid nu != mu)
            for nu in range(spatial_ndim):
                if nu == mu:
                    continue

                # Compute planar staples (upper + lower) along mu-nu plane
                planar_staples = compute_planar_staples(
                    x, mu, nu,
                    prefix_dims=2,
                    sites_before_link=self.sites_before_link,
                    return_sum=True
                )

                # Learnable weights for this mu-nu pair, reshape, & complexify
                w = self.weight[mu, ind:ind+2].reshape(-1, self.in_channels, 2)
                w = torch.view_as_complex(w)

                # Sum contributions from weighted staples
                staples += einsum(planar_staples, w)
                ind += 2

            # Average over input channels for this link direction if needed
            if self.in_channels > 1:
                x_mu = x_mu.mean(dim=1, keepdim=True)

            s_1, s_2 = torch.tensor_split(staples, 2, dim=1)

            # Covariant update of the link
            output_stack[mu] = x_mu + s_1.adjoint() + x_mu @ s_2 @ x_mu

        x_updated = torch.stack(output_stack, dim=link_axis)

        # Remove the added channel dimension if necessary
        if self.true_out_channels is None:
            x_updated = x_updated.squeeze(1)

        return x_updated

    def set_param2zero(self):
        """Set all trainable parameters to zero."""
        torch.nn.init.zeros_(self.weight)

    def set_param2normal(self, mean: float = 0.0, std: float = 1.0):
        """Set all trainable parameters to Gaussian with given mean and std."""
        torch.nn.init.normal_(self.weight, mean=mean, std=std)


# =============================================================================
def einsum(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Applies a linear transformation over the channel dimension using einsum.

    Parameters:
        x (Tensor): Input tensor of shape (B, C_in, *), where * represents any
            number of spatial or additional dimensions.
        w (Tensor): Weight tensor of shape (C_out, C_in), representing the
            linear mapping from input to output channels.

    Returns:
        Tensor: Output tensor of shape (B, C_out, *), with the same shape as
            x but with C_out channels instead of C_in.
    """
    bsize, c_in = x.shape[:2]
    out = torch.einsum('bcn,oc->bon', x.reshape(bsize, c_in, -1), w)
    return out.view(bsize, -1, *x.shape[2:])


# =============================================================================
def _test_gauge_equivaraince():
    """Shows the gauge equivariance of the transformation in GaugeLinkConv."""

    import normflow  # pylint: disable=import-outside-toplevel
    shape = (2, 2, 2, 2, 4)  # 2^4 lattice; the last axis is the "mu" axis.
    prior = normflow.prior.SUnPrior(3, shape=shape)

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
