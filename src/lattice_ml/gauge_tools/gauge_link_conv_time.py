# Copyright (c) 2025 Javad Komijani

"""
Wilson line convolutional layers for lattice gauge theory.

This module provides gauge-equivariant layers that update lattice gauge links
using short Wilson lines starting at the tail of a link and ending at its head.
"""

from typing import List
import torch

#from .wilson_staples import compute_planar_staples
from lattice_ml.gauge_tools.wilson_staples import compute_planar_staples


from torch.nn import Module


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
        ndim: int,
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
        ndim : int
            Number of spacetime dimensions of the lattice.
        sites_before_link : bool, default=True
            Whether spatial lattice axes come before the link axis.
        """
        super().__init__()
        self.ndim = ndim

        self.true_in_channels = in_channels
        self.true_out_channels = out_channels
        self.in_channels = 1 if in_channels is None else in_channels
        self.out_channels = 1 if out_channels is None else out_channels

        self.sites_before_link = sites_before_link
        self._link_axis = -3 if sites_before_link else 2  # 2: batch & channel

        # Learnable weight tensor for each valid (mu, nu) pair
        """
        shape = (ndim, 2*(ndim-1), self.out_channels, self.in_channels, 2)

        self.weight = torch.nn.Parameter(torch.randn(*shape) * scale)
        """

        self.poly_order = 5  # e.g., degree-3 polynomial
        shape = (self.poly_order + 1, ndim, 2*(ndim-1), self.out_channels, self.in_channels, 2)
        scale = 0.0 #it was 0.0001 works the best sofar
        self.weight_poly = torch.nn.Parameter(torch.rand(*shape) * scale)
    
    def time_dependent_weight(self, t: torch.Tensor | float) -> torch.Tensor:
        """
        Evaluate the time-dependent weight tensor
            W(t) = \sum_{k=0}^{K} A_k \, t^{k}
        where each coefficient A_k has shape (D, 2*(D-1), C_out, C_in, 2)
        and the last axis packs (real, imag) parts for later conversion to complex.

        Parameters
        ----------
        t : float or Tensor
            Time value(s) at which to evaluate W(t).
            - If scalar (float or 0-D tensor), returns a shared weight for the whole batch.
        -    If shape (B,), returns per-sample weights.

        Returns
        -------
        Tensor
            Real-packed polynomial evaluation of W(t).
            Shapes:
            - If t is scalar: (D, 2*(D-1), C_out, C_in, 2)
            - If t is (B,):  (B, D, 2*(D-1), C_out, C_in, 2)
        """
    
        # Shapes:
        # self.weight_poly: (K+1, ndim, 2*(ndim-1), C_out, C_in, 2)
        Kp1, *_ = self.weight_poly.shape

        # Make t a tensor on the same device/dtype
        t = torch.as_tensor(t, device=self.weight_poly.device, dtype=self.weight_poly.dtype)

        powers = torch.arange(Kp1, device=self.weight_poly.device, dtype=self.weight_poly.dtype)

        if t.ndim == 0:
            # ----- scalar t -----
            # t_pows: (K+1,) -> (K+1,1,1,1,1,1) to align with weight_poly dim0
            t_pows = (t**powers).view(Kp1, 1, 1, 1, 1, 1)
            # Elementwise multiply, then sum over polynomial dim (dim=0)
            weight_t = (self.weight_poly * t_pows).sum(dim=0)
            # -> (ndim, 2*(ndim-1), C_out, C_in, 2)
            return weight_t

        elif t.ndim == 1:
            # ----- per-batch t of shape (B,) -----
            # Build (B, K+1) table of powers
            t_pows = t.unsqueeze(1)**powers.unsqueeze(0)  # (B, K+1)
            # Contract polynomial dim: (B, K+1) x (K+1, ...) -> (B, ...)
            weight_t = torch.einsum('bk,k...->b...', t_pows, self.weight_poly)
            # -> (B, ndim, 2*(ndim-1), C_out, C_in, 2)
            return weight_t

        else:
            raise ValueError("t must be scalar () or 1D (B,).")



    def forward(self, x: torch.Tensor, t: torch.Tensor | float = 0.0) -> torch.Tensor:
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
        ndim = self.ndim
        link_axis = self._link_axis

        x_unbound = torch.unbind(x, dim=link_axis)

        output_stack: List[torch.Tensor] = [None] * ndim

        weight_t = self.time_dependent_weight(t)
        per_sample = (weight_t.ndim == 6)  # (B, ndim, 2*(ndim-1), C_out, C_in, 2)
        # scalar-t case is 5D:          (ndim, 2*(ndim-1), C_out, C_in, 2)  

        # Loop over lattice directions
        for mu in range(ndim):
            x_mu = x_unbound[mu]

            shape = (x_mu.shape[0], 2 * self.out_channels, *x_mu.shape[2:])
            staples = torch.zeros(shape, dtype=x.dtype, device=x.device)

            ind = 0  # index for weight slices (ind+=2 for each valid nu != mu)
            for nu in range(ndim):
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
                #w = self.weight[mu, ind:ind+2].reshape(-1, self.in_channels, 2)
                #w = weight_t[mu, ind:ind+2].reshape(-1, self.in_channels, 2)

                #w = torch.view_as_complex(w)

                if per_sample:
                    # per-sample weights
                    w_realimag = weight_t[:, mu, ind:ind+2]                       # (B, 2, C_out, C_in, 2)
                    w = w_realimag.reshape(w_realimag.shape[0], -1, self.in_channels, 2)  # (B, 2*C_out, C_in, 2)
                    w = torch.view_as_complex(w)                                   # (B, 2*C_out, C_in)
                else:
                    # shared weights
                    w_realimag = weight_t[mu, ind:ind+2]            # (2, C_out, C_in, 2)
                    w = w_realimag.reshape(-1, self.in_channels, 2) # (2*C_out, C_in, 2)
                    w = torch.view_as_complex(w)                    # (2*C_out, C_in)                       # (B, 2*C_out, C_in)

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
        
        #x_updated = naive_project_su3(x_updated)
        
        return x_updated

    def set_param2zero(self):
        """Set all trainable parameters to zero."""
        torch.nn.init.zeros_(self.weight_poly)

    def set_param2normal(self, mean: float = 0.0, std: float = 1.0):
        """Set all trainable parameters to Gaussian with given mean and std."""
        torch.nn.init.normal_(self.weight_poly, mean=mean, std=std)


# =============================================================================
def einsum_(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
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

def einsum(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    x: (B, C_in, ...)
    w: (C_out, C_in)          # shared weights
       or (B, C_out, C_in)    # per-sample weights
    returns: (B, C_out, ...)
    """
    B, C_in = x.shape[:2]
    tail = x.shape[2:]
    x_flat = x.reshape(B, C_in, -1)  # (B, C_in, N)

    if w.ndim == 2:
        # shared
        out = torch.einsum('bcn,oc->bon', x_flat, w)       # (B, C_out, N)
    else:
        # per-sample
        out = torch.einsum('bcn,boc->bon', x_flat, w)      # (B, C_out, N)

    return out.view(B, -1, *tail)


# =============================================================================
def naive_project_su3(y):
    """
    Naively projects a 3x3 complex matrix to SU(3) by orthonormalizing rows.

    This method assumes the input matrix is close to the identity. It first
    orthonormalizes the first two rows, then reconstructs the third row to
    enforce unitarity and determinant = 1.

    Notes:
    1. Although not necessary, the matrix is initially normalized to ensure
       determinnat 1.
    2. The changes are not in-place because PyTorch cannot handle
       backpropagation of derivatives (if the adjointstate method is not used).
    """
    # Normalize matrix to ensure determinant is 1 (special unitary)
    # Explicit calculation of determinant is faster than torch.linalg.det!
    y_00, y_01, y_02 = torch.unbind(y[..., 0, :], dim=-1)
    y_10, y_11, y_12 = torch.unbind(y[..., 1, :], dim=-1)
    y_20, y_21, y_22 = torch.unbind(y[..., 2, :], dim=-1)
    det = (
        y_20 * (y_01 * y_12 - y_02 * y_11)
        + y_21 * (y_02 * y_10 - y_00 * y_12)
        + y_22 * (y_00 * y_11 - y_01 * y_10)
    )

    y = y / det[..., None, None]**(1/3.)

    # Unbind rows for further calculations
    y_0, y_1, _ = torch.unbind(y, dim=-2)

    # Normalize the first row
    norm_sq = torch.sum(y_0.conj() * y_0, dim=-1, keepdim=True)
    y_0 = y_0 / torch.sqrt(norm_sq)

    # Compute inner product of first two rows
    vdot = torch.sum(y_0.conj() * y_1, dim=-1, keepdim=True)
    # Orthogonalize second row against the first
    y_1 = y_1 - y_0 * vdot

    # Normalize the second row
    norm_sq = torch.sum(y_1 * y_1.conj(), dim=-1, keepdim=True)
    y_1 = y_1 / torch.sqrt(norm_sq)

    # Reconstruct third row as complex conjugate of cross product of first two
    y_2 = torch.stack(
        ((y_0[..., 1] * y_1[..., 2] - y_0[..., 2] * y_1[..., 1]).conj(),
         (y_0[..., 2] * y_1[..., 0] - y_0[..., 0] * y_1[..., 2]).conj(),
         (y_0[..., 0] * y_1[..., 1] - y_0[..., 1] * y_1[..., 0]).conj()
        ),
        dim = -1
    )

    y = torch.stack((y_0, y_1, y_2), dim=-2)

    return y

# =============================================================================
def _test_gauge_equivaraince():
    """Shows the gauge equivariance of the transformation in GaugeLinkConv."""

    import normflow  # pylint: disable=import-outside-toplevel
    shape = (2, 2, 2, 2, 4)  # 2^4 lattice; the last axis is the "mu" axis.
    prior = normflow.prior.SUnPrior(3, shape=shape)

    # Define `x` and transform it with instances of GaugeLinkConv
    gauge_link_conv1 = GaugeLinkConv(None, 5, ndim=4)
    gauge_link_conv2 = GaugeLinkConv(5, None, ndim=4)
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

_test_gauge_equivaraince()