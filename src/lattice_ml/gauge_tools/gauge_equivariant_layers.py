# Javad Komijani, Feb 2026

"""
Lattice Gauge Equivariant Convolution layer for SU(N) lattice fields.

Mainly implements `M. Favoni et al., arXiv:2012.12901`.

The layer acts on a tuple (U, W) where:
- U : gauge link field (SU(N) matrices on lattice links)
- W : Lie-algebra

The convolution is gauge-equivariant and works in arbitrary spatial dimension.
"""

# pylint: disable=invalid-name


from typing import Tuple
import torch

from .wilson_loops import compute_wilson_1x1_loop


__all__ = [
    'Conv',
    'Bilinear',
    'TraceConditionedNet',
    'ColorSingletonShift',
    'Normalize',
    'StateInitializer'
]


# =============================================================================
class Conv(torch.nn.Module):
    """Lattice Gauge Equivariant Convolution [Eq. (5), arXiv:2012.12901]."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_ndim: int,
        kernel_size: int = 3,
        sites_before_link: bool = True,
    ):
        """Initialize the Conv module.

        Parameters
        ----------
        in_channels: int
            Number of input channels.
        out_channels: int
            Number of output channels.
        spatial_ndim: int
            Number of spatial dimensions of the lattice.
        kernel_size: int, default=3
            The kernel size, which currently only 3 is supported.
        sites_before_link: bool, default=True
            Whether spatial lattice axes come before the link axis.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_ndim = spatial_ndim
        self.kernel_size = kernel_size
        self.sites_before_link = sites_before_link

        if kernel_size != 3:
            raise NotImplementedError("Only kernel_size = 3 is supported.")

        def make_parameter(shape):
            weight = (torch.randn(shape) + 1j * torch.randn(shape)) / 10
            return torch.nn.Parameter(weight)

        # Channel mixing weights (in → out)
        channels = (out_channels, in_channels)
        self.weight_0 = make_parameter(channels)
        self.weight_plus = make_parameter((*channels, spatial_ndim))
        self.weight_minus = make_parameter((*channels, spatial_ndim))

    def forward(self, state: Tuple[torch.Tensor, torch.Tensor]):
        """
        Parameters
        ----------
        state : (U, W)
            U : gauge links (B, ..., D, N, N) or (B, D, ..., N, N)
            W : gauge loops (B, C, ..., N, N)

        Returns
        -------
        (U, W_out) : Tuple
            Gauge links unchanged, updated gauge loops.
        """
        U, W = state

        U = U.unsqueeze(1)  # add a singleton channel

        link_axis = -3 if self.sites_before_link else 2
        links = torch.unbind(U, dim=link_axis)

        # Center term
        W_out = torch.einsum("oc,bc...ij->bo...ij", self.weight_0, W)

        for mu in range(self.spatial_ndim):
            U_mu = links[mu]

            # Forward neighbour
            W_transport = U_mu @ shift_lattice(W, -1, 2 + mu) @ U_mu.adjoint()
            weight = self.weight_plus[..., mu]
            W_out += torch.einsum("oc,bc...ij->bo...ij", weight, W_transport)

            # Backward neighbour (your correct version)
            W_transport = shift_lattice(U_mu.adjoint() @ W @ U_mu, 1, 2 + mu)
            weight = self.weight_minus[..., mu]
            W_out += torch.einsum("oc,bc...ij->bo...ij", weight, W_transport)

        U = U.squeeze(1)  # remove a singleton channel

        return U, W_out


# =============================================================================
class Bilinear(torch.nn.Module):
    """Local gauge-equivariant bilinear layer [Eq. (6), arXiv:2012.12901]."""

    def __init__(
        self,
        in_channels1: int,
        in_channels2: int,
        out_channels: int,
    ):
        super().__init__()
        self.in_channels1 = in_channels1
        self.in_channels2 = in_channels2
        self.out_channels = out_channels

        # Learnable bilinear tensor α[o, a, b]
        channels = (out_channels, in_channels1, in_channels2)
        weight = (torch.randn(channels) + 1j * torch.randn(channels)) / 10
        self.weight = torch.nn.Parameter(weight)

    def forward(
        self,
        state1: Tuple[torch.Tensor, torch.Tensor],
        state2: Tuple[torch.Tensor, torch.Tensor]
    ):
        """
        Parameters
        ----------
        state1 : (U, W1)
        state2 : (U, W2)
            U : gauge links (B, ..., D, N, N) or (B, D, ..., N, N)
            W1 : gauge loops (B, C1, ..., N, N)
            W2 : gauge loops (B, C2, ..., N, N)

        Returns
        -------
        (U, W_out) : Tuple
            Gauge links unchanged, updated gauge loops.
        """
        U, W1 = state1
        U, W2 = state2

        # Expand channels for broadcasting
        # W1_exp: (B, C1, 1, ..., N, N)
        # W2_exp: (B, 1, C2, ..., N, N)
        W1_exp = W1.unsqueeze(2)
        W2_exp = W2.unsqueeze(1)

        # Matrix multiplication in color space (last two axes)
        # Result: (B, C1, C2, ..., N, N)
        W_out = torch.matmul(W1_exp, W2_exp)

        # Mix channels with α[o, j, k]
        # weight[o,j,k] x W_out[b,j,k,...,i,j] -> sum_{j,k} -> out[o,...,i,j]
        W_out = torch.einsum("ojk,bjk...mn->bo...mn", self.weight, W_out)

        return U, W_out


# =============================================================================
class TraceConditionedNet(torch.nn.Module):
    """
    Gauge-equivariant transformation conditioned on the trace invariant.

    This module extracts the trace of the Wilson-like field `W`, producing a
    gauge-invariant complex scalar at each batch/channel/lattice location.
    The scalar is represented as a real tensor with the innermost dimension
    of size 2 (real and imaginary parts). This tensor is processed by `net`,
    which can be any scalar map (e.g., neural network, linear layer, or
    elementwise activation). The output is then interpreted as a complex
    tensor again.

    The resulting complex scalar field is lifted back to color space via the
    identity matrix, yielding a gauge-equivariant update of `W`. The gauge
    field `U` is left unchanged.

    No external conditioning is required — the transformation depends solely
    on gauge-invariant information extracted from `W`.

    Notes
    -----
    - Gauge equivariance is preserved because the trace is gauge invariant and
      the update is proportional to the identity in color space.
    - `net` operates on real representations of the trace scalars.
    """

    def __init__(self, net: torch.nn.Module):
        super().__init__()
        self.net = net

    def forward(self, state: Tuple[torch.Tensor, torch.Tensor]):
        """
        Parameters
        ----------
        state : (U, W)
            U : gauge links (B, ..., D, N, N) or (B, D, ..., N, N)
            W : gauge loops (B, C, ..., N, N)

        Returns
        -------
        (U, W_out) : Tuple
            Gauge links unchanged, updated gauge loops.
        """
        U, W = state

        # Compute the trace and apply the transformatios
        trace = torch.einsum('...ii->...', W)
        shift = torch.view_as_complex(self.net(torch.view_as_real(trace)))

        # Identity in color space
        n_c = W.shape[-1]
        eye = torch.eye(n_c, dtype=W.dtype, device=W.device)

        # Apply shift
        W_out = W + shift[..., None, None] * eye

        return U, W_out


# =============================================================================
class ColorSingletonShift(torch.nn.Module):
    """
    Gauge-equivariant identity-matrix shift applied to the color space.

    Performs the linear map:
        W = W + s · I

    where `s` is a tensor broadcastable to the outer dimensions of `W` (all
    axes except the last two color/matrix axes), and `I` is the identity in
    color space. Singleton dimensions are automatically added if necessary.


    Gauge equivariance is preserved because the update is proportional to the
    identity matrix. The gauge field `U` is unchanged.
    """

    def forward(self, state, shift):
        """
        Parameters
        ----------
        state : Tuple[Tensor, Tensor]
            U : gauge links (B, ..., D, N, N) or (B, D, ..., N, N)
            W : gauge loops (B, C, ..., N, N)

        shift : Tensor
            Tensor broadcastable to the outer dimensions of W. Singleton
            dimensions will be automatically added to match remaining axes.

        Returns
        -------
        (U, W_out) : Tuple
            Gauge links unchanged, updated gauge loops.
        """
        U, W = state

        # Add singleton dimensions to match W if needed
        if shift.ndim < W.ndim - 2:
            expand_dims = W.ndim - shift.ndim
            shift = shift.view(*shift.shape, *([1] * expand_dims))
        elif shift.ndim > W.ndim - 2:
            raise ValueError("shift.ndim is not consistent")

        # Identity in color space
        n_c = W.shape[-1]
        eye = torch.eye(n_c, dtype=W.dtype, device=W.device)

        # Apply shift
        W_out = W + shift * eye

        return U, W_out


# =============================================================================
class Normalize(torch.nn.Module):
    """
    Normalize matrices by their Frobenius norm scaled by sqrt(n_c) and averaged
    over channels.

    For unitary matrices this scale is one, so the output is unchanged.
    """

    def forward(self, state: Tuple[torch.Tensor, torch.Tensor]):
        """
        Parameters
        ----------
        state : (U, W)
            U : gauge links (B, ..., D, N, N) or (B, D, ..., N, N)
            W : gauge loops (B, C, ..., N, N)

        Returns
        -------
        (U, W_out) : Tuple
            Gauge links unchanged, updated gauge loops.
        """
        U, W = state
        n_c = W.shape[-1]
        norm = torch.linalg.matrix_norm(W, keepdim=True) / n_c ** 0.5
        norm = torch.mean(norm, dim=1, keepdim=True)
        norm = norm.clamp_min(1e-12)  # avoid division by accidental zero
        return U, W / norm


# =============================================================================
class StateInitializer(torch.nn.Module):
    """
    Initialize the lattice gauge equivariant network state from gauge links.

    This layer is the entry point of an LGE-CNN. It maps gauge links U to the
    network state (U, W), where W consists of local gauge-covariant features
    constructed from 1×1 Wilson plaquettes at each lattice site.

    The gauge links U are passed through unchanged, while W is created by
    stacking all oriented plaquettes as a channel dimension.

    Parameters
    ----------
    sites_before_link : bool, default=True
        Whether spatial lattice axes appear before the link-direction axis
        in the input tensor layout.

    Input
    -----
    U : torch.Tensor
        Gauge link tensor with shape
        (B, ..., D, Nc, Nc) if sites_before_link=True, or
        (B, D, ..., Nc, Nc) otherwise.

    Output
    ------
    state : Tuple[torch.Tensor, torch.Tensor]
        (U, W) where
        - U : unchanged gauge links
        - W : plaquette features stacked along a new channel axis,
              shape (B, C, ..., Nc, Nc) with C = D*(D-1)
    """

    def __init__(self, sites_before_link: bool = True):
        super().__init__()
        self.sites_before_link = sites_before_link

    def forward(self, U: torch.Tensor):
        """
        Parameters
        ----------
        U : gauge links (B, ..., D, N, N) or (B, D, ..., N, N)

        Returns
        -------
        (U, W) : Tuple
            U : gauge links (B, ..., D, N, N) or (B, D, ..., N, N)
            W : torch.Tensor
                Tensor of 1×1 Wilson loops. All planar loops associated to a
                site are stacked along a new channel axis as (B, C, ..., N, N).
        """
        W = compute_wilson_1x1_loop(
            U, sites_before_link=self.sites_before_link
        )
        return U, W


# =============================================================================
def shift_lattice(x, step, dim):
    """Periodic shift along a lattice dimension."""
    return torch.roll(x, shifts=step, dims=dim)


# =============================================================================
def _test_gauge_equivaraince():
    """Shows the gauge equivariance of the transformation in GaugeLinkConv."""

    from normflow.prior import SUnPrior

    shape = (2, 2, 2, 2, 4)  # 2^4 lattice; the last axis is the "mu" axis.
    prior = SUnPrior(3, shape=shape)

    # Define `x` and transform it with instances of GaugeLinkConv
    initializer = StateInitializer()
    conv1 = Conv(12, 12, spatial_ndim=4)
    conv2 = Conv(12, 16, spatial_ndim=4)
    bilin = Bilinear(12, 16, 4)
    x = prior.sample(2)

    state = initializer(x)
    state = conv1(state)
    state = bilin(state, conv2(state))

    w = state[1]

    # Now gauge transform `x`; only the links connected to the origin
    q = prior.sample(1)[0, 0, 0, 0, 0, 0]
    for i in range(4):
        x[0, 0, 0, 0, 0, i] = q @ x[0, 0, 0, 0, 0, i]
    x[0, -1, 0, 0, 0, 0] = x[0, -1, 0, 0, 0, 0] @ q.adjoint()
    x[0, 0, -1, 0, 0, 1] = x[0, 0, -1, 0, 0, 1] @ q.adjoint()
    x[0, 0, 0, -1, 0, 2] = x[0, 0, 0, -1, 0, 2] @ q.adjoint()
    x[0, 0, 0, 0, -1, 3] = x[0, 0, 0, 0, -1, 3] @ q.adjoint()

    # Use the gauge transformed x & transform it w/ instances of GaugeLinkConv
    state = initializer(x)
    state = conv1(state)
    state = bilin(state, conv2(state))

    z = state[1]

    # Undo the gauge transformation on `z` to check the gauge equivarience.
    q = q[None, None, :, :]
    z[0, :, 0, 0, 0, 0] = q.adjoint() @ z[0, :, 0, 0, 0, 0] @ q

    print(f"Gauge Equivariant if {(z - w).abs().mean()} is approximately 0")
