# Copyright (c) 2025-2026 Javad Komijani

"""
Wilson line convolutional layers for lattice gauge theory.

This module provides gauge-equivariant layers that update lattice gauge links
using short Wilson lines starting at the tail of a link and ending at its head.
"""

# pylint: disable=too-many-arguments, too-many-positional-arguments

import torch

from .wilson_staples import compute_staples
from .time_embedding import TimeEmbeddedWeight


__all__ = [
    "TimeConditionedGaugeLinkConv",
    "GaugeLinkConv",  # alias -> TimeConditionedGaugeLinkConv
]


# =============================================================================
class TimeConditionedGaugeLinkConv(torch.nn.Module):
    """Gauge-equivariant link convolution layer for lattice gauge fields.

    This class updates gauge links using gauge-covariant linear combinations of
    Wilson staples. Only nearby links are mixed, similar to a convolution in
    standard ML layers. The linear weights are time-dependent, allowing dynamic
    evolution of the gauge links.

    The name reflects that it is:
        - Gauge-covariant (preserves link transformations)
        - Operates on links (not plaquettes or other lattice objects)
        - Convolution-like (local aggregation over nearby staples)
        - Time-conditioned (weights depend on time via a learned embedding)

    The output is not unitary, but is scaled so its Frobenius norm equals
    sqrt(n_c), as in unitary matrices.

    Discussion:
        The output takes the form `normalize((I + A) @ U)` for some matrix A.

        With `restrict_to_algebra=True`, A is guaranteed to lie in the algebra
        by construction, `A = Pi_alg[U @ Gamma]`: the traceless-anti-Hermitian
        projection of U times a single shared staple map Gamma.

        With `restrict_to_algebra=False`, four independently weighted staple
        maps s_1, s_2, s_3, s_4 are learned instead of one shared Gamma, and
        A is built from them without being restricted to the algebra, giving
        more flexibility at the cost of a 4x larger staple-mixing weight.

        - `legacy=True` uses the formula of Algorithm 1 of arXiv:2605.06134.
        - `legacy=False` uses a modified formula that reduces exactly to the
          `restrict_to_algebra=True` formula above when
          `s_1 = s_2 = s_3 = s_4 = -Gamma / 2`, giving it a clean
          interpretation as "restrict_to_algebra, but with independently
          learned, unconstrained staple maps per role."

        Additionally, the weights are complex when `restrict_to_algebra=False`
        and `legacy=False`.

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
        sum_over_staples: bool = True,
        normalize_output: bool = True,
        restrict_to_algebra: bool = False,
        legacy: bool = False,
        time_emb_dim: int | None = None,
        **time_embed_kwargs
    ):
        """Initialize the TimeConditionedGaugeLinkConv module.

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
        sum_over_staples: bool, default=True
            Whether to sum over all staples instead of keeping them separate.
        normalize_output: bool, default=True
            Whether to normalize the output to have Frobenius norm sqrt(n_c).
        restrict_to_algebra: bool, default=False
            If True, the update is normalize((I + A) @ U) with A
            restricted to the Lie algebra (anti-Hermitian), and `legacy`
            has no effect. If False, A is an unconstrained matrix, using
            four-times channel count.
        legacy: bool, default=False
            Only meaningful when restrict_to_algebra=False; ignored
            otherwise. See the class docstring for details.
        time_emb_dim: in | None, default=None
            If given, the input `t` is treated as an already-embedded global
            time embedding of size `time_emb_dim`, and to be projected via
            a single `nn.Linear`.
        **time_embed_kwargs:
            Additional options to pass to `TimeEmbeddedWeight`.
            Ignored if `time_emb_dim` is given.
        """
        super().__init__()

        self.true_in_channels = in_channels
        self.true_out_channels = out_channels
        self.in_channels = 1 if in_channels is None else in_channels
        self.out_channels = 1 if out_channels is None else out_channels
        self.normalize_output = normalize_output
        self.restrict_to_algebra = restrict_to_algebra
        self.legacy = legacy

        self.wilson_staple_linear = TimeConditionedStapleLayer(
            self.in_channels,
            self.out_channels * (1 if restrict_to_algebra else 4),
            spatial_ndim,
            sites_before_link,
            sum_over_staples,
            complex_weights=(not legacy and not restrict_to_algebra),
            time_emb_dim=time_emb_dim,
            **time_embed_kwargs
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
           t (torch.Tensor): Raw time, or an already-embedded global time
               embedding if `time_emb_dim` was given at construction.
           x (torch.Tensor): Tensor containing the gauge links.

        Returns:
            torch.Tensor: Updated gauge field tensor with the same shape as x.
        """
        # Add a channel dimension if input channels are None
        if self.true_in_channels is None:
            x = x.unsqueeze(1)

        staples = self.wilson_staple_linear(t, x)

        if self.in_channels > 1:
            x = x.mean(dim=1, keepdim=True)

        if self.restrict_to_algebra:
            corrections = anti_hermitian_traceless(x @ staples)
            x = x + corrections @ x
        else:
            s_1, s_2, s_3, s_4 = torch.tensor_split(staples, 4, dim=1)
            trace = torch.einsum('...ii->...', x @ s_3 - (x @ s_4).adjoint())
            trace = trace[..., None, None]
            n_c = x.shape[-1]
            if self.legacy:
                x = (1 + trace / n_c) * x
                x = x + s_1.adjoint() - x @ s_2 @ x
            else:
                x = (1 + trace / n_c) * x + ((x @ s_1).adjoint() - x @ s_2) @ x

        if self.normalize_output:
            x = normalize_matrix(x)

        # Remove the added channel dimension if necessary
        if self.true_out_channels is None:
            x = x.squeeze(1)

        return x


# =============================================================================
class TimeConditionedStapleLayer(torch.nn.Module):
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
        sum_over_staples: bool = True,
        complex_weights: bool = False,
        time_emb_dim: int | None = None,
        **time_embed_kwargs
    ):
        """Initialize the TimeConditionedStapleLayer module.

        Parameters
        ----------
        in_channels: int | None
            Number of input channels. If None, a singleton channel is added.
        out_channels: int | None
            Number of output channels. If None, a singleton channel is removed.
        spatial_ndim: int
            Number of spatial lattice dimensions.
        sites_before_link: bool, default=True
            Whether spatial lattice axes come before the link axis.
        sum_over_staples: bool, default=True
            Whether to sum over all staples instead of keeping them separate.
        complex_weights: bool, default=False
            If True, the staple-mixing weights have independently learned
            real and imaginary parts.
        time_emb_dim: in | None, default=None
            If given, the input `t` is treated as an already-embedded global
            time embedding of size `time_emb_dim`, and to be projected via
            a single `nn.Linear`.
        **time_embed_kwargs:
            Additional options to pass to `TimeEmbeddedWeight`.
            Ignored if `time_emb_dim` is given.
        """
        super().__init__()

        self.spatial_ndim = spatial_ndim
        self.sites_before_link = sites_before_link
        self.sum_over_staples = sum_over_staples
        self.complex_weights = complex_weights

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

        # Learnable time-dependent weight tensor; an extra trailing axis of
        # size 2 holds independent real & imaginary parts if complex_weights.
        weight_shape = (self.out_channels, self.in_channels * num_staples)
        if complex_weights:
            weight_shape = (*weight_shape, 2)

        if time_emb_dim is None:
            self.weight_fn = TimeEmbeddedWeight(
                weight_shape=weight_shape, **time_embed_kwargs
            )
        else:
            self.weight_fn = _LinearWeight(time_emb_dim, weight_shape)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Apply the time-dependent linear staple map to gauge links.

        Args:
           t (torch.Tensor): Raw time, or an already-embedded global time
               embedding if `time_emb_dim` was given at construction.
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
        weight = self.weight_fn(t)
        if self.complex_weights:
            weight = weight[..., 0] + 1j * weight[..., 1]
        else:
            weight = weight + 0j
        staples = torch.einsum('bi...,boi->bo...', staples, weight)

        # Remove channel if out_channels=None
        if self.true_out_channels is None:
            staples = staples.squeeze(1)

        return staples


# =============================================================================
class _LinearWeight(torch.nn.Module):
    """Projects a precomputed embedding to a weight tensor via one `Linear`."""

    def __init__(self, emb_dim: int, weight_shape: tuple[int, ...]):
        super().__init__()

        self.weight_shape = weight_shape
        n_weight = int(torch.tensor(weight_shape).prod())
        self.linear = torch.nn.Linear(emb_dim, n_weight)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """Compute the weight tensor for a given embedding.

        Args:
            emb (torch.Tensor): Embedding of shape `(*batch, emb_dim)`.

        Returns:
            Tensor: Weight tensor of shape `(*batch, *self.weight_shape)`.
        """
        batch_shape = emb.shape[:-1]
        return self.linear(emb).reshape(*batch_shape, *self.weight_shape)


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
def anti_hermitian_traceless(x: torch.Tensor) -> torch.Tensor:
    """
    Project the input onto the space of traceless anti-Hermitian matrices.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor with square matrices in the last two dimensions.

    Returns
    -------
    torch.Tensor
        Tensor of the same shape as `x`, where each matrix is projected to be
        anti-Hermitian and traceless.
    """
    # Anti-Hermitian part
    x = (x - x.adjoint()) / 2

    # Remove trace
    trace = torch.einsum("...ii->...", x)[..., None, None]
    n = x.shape[-1]
    eye = torch.eye(n, device=x.device, dtype=x.dtype)

    return x - (trace / n) * eye


# =============================================================================
# Keep for legacy
GaugeLinkConv = TimeConditionedGaugeLinkConv


# =============================================================================
def _test_gauge_equivaraince():
    """Shows the gauge equivariance of the transformation in GaugeLinkConv."""

    # pylint: disable=import-outside-toplevel
    from normflow.prior import SUnPrior

    t = torch.rand(1)

    shape = (2, 2, 2, 2, 4)  # 2^4 lattice; the last axis is the "mu" axis.
    prior = SUnPrior(3, shape=shape)

    # Define `x` and transform it with instances of GaugeLinkConv
    gauge_link_conv1 = GaugeLinkConv(None, 5, spatial_ndim=4)
    gauge_link_conv2 = GaugeLinkConv(5, None, spatial_ndim=4)
    x = prior.sample(2)
    y = gauge_link_conv2(t, gauge_link_conv1(t, x))

    # Now gauge transform `x`; only the links connected to the origin
    q = prior.sample(1)[0, 0, 0, 0, 0, 0]
    for i in range(4):
        x[0, 0, 0, 0, 0, i] = q @ x[0, 0, 0, 0, 0, i]
    x[0, -1, 0, 0, 0, 0] = x[0, -1, 0, 0, 0, 0] @ q.adjoint()
    x[0, 0, -1, 0, 0, 1] = x[0, 0, -1, 0, 0, 1] @ q.adjoint()
    x[0, 0, 0, -1, 0, 2] = x[0, 0, 0, -1, 0, 2] @ q.adjoint()
    x[0, 0, 0, 0, -1, 3] = x[0, 0, 0, 0, -1, 3] @ q.adjoint()

    # Use the gauge transformed x & transform it w/ instances of GaugeLinkConv
    z = gauge_link_conv2(t, gauge_link_conv1(t, x))

    # Undo the gauge transformation on `z` to check the gauge equivarience.
    for i in range(4):
        z[0, 0, 0, 0, 0, i] = q.adjoint() @ z[0, 0, 0, 0, 0, i]
    z[0, -1, 0, 0, 0, 0] = z[0, -1, 0, 0, 0, 0] @ q
    z[0, 0, -1, 0, 0, 1] = z[0, 0, -1, 0, 0, 1] @ q
    z[0, 0, 0, -1, 0, 2] = z[0, 0, 0, -1, 0, 2] @ q
    z[0, 0, 0, 0, -1, 3] = z[0, 0, 0, 0, -1, 3] @ q

    print(f"Gauge Equivariant if {(z - y).abs().mean()} is approximately 0")
