# Copyright (c) 2025 Javad Komijani & Lara Turgut

"""
UNet was introduced by Ronneberger et al. (2015) for image segmentation,
particularly in biomedical imaging. It features an encoder-decoder structure
with skip connections to retain spatial information. Later, UNet was adapted
for diffusion models, where it was modified to include time-step embeddings,
self-attention layers, and conditioning mechanisms (e.g. using cross-attention)
for text-to-image generation.

The term **time-step embedding** was first used in the Denoising Diffusion
Probabilistic Models (DDPM) paper (Ho et al., 2020) to condition the model on
the current timestep during the denoising process.

In this model, the time dependence is implemented differently:
each GaugeLinkConv block includes a learnable time-dependent coefficient.
Simply adding an external time encoder, as done in standard diffusion UNets,
would not in general preserve gauge covariance. Hence, the time dependence is
embedded directly within the gauge-equivariant layers.

References:
- Ronneberger, O., Fischer, P., and Brox, T. (2015). U-Net: Convolutional
  Networks for Biomedical Image Segmentation, [arXiv:1505.04597].
- Ho, J., Jain, A., and Abbeel, P. (2020). Denoising Diffusion Probabilistic
  Models, [arXiv:2006.11239].
"""

import copy
import torch
from typing import Tuple

from scipy.linalg import sqrtm

from torch.nn import Module
from torch.nn import ModuleList

from lattice_ml.gauge_tools import GaugeLinkConv
from lattice_ml.functions import pow_special_unitary_group
# from lattice_ml.functions import project_onto_special_unitary
from lattice_ml.functions import project_onto_unitary


__all__ = [
    "UNet",
    "UNetEncoderLayer",
    "UNetDecoderLayer",
    "UNetBottleneck"
]

matmul = torch.matmul


def make_unet(spatial_ndim=4):
    """An example showing how to make an instance of UNet."""

    kwargs = {"spatial_ndim": spatial_ndim}

    # Encoding (contractive) path
    encoder_layers = [
        UNetEncoderLayer(channels=(1, 1), **kwargs),
    ]

    # bottleneck
    bottleneck = UNetBottleneck(channels=(1, 1), **kwargs)

    # Decoding (expansive) path
    decoder_layers = [
        UNetDecoderLayer(channels=(1, 1), **kwargs),
    ]

    return UNet(encoder_layers, bottleneck, decoder_layers)


def gauge_downsampler(
    x: torch.Tensor,
    prefix_dims: int = 1,
    sites_before_link: bool = True,
) -> torch.Tensor:
    """
    Downsample gauge links by a factor of 2 along all lattice axes,
    i.e. construct a coarse lattice Λ_even from the fine lattice Λ.

    Assumes:
    - sites_before_link=True:  x.shape = (*prefix, L1,...,Ld, D, Nc, Nc)
    - sites_before_link=False: x.shape = (*prefix, D, L1,...,Ld, Nc, Nc)

    For each direction μ, retain only links whose tails lie on even lattice
    sites and construct coarse μ-links by multiplying adjacent fine μ-links:
        U_coarseμ(x_even) = U_fineμ(x_even) @ U_fineμ(x_even + μ),
        where x_even ∈ Λ_even denotes a single even lattice site.

    Returns a tensor with the same axis order as the input, but with each
    lattice extent Lk halved.

    """
    if sites_before_link:
        link_axis_ = -3
        spatial_start = prefix_dims
        spatial_end = x.ndim - 3
        d = spatial_end - spatial_start  # actually D = d but I keep
        # them separate in case we downsample only in some directions.
        # After removing link axis, stack back at boundary between sites and
        # matrices
        stack_dim = prefix_dims + d
        def roll_dim(mu): return prefix_dims + mu
    else:
        link_axis_ = prefix_dims
        spatial_start = prefix_dims + 1
        spatial_end = x.ndim - 2
        d = spatial_end - spatial_start
        stack_dim = prefix_dims  # insert link axis right after *prefix
        def roll_dim(mu): return prefix_dims + 1 + mu

    # Unbind the link-direction axis: list of tensors, one per direction μ
    links = torch.unbind(x, dim=link_axis_)  # length D

    # Build an index that selects even sites (stride 2) on all axes.
    sample_link = links[0]
    even_idx = [slice(None)] * sample_link.ndim
    for ax in range(spatial_start, spatial_start + d):
        even_idx[ax] = slice(0, None, 2)
    even_idx = tuple(even_idx)

    coarse_links = []
    for mu, u in enumerate(links):
        # u has shape (*prefix, L1,...,Ld, Nc, Nc)
        u_even = u[even_idx]
        u_shift = torch.roll(u, shifts=-1, dims=roll_dim(mu))
        u_shift_even = u_shift[even_idx]
        u_coarse = matmul(u_even, u_shift_even)
        coarse_links.append(u_coarse)

    # Stack coarse directions back into a link axis at the proper place
    x_coarse = torch.stack(coarse_links, dim=stack_dim)
    return x_coarse


def inv_via_solve(H: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    *batch, n, _ = H.shape
    eye = torch.eye(n, dtype=H.dtype, device=H.device).expand(*batch, n, n)
    H = H + eps * eye
    return torch.linalg.solve(H, eye)


def torch_matrix_sqrt(Q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # Q: (..., N, N), complex
    w, V = torch.linalg.eig(Q)                  # w: (..., N), V: (..., N, N)
    w_sqrt = torch.sqrt(w)                      # principal branch
    Vinv = torch.linalg.inv(V)
    return V @ torch.diag_embed(w_sqrt) @ Vinv


def project_onto_special_unitary(matrix, eps: float = 1e-12):
    dtype0 = matrix.dtype
    q = project_onto_unitary(matrix.to(torch.complex128))

    detq = torch.linalg.det(q)
    # Avoid NaNs if detq is weird
    detq = detq / detq.abs().clamp_min(eps)

    rdet_angle = torch.angle(detq) / q.shape[-1]
    phase_factor = torch.exp(-1j * rdet_angle)[..., None, None]
    return (q * phase_factor).to(dtype0)


def gauge_upsampler(
    # fine lattice BEFORE transform (contains a,b,...)
    x_fine_pre: torch.Tensor,
    # coarse lattice AFTER transform (contains A')
    x_coarse_post: torch.Tensor,
    prefix_dims: int = 1,
    sites_before_link: bool = True,
    set_id: bool = True,
    sun: bool = True,
) -> torch.Tensor:
    """
    Upsample transformed coarse links A′ back to fine links a′ and b′.
    For each direction μ, construct fine μ-links as:
    a′ = A′ @ b⁻¹
    b′ = a⁻¹ @ A′
    where a and b are the fine links before transformation, and A′ is the
    coarse link after transformation.
    """
    def chk(name: str, T: torch.Tensor):
        if torch.isnan(T).any() or torch.isinf(T).any():
            raise RuntimeError(f"{name} has NaN/Inf")

    if sites_before_link:
        link_axis_ = -3
        spatial_start = prefix_dims
        spatial_end = x_fine_pre.ndim - 3
        d = spatial_end - spatial_start
        def roll_dim(mu): return prefix_dims + mu
    else:
        link_axis_ = prefix_dims
        spatial_start = prefix_dims + 1
        spatial_end = x_fine_pre.ndim - 2
        d = spatial_end - spatial_start
        def roll_dim(mu): return prefix_dims + 1 + mu

    fine_links_pre = list(torch.unbind(x_fine_pre, dim=link_axis_))
    coarse_links_post = list(torch.unbind(x_coarse_post, dim=link_axis_))

    # even-site index (stride 2) across spatial dims
    sample = fine_links_pre[0]
    even_idx = [slice(None)] * sample.ndim
    for ax in range(spatial_start, spatial_start + d):
        even_idx[ax] = slice(0, None, 2)
    even_idx = tuple(even_idx)

    def odd_idx_for_mu(mu):
        odd = list(even_idx)
        ax = roll_dim(mu)
        odd[ax] = slice(1, None, 2)
        return tuple(odd)

    fine_links_post = [u.clone() for u in fine_links_pre]

    for mu in range(len(fine_links_pre)):
        u_pre = fine_links_pre[mu]      # (*prefix, L..., Nc, Nc)
        A_post = coarse_links_post[mu]   # (*prefix, L/2..., Nc, Nc)
        chk("A_post_raw", A_post)

        # Extract a and b from fine PRE lattice
        a = u_pre[even_idx]
        u_shift = torch.roll(u_pre, shifts=-1, dims=roll_dim(mu))
        b = u_shift[even_idx]
        chk("a_raw", a)
        chk("b_raw", b)

        A_pre = matmul(a, b)
        chk("A_pre_raw", A_pre)

        # set_id = True
        # sun = True

        if not set_id:
            if sun:
                Q = matmul(
                    matmul(
                        matmul(
                            matmul(a.adjoint(), A_pre),
                            A_post.adjoint()
                        ),
                        A_pre
                    ),
                    b.adjoint()
                )
                chk("Q_sun_raw", Q)

                Q = project_onto_special_unitary(Q)
                chk("Q_sun_proj", Q)
                Qc = Q.to(torch.complex128)
                Sq = pow_special_unitary_group(Qc, 0.5)
                Sq = Sq.to(Q.dtype)
                chk("Sq_sun", Sq)
            else:
                bb_dag = matmul(b, b.adjoint())
                chk("bb_dag", bb_dag)

                denom = matmul(
                    matmul(
                        a.adjoint(), A_post), matmul(
                        A_post.adjoint(), a))
                chk("denom", denom)

                bb_dag_inv = inv_via_solve(bb_dag)
                chk("bb_dag_inv", bb_dag_inv)

                denom_inv = inv_via_solve(denom)
                chk("denom_inv", denom_inv)

                Q = matmul(
                    matmul(
                        bb_dag_inv,
                        matmul(matmul(b, A_post.adjoint()), a)
                    ),
                    denom_inv
                )

                Sq = torch_matrix_sqrt(Q)
                chk("Sq", Sq)
        else:
            Nc = A_pre.size(-1)
            Sq = torch.eye(Nc, dtype=A_pre.dtype, device=A_pre.device).expand(
                A_pre.shape[:-2] + (Nc, Nc)
            )
            chk("Sq_identity", Sq)

        a_post = matmul(matmul(A_post, b.adjoint()), Sq)
        chk("a_post", a_post)

        b_post = matmul(matmul(Sq, a.adjoint()), A_post)
        chk("b_post", b_post)

        fine_links_post[mu][even_idx] = a_post
        fine_links_post[mu][odd_idx_for_mu(mu)] = b_post

    x_fine_post = torch.stack(fine_links_post, dim=link_axis_)
    chk("x_fine_post", x_fine_post)
    return x_fine_post

# =============================================================================


class UNet(Module):
    """A gauge-equivariant U-Net model.

    A U-Net model consists of an encoder, a bottleneck, and a decoder.
    The encoder extracts hierarchical features, the bottleneck processes them
    at the lowest resolution, and the decoder reconstructs the output while
    integrating skip connections. The module can handle conditional variables
    too if provided.

    Args:
        encoder_layers (List[UNetEncoderLayer]): A list of encoder layers.
        bottleneck (Callable): A layer between encoder and decoder.
        decoder_layers (List[UNetDecoderLayer]): A list of decoder layers.

    Note: It is the user's responsibility to ensure that the shapes of inputs
    and outputs of all layers match.
    """

    def __init__(self, encoder_layers, bottleneck, decoder_layers):

        super().__init__()
        self.encoder = UNetEncoder(encoder_layers)
        self.bottleneck = bottleneck
        self.decoder = UNetDecoder(decoder_layers)

    def forward(self, *args, **cond):
        """
        Perform the forward pass of the U-Net model.

        Support two calling conventions:

        1. Standard usage:
            forward(data, **cond)
            - `data` is the input tensor.
            - Pass additional conditional variables as keyword arguments.

        2. Alternate usage:
            forward(t, data, **cond)
            - Interpret the first positional argument `t` as `time` condition.
            - Internally handle this as forward(data, time=t, **cond).
            - `t` must have either:
                * `ndim == 1`: a vector of shape `(batch_size,)` giving
                  a per-sample time value.
                * `ndim == 0`: a scalar, which will be broadcast to
                  shape `(batch_size,)`, so all samples share the same time.

        Args:
            data (Tensor): Input tensor when using standard usage. When using
                alternate usage, this is the second positional argument.
            **cond: Optional keyword arguments containing conditional variables
                that influence the output, such as `time` in diffusion models.

        Returns:
            Tensor: Output tensor after processing through the encoder,
            bottleneck, and decoder.
        """
        # Handle (t, data) case
        if len(args) == 2:
            t, data = args
            cond["time"] = t if t.ndim > 0 else t.repeat(data.shape[0])
        elif len(args) == 1:
            data = args[0]
        else:
            raise ValueError("forward() accepts only 1 or 2 postional args.")

        data = data.unsqueeze(1)
        data, skips = self.encoder(data, **cond)
        data = self.bottleneck(data, **cond)
        data = self.decoder(data, skips, **cond)
        data = data.squeeze(1)

        return data

    def enlarge_architecture(
        self,
        encoder_layer,
        decoder_layer,
        inplace: bool = False,
        **bottleneck_kwargs
    ):
        """
        Enlarges the UNet architecture by adding layers to the encoder and
        decoder. The encoder appends the new layer to the end of its layers
        list, while the decoder prepends it.

        Also adjusts the bottleneck to match the new channel structure,
        based on the provided keyword arguments.

        Args:
            encoder_layer: Module to add at the end of the encoder.
            decoder_layer: Module to insert at the beginning of the decoder.
            inplace (bool): If True, modifies the current UNet instance.
                            If False, makes a copy and returns a modified copy.
            **bottleneck_kwargs: Arguments to update bottleneck channels,
                                 e.g., in_channels or out_channels.

        Returns:
            UNet: Modified UNet with expanded architecture.
        """
        unet = self if inplace else copy.deepcopy(self)

        unet.encoder.enlarge_architecture(encoder_layer)
        unet.decoder.enlarge_architecture(decoder_layer)
        # Note: The bottleneck does not add new layers but adapts its channel
        # dimensions to maintain compatibility with the updated encoder and
        # decoder. This is done via adjust_channels(**bottleneck_kwargs).
        unet.bottleneck.adjust_channels(**bottleneck_kwargs)

        return unet


class UNetEncoder(Module):
    """An encoder module for the U-Net model.

    This encoder consists of a list of `UNetEncoderLayer` instances provided at
    the time of instantiation. It sequentially processes the input tensor,
    applying each layer in order. Intermediate feature maps are stored for skip
    connections, which help retain spatial information in the UNet decoder.

    Args:
        encoder_layers: A list of `UNetEncoderLayer` instances.
    """

    def __init__(self, encoder_layers):

        super().__init__()
        self.layers = ModuleList(encoder_layers)

    def forward(self, data, **cond):
        """Performs the forward pass of the UNet encoder.

        Args:
            data (Tensor): The input tensor.
            cond (dict, optional): A dictionary of conditional variables that
                influences the output, such as time in diffusion models.

        Returns:
            Tuple:
                - Tensor: The result after processing through the encoder.
                - List[Tensor]: The intermediate results for skip connections.
        """
        skips = []  # # Stores intermediate results for skip connections

        for layer in self.layers:
            data, skip = layer(data, **cond)
            skips.append(skip)

        return data, skips  # Return final output and skip connections

    def enlarge_architecture(self, encoder_layer):
        """
        Adds a new layer to the end of the encoder's architecture.

        This method enlarges the encoder by appending the provided layer to its
        internal list of layers. The added layer should be compatible with the
        output of the existing final encoder layer.

        Args:
            encoder_layer: A layer/module to append to the encoder.
        """
        self.layers.append(encoder_layer)


class UNetDecoder(Module):
    """A decoder module for the UNet model.

    This encoder consists of a list pf `UNetDecoderLayer` instances provided at
    the time of instantiation. It sequentially processes the input tensor,
    applying each layer in order. It progressively upsamples the encoded
    feature maps and integrates skip connections from the encoder to restore
    spatial details.

    Args:
        decoder_layers: A list of `UNetDecoderLayer` instances.
    """

    def __init__(self, decoder_layers):

        super().__init__()
        self.layers = ModuleList(decoder_layers)

    def forward(self, data, skips, **cond):
        """Performs the forward pass of the UNet decoder.

        Args:
            data (Tensor): The input tensor, typically the final output of
                the encoder.
            skips (List[Tensor]): A list of skip connection tensors from the
                encoder.
            cond (dict, optional): A dictionary of conditional variables that
                influences the output, such as time in diffusion models.

        Returns:
            Tensor: The reconstructed output tensor after processing through
            the decoder.
        """
        assert len(self.layers) == len(skips), "mismatch in number of layers"

        for layer, skip in zip(self.layers, reversed(skips)):
            data = layer(data, skip, **cond)

        return data

    def enlarge_architecture(self, decoder_layer):
        """
        Adds a new layer to the beginning of the decoder's architecture.

        This method enlarges the decoder by inserting the provided layer at
        the beginning of its internal list of layers. The added layer should be
        compatible with the output of the bottleneck or the previous decoder
        input.

        Args:
            decoder_layer: A layer/module to insert into the decoder.
        """
        self.layers.insert(0, decoder_layer)


# =============================================================================
class UNetEncoderLayer(Module):
    """
    A single layer of the UNet encoder network. It processes input through
    multiple stages to extract feature representations. This layer consists
    of the following components:

    1. **Block 1**:
        - Uses GaugeLinkConv

    2. **Block 2**:
        - Uses GaugeLinkConv

    3. **Down-Sampling Block**:
        - Gauge downsampler with stride 2.

    Args:
        - channels (Tuple[int]): A tuple of integers specifying the number of
          input and output channels for the convolutions.
    """

    def __init__(
        self,
        channels: Tuple[int, int] = (None, None),
        bot_num_blocks: int = 3,
        **conv_kwargs
    ):
        super().__init__()

        self.bot_num_blocks = bot_num_blocks

        self.block1 = GaugeLinkConv(
            in_channels=channels[0],
            out_channels=channels[1],
            **conv_kwargs,
        )

        self.block2 = GaugeLinkConv(
            in_channels=channels[1],
            out_channels=channels[1],
            **conv_kwargs,
        )

        if bot_num_blocks == 4:
            self.block3 = GaugeLinkConv(
                in_channels=channels[1],
                out_channels=channels[1],
                **conv_kwargs,
            )

            self.block4 = GaugeLinkConv(
                in_channels=channels[1],
                out_channels=channels[1],
                **conv_kwargs,
            )

        # self.gate = torch.nn.Parameter(torch.zeros(()))

    def forward(self, data, time=None):
        """
        Forward pass through the encoder layer.

        Args:
            data (Tensor): The input tensor (e.g., image or features).
            time (Tensor): A 1D tensor representing time steps (batch axis).

        Returns:
            Tensor: The result after processing through the down-sampler.
            Tensor: The intermediate result before the down-sampler.
        """
        # h = self.block1(time, data)          # first block: always on
        # data = data + self.gate * self.block2(time, h)  # second block: gated
        # y = gauge_downsampler(data, prefix_dims=2, sites_before_link=True)

        data = self.block1(time, data)
        data = self.block2(time, data)
        # data = self.block3(time, data)
        y = gauge_downsampler(data, prefix_dims=2, sites_before_link=True)

        return y, data


# =============================================================================
class UNetDecoderLayer(Module):
    """
    A single layer of the UNet decoder network. It processes input through
    multiple stages to reconstruct spatial dimensions. This layer consists of:

    1. **Up-Sampling Block**:
        - An upsampling block layer with scale factor of 2, increasing the
          spatial size of the input data.
        - This block enables the network to reconstruct multi-scale features.

    2. **Block 1**:
        - Uses GaugeLinkSmear

    3. **Block 2**:
        - Uses GaugeLinkSmear

    Remarks:
    - **Up-sampling Block**:
        - The up-sampling stride helps to recover the spatial dimensions lost
          during down-sampling in the encoder.

    Args:
        - Same as in `UNetEncoderLayer`, but typically requires matching
          output channels for up-sampling and reconstruction.
    """

    def __init__(
        self,
        channels: Tuple[int, int] = (None, None),
        set_id: bool = True,
        sun: bool = True,
        **conv_kwargs
    ):
        super().__init__()

        self.block1 = GaugeLinkConv(
            in_channels=channels[0],
            out_channels=channels[0],
            **conv_kwargs,
        )

        self.block2 = GaugeLinkConv(
            in_channels=channels[0],
            out_channels=channels[1],
            **conv_kwargs,
        )

        # self.gate = torch.nn.Parameter(torch.zeros(()))
        self.set_id = set_id
        self.sun = sun

    def forward(self, data, skip_connection, time=None):
        """
        Forward pass through the decoder layer.

        Args:
            data (Tensor): The input tensor.
            skip_connection (Tensor): Skip connection from the encoder.
            time (Tensor): A 1D tensor representing time steps (batch axis).

        Returns:
            Tensor: The output tensor after processing through all sub-layers.
        """
        # Upsample `data` and concatenate the ouput with `skip_connection`

        data = gauge_upsampler(
            skip_connection,
            data,
            prefix_dims=2,
            sites_before_link=True,
            set_id=self.set_id,
            sun=self.sun)

        data = self.block1(time, data)
        data = self.block2(time, data)
        # data = self.block3(time, data)

        # h = data + self.gate * self.block1(time, data)
        # data = self.block2(time, h)

        return data

# =============================================================================


class UNetBottleneck(UNetEncoderLayer):
    """
    A bottleneck layer for UNet, consisting of:

    1. **Block 1**:
        - A GaugeSmearLink

    2. **Block 2**:
        - A GaugeSmearLink

    This class is a subclass of `UNetEncoderLayer`, but does not include a
    downsampler. It explicitly sets `downsampler = None` to reflect this
    behavior.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, data, time=None):
        """
        Forward pass through the encoder layer.

        Args:
            data (Tensor): The input tensor.
            time (Tensor): A 1D tensor representing time steps (batch axis).

        Returns:
            Tensor: The result after processing through layers.
        """
        data = self.block1(time, data)
        data = self.block2(time, data)

        if self.bot_num_blocks == 4:
            data = self.block3(time, data)
            data = self.block4(time, data)

        # h = self.block1(time, data)          # first block: always on
        # data = data + self.gate * self.block2(time, h)   # second block:
        # gated

        return data
