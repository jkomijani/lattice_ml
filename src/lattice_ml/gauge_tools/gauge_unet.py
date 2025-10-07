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

References:
- Ronneberger, O., Fischer, P., and Brox, T. (2015). U-Net: Convolutional
  Networks for Biomedical Image Segmentation, [arXiv:1505.04597].
- Ho, J., Jain, A., and Abbeel, P. (2020). Denoising Diffusion Probabilistic
  Models, [arXiv:2006.11239].

See Appendix C of Ho et al. for their network and Appendix D for an interesting
discussion about the time steps that capture features like gender.
"""


import copy
import torch
from typing import Tuple

from torch.nn import Module
from torch.nn import ModuleList

from lattice_ml.gauge_tools import GaugeLinkConv


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
    Downsample gauge links by factor 2 along all lattice axes.
    Assumes:
      - sites_before_link=True:  x.shape = (*prefix, L1,...,Ld, D, Nc, Nc)
      - sites_before_link=False: x.shape = (*prefix, D, L1,...,Ld, Nc, Nc)

    For each direction μ, keeps only links whose tails lie on even sites and
    constructs coarse links by multiplying adjacent fine links along μ:
        U_coarseμ(x_even) = U_fineμ(x_even) @ U_fineμ(x_even + μ)
    Returns a tensor with the same axis order as the input but each Lk halved.
    """
    if sites_before_link:
        # axes: (*prefix, L1...Ld, D, Nc, Nc)
        link_axis_in_x = -3
        spatial_start = prefix_dims
        spatial_end   = x.ndim - 3  # up to (but not incl.) link axis
        d = spatial_end - spatial_start # actually D = d but I keep them separate in case we downsample only in some directions.
        # After removing link axis, stack back at boundary between sites and matrices
        stack_dim = prefix_dims + d
        # dims used for torch.roll
        def roll_dim(mu): return prefix_dims + mu
    else:
        # axes: (*prefix, D, L1...Ld, Nc, Nc)
        link_axis_in_x = prefix_dims
        spatial_start = prefix_dims + 1
        spatial_end   = x.ndim - 2
        d = spatial_end - spatial_start
        stack_dim = prefix_dims  # insert link axis right after *prefix
        def roll_dim(mu): return prefix_dims + 1 + mu

    # Unbind the link-direction axis: list of tensors, one per direction μ
    links = torch.unbind(x, dim=link_axis_in_x)  # length D

    # Build an index that selects even sites (stride 2) on all lattice axes
    # for the per-μ tensors (which have the link axis removed).
    sample_link = links[0]
    even_idx = [slice(None)] * sample_link.ndim
    for ax in range(spatial_start, spatial_start + d):
        even_idx[ax] = slice(0, None, 2)
    even_idx = tuple(even_idx)

    coarse_links = []
    for mu, u in enumerate(links):
        # u has shape (*prefix, L1,...,Ld, Nc, Nc) regardless of sites_before_link
        u_even = u[even_idx]
        u_shift = torch.roll(u, shifts=-1, dims=roll_dim(mu))
        u_shift_even = u_shift[even_idx]
        u_coarse = matmul(u_even, u_shift_even)
        coarse_links.append(u_coarse)

    # Stack coarse directions back into a link axis at the proper place
    x_coarse = torch.stack(coarse_links, dim=stack_dim)
    return x_coarse

def _polar_unitary(X: torch.Tensor) -> torch.Tensor:
    """
    Unitary factor of the polar decomposition via SVD:
      X = U Σ Vᴴ  =>  polar_unitary(X) = U Vᴴ
    Works batched and on CUDA. Preserves gradients.
    """
    # For real X you still want complex-safe det adjustment later; keep dtype as is here.
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)  # (..., n, n), (..., n), (..., n, n)
    Uh = Vh.conj().transpose(-1, -2)
    return U @ Uh  # (..., n, n)

def _project_to_suN(U: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Renormalize a unitary-ish matrix to SU(N): det -> 1.
    Uses complex dtype for phase handling, then casts back to input dtype.
    """
    n = U.shape[-1]
    # Work in complex for robust phase; upcast if needed
    Uc = U if torch.is_complex(U) else U.to(torch.complex64 if U.dtype==torch.float32 else torch.complex128)
    detU = torch.det(Uc)  # (...,)

    # Avoid division by ~0: push magnitude away from 0
    mag = detU.abs().clamp_min(eps)
    detU_safe = detU / mag

    factor = detU_safe.pow(-1.0 / n).reshape(*detU.shape, 1, 1)
    U_su = (Uc * factor).to(dtype=U.dtype)
    return U_su

def _sqrtm_unitary(Q: torch.Tensor, *, project_back: bool = True, herm_tol: float = 1e-7) -> torch.Tensor:
    """
    Batched matrix square root using eig/eigh, then optional projection to SU(N).
    Shapes: (..., n, n) -> (..., n, n)
    """
    assert Q.shape[-1] == Q.shape[-2], "Q must be square"
    *batch, n, _ = Q.shape

    # Ensure complex for general eig (real inputs can have complex eigenpairs)
    if torch.is_complex(Q):
        Qc = Q
    else:
        Qc = Q.to(torch.complex64 if Q.dtype==torch.float32 else torch.complex128)

    # Heuristic: if nearly Hermitian, prefer eigh
    near_herm = False
    if herm_tol is not None:
        resid = torch.linalg.matrix_norm(Qc - Qc.mH)
        near_herm = (resid.item() < herm_tol) if resid.numel()==1 else False

    if near_herm:
        w, V = torch.linalg.eigh(Qc)                  # (..., n), (..., n, n)
        w_clamped = torch.clamp(w.real, min=0.0).to(w.dtype)
        sqrt_w = torch.sqrt(w_clamped).to(Qc.dtype)   # (..., n)
        Sq = V @ torch.diag_embed(sqrt_w) @ V.mH      # (..., n, n)
    else:
        w, V = torch.linalg.eig(Qc)                   # (..., n), (..., n, n)
        sqrt_w = torch.sqrt(w)                        # principal branch
        Vinv = torch.linalg.inv(V)
        Sq = V @ torch.diag_embed(sqrt_w) @ Vinv

    if project_back:
        # Replace torch.linalg.polar with SVD-based polar
        U_polar = _polar_unitary(Sq)
        U_su = _project_to_suN(U_polar)
        return U_su.to(dtype=Q.dtype)

    return Sq.to(dtype=Q.dtype)


def gauge_upsampler(
    x_fine_pre: torch.Tensor,          # fine lattice BEFORE transform (contains a,b,...)
    x_coarse_post: torch.Tensor,       # coarse lattice AFTER transform (contains A')
    prefix_dims: int = 1,
    sites_before_link: bool = True,
    project_back_to_suN: bool = True,
) -> torch.Tensor:
    """
    Upsample a transformed coarse lattice A' back to fine links (a', b')
    using the original fine lattice before transform (a, b) as 'middle links'.

    For each direction μ and for links whose tails lie on even sites:
        A  = a @ b               (from x_fine_pre)
        Q  = A @ A'^† @ A
        a' = A' @ b^† @ sqrt(Q)
        b' = sqrt(Q) @ a^† @ A'

    All other fine links (not the even-tail pair (a,b) along μ) are copied
    unchanged from x_fine_pre.

    Shapes match gauge_downsampler:
      - sites_before_link=True:  x_fine_pre = (*prefix, L1,...,Ld, D, Nc, Nc)
      - sites_before_link=False: x_fine_pre = (*prefix, D, L1,...,Ld, Nc, Nc)
    The returned tensor has the same shape/order as x_fine_pre.
    """
    x = x_fine_pre
    if sites_before_link:
        # axes: (*prefix, L1...Ld, D, Nc, Nc)
        link_axis_in_x = -3
        spatial_start = prefix_dims
        spatial_end   = x.ndim - 3
        d = spatial_end - spatial_start
        stack_dim = prefix_dims + d
        def roll_dim(mu): return prefix_dims + mu
    else:
        # axes: (*prefix, D, L1...Ld, Nc, Nc)
        link_axis_in_x = prefix_dims
        spatial_start = prefix_dims + 1
        spatial_end   = x.ndim - 2
        d = spatial_end - spatial_start
        stack_dim = prefix_dims
        def roll_dim(mu): return prefix_dims + 1 + mu

    # Unbind directions for the fine PRE lattice and coarse POST lattice
    fine_links_pre = list(torch.unbind(x, dim=link_axis_in_x))       # D tensors of shape (*prefix, L..., Nc, Nc)
    coarse_links_post = list(torch.unbind(x_coarse_post, dim=stack_dim))  # D tensors of shape (*prefix, L/2..., Nc, Nc)

    # Build the even-site index tuple (stride-2) on spatial axes
    sample = fine_links_pre[0]
    even_idx = [slice(None)] * sample.ndim
    for ax in range(spatial_start, spatial_start + d):
        even_idx[ax] = slice(0, None, 2)
    even_idx = tuple(even_idx)

    # Build the companion index for the *second* link (odd tail) along μ:
    # it's identical to even_idx except shifted by +1 on axis roll_dim(mu)
    def odd_idx_for_mu(mu):
        odd = list(even_idx)
        ax = roll_dim(mu)
        # 'even' selects 0,2,4,... ; the second link 'b' sits at those +1 positions -> 1,3,5,...
        odd[ax] = slice(1, None, 2)
        return tuple(odd)

    # Initialize output with a copy of the original fine PRE lattice (adopt middle/other links)
    fine_links_post = [u.clone() for u in fine_links_pre]

    # Do the split for every direction μ
    for mu in range(len(fine_links_pre)):
        u_pre = fine_links_pre[mu]          # (*prefix, L..., Nc, Nc)
        A_post = coarse_links_post[mu]      # (*prefix, L/2..., Nc, Nc)

        # Extract a (even-tail links) and b (the immediate next link along μ) from the fine PRE lattice
        a = u_pre[even_idx]  # (*prefix, L/2..., Nc, Nc)
        u_shift = torch.roll(u_pre, shifts=-1, dims=roll_dim(mu))
        b = u_shift[even_idx]

        # A from fine PRE
        A_pre = matmul(a, b)

        # Q = A A'^† A
        Q = matmul(matmul(A_pre, A_post.adjoint()), A_pre)

        # sqrt(Q) with optional projection back to SU(N)
        Sq = _sqrtm_unitary(Q, project_back=project_back_to_suN)

        # a' = A' b^† sqrt(Q)
        a_post = matmul(matmul(A_post, b.adjoint()), Sq)
        # b' = sqrt(Q) a^† A'
        b_post = matmul(matmul(Sq, a.adjoint()), A_post)

        # Scatter a' back to even-tail positions and b' back to the “+1 along μ” positions
        out_mu = fine_links_post[mu]
        out_mu[even_idx] = a_post
        out_mu[odd_idx_for_mu(mu)] = b_post
        fine_links_post[mu] = out_mu

    # Stack directions back into a link axis at the proper place
    x_fine_post = torch.stack(fine_links_post, dim=link_axis_in_x)
    return x_fine_post


# =============================================================================
class UNet(Module):
    """A UNet model.

    A UNet model consists of an encoder, a bottleneck, and a decoder.
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
        Perform the forward pass of the UNet model.

        Support two calling conventions:

        1. Standard usage:
            forward(data, **cond)
            - `data` is the input tensor (e.g., image or features).
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

        data, skips = self.encoder(data, **cond)
        data = self.bottleneck(data, **cond)
        data = self.decoder(data, skips, **cond)

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
    """An encoder module for the UNet model.

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
            data (Tensor): The input tensor (e.g., image or features).
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
                the encoder with shape `(batch_size, channels, height, width)`.
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

    1. **Convolutional Block 1**:
        - Uses GaugeLinkConv

    2. **Time Embedding Block**:
        - TODO

    3. **Convolutional Block 2**:
        - Optional GaugeLinkConv

    4. **Down-Sampling Block**:
        - Gauge downsampler with stride 2.

    Args:
        - channels (Tuple[int]): A tuple of integers specifying the number of
          input and output channels for the convolutions.
        - spatial_ndim (int, optional): The number of spatial dimensions).
          Defaults to 2.
        - bias , group norm and activation removed!
    """

    def __init__(
        self,
        channels: Tuple[int, int] = (None, None),
        spatial_ndim: int = 2,
        downsampler='gauge',
        time_embedding=True
    ):

        super().__init__()

        self.conv_block1 = GaugeLinkConv(in_channels=channels[0], out_channels=channels[1], ndim= spatial_ndim)

        '''
        if time_embedding:
            self.time_encoder = SinusoidalTimeEncoder(
                channels[1], trainable_freq=True, trainable_ampl=True
            )
        else:
            self.time_encoder = None
        '''

        self.spatial_ndim = spatial_ndim  # to be used in time_encoder.forward

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
    
        data = data.unsqueeze(1)
        data = self.conv_block1(data)


        '''
        if self.time_encoder is not None:
            data += self.time_encoder(time, self.spatial_ndim)
        '''

        return gauge_downsampler(data, prefix_dims=2), data


# =============================================================================
class UNetDecoderLayer(Module):
    """
    A single layer of the UNet decoder network. It processes input through
    multiple stages to reconstruct spatial dimensions. This layer consists of:

    1. **Up-Sampling Block**:
        - An upsampling block layer with scale factor of 2, increasing the
          spatial size of the input data.
        - This block enables the network to reconstruct multi-scale features.

    1. **Convolutional Block 1**:
        - Uses GaugeConvBlock

    2. **Time Embedding Block**:
        - Applies sinusoidal functions to encode the time information and adds
          the result to the output of the first block.
        - The embedding length matches the number of channels.
        - The amplitudes and frequencies of the sinusoidal functions are
          trainable parameters.

    3. **Convolutional Block 2**:
        - Optional

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
        spatial_ndim: int = 2,
        upsampler='gauge',
        time_embedding=True
    ):
        super().__init__()

        # Build the up-sampling block
        '''
        match upsampler:
            case 'Upsample':
                self.upsampler = gauge_upsampler()
            case _:
                self.upsampler = torch.nn.Identity()
        '''

        self.conv_block1 = GaugeLinkConv(in_channels=channels[0], out_channels=channels[1], ndim=spatial_ndim)

        '''
        if time_embedding:
            self.time_encoder = SinusoidalTimeEncoder(
                channels[0], trainable_freq=True, trainable_ampl=True
            )
        else:
            self.time_encoder = None
        '''

        self.spatial_ndim = spatial_ndim  # to be used in time_encoder.forward

    def forward(self, data, skip_connection, time=None):
        """
        Forward pass through the decoder layer.

        Args:
            data (Tensor): The input tensor (e.g., image or features).
            skip_connection (Tensor): Skip connection from the encoder.
            time (Tensor): A 1D tensor representing time steps (batch axis).

        Returns:
            Tensor: The output tensor after processing through all sub-layers.
        """
        # Upsample `data` and concatenate the ouput with `skip_connection`

        data = gauge_upsampler(skip_connection, data, prefix_dims=2)

        data = self.conv_block1(data)

        data = data.squeeze(1)
        '''
        if self.time_encoder is not None:
            data += self.time_encoder(time, self.spatial_ndim),
        '''

        return data


# =============================================================================
class UNetBottleneck(UNetEncoderLayer):
    """
    A bottleneck layer for UNet, consisting of:

    1. **Convolutional Block 1**:
        - A GaugeLinkConv

    2. **Time Embedding Block** (optional):
        - Applies sinusoidal functions to encode the time information and adds
          the result to the output of the first block.
        - The embedding length matches the number of channels.
        - The amplitudes and frequencies of the sinusoidal functions are
          trainable parameters.

    3. **Convolutional Block 2**:
        - Optional

    This class is a subclass of `UNetEncoderLayer`, but does not include a
    downsampler. It explicitly sets `downsampler = None` to reflect this
    behavior.
    """
    def __init__(self, **kwargs):
        super().__init__(downsampler=None, **kwargs)

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
        data = self.conv_block1(data)

        '''
        if self.time_encoder is not None:
            data += self.time_encoder(time, self.spatial_ndim)
        '''

        return data

    def adjust_channels(self, in_channels=None, out_channels=None):
        """
        Adjusts the input and/or output channels of the bottleneck.

        This method updates the bottleneck to reflect changes in the encoder
        and decoder channel dimensions. It is typically called when the UNet
        architecture is enlarged with new layers that affectthe data flow into
        or out of the bottleneck.

        Args:
            in_channels (int, optional): New number of input channels.
            out_channels (int, optional): New number of output channels.

        Note:
            This method does not add layers; it modifies existing layers
            (e.g., convolutions) to match updated dimensions.
        """
        print("OOPS: Not Implemented Yet!")


# =============================================================================
class SinusoidalTimeEncoder(Module):
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
        trainable_freq (bool): Frequencies are trainable (defaults to False).
        trainable_ampl (bool): Amplitudes are trainable (defaults to False).
    """
    # Note that in the mentioned paper d_model, which is out n_embed, is 512,
    # and approximately 25000 source tokens and 25000 target tokens are used.
    # The shortest and largest wavelengths are `2 \pi` and  10000 x `2 \pi`,
    # resepectively. Therefore, the shortes wavelength covers about 6 tokens
    # and the longest wavelength contains about 6 x 10000 tokens.
    # For the default setting, we assume that time varies from 0 to 1 with time
    # steps of 0.001, a typical time step in solving a differential equation
    # for relatively smooth functions. Then the minimum and maximum angular
    # frequencies can be set to 1 and 1000 as the default choise.

    def __init__(
        self,
        n_embed: int,
        min_freq: float = 1.0,
        max_freq: float = 1000.0,
        trainable_freq: bool = False,
        trainable_ampl: bool = False
    ):

        assert n_embed % 2 == 0, "Embedding length must be even."

        super().__init__()

        self.n_embed = n_embed
        self.min_freq = min_freq
        self.max_freq = max_freq
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

    def forward(self, time, spatial_ndim=0):
        """
        Computes the sinusoidal time encodingdding.

        Args:
            time (Tensor): A 1D tensor representing time steps (batch axis).
            spatial_ndim (int): for reshaping the output (default is 0)

        Returns:
            Tensor: A tensor of original shape (batch_size, n_embed) with
                    sinusoidal encoding. The tensor then reshaped to have
                    `spatial_ndim` additional inner axis with unit lenght.
        """
        if self.trainable_freq:
            angle = time[:, None] * self.freq_ratio[None, :] * self.max_freq
        else:
            angle = time[:, None] * self.freq[None, :]

        shape = (len(time), self.n_embed, *(1,) * spatial_ndim)
        encoded_time = torch.zeros(shape[:2], device=time.device)

        encoded_time[:, 0::2] = torch.sin(angle)
        encoded_time[:, 1::2] = torch.cos(angle)

        if self.trainable_ampl:
            encoded_time = self.ampl[None, :] * encoded_time

        return encoded_time.reshape(*shape)


def _test_time_encoding(n_embed=512, **kwargs):
    import matplotlib.pyplot as plt
    plt.ion()
    t = torch.linspace(0, 1, 1000)
    plt.pcolor(SinusoidalTimeEncoder(n_embed, **kwargs)(t).detach())