# Copyright (c) 2024 Javad Komijani


import torch

ifftn, fftn = torch.fft.ifftn, torch.fft.fftn


# =============================================================================
def spectral_split(x, *, cuts, dims,
        norm=None, spectral_scale=None, as_nested=False
        ):
    """
    Computes multi dimensional FFT of input tensor, along the specified
    dimensions, divides the spectrum in two pieces along each of those
    dimensions, performs the multi dimenstional inverse FFT of each block,
    and returns all blocks as a list. A symmetric normalization factor for both
    FFT and its inverse are assumed by default (unlike torch convention).

    Paramerers
    ----------
    x : Tensor
        the input tensor.
    cuts : Tuple[ind]
        The cut between low and high frequencies in corresponding dimensions.
        Each cut must be 0 or a positive, even integer. If 0, the cut is set to
        the middle frequency.
    dims : Tuple[int]
        dimensions to be splitted.
    norm : str, optional
        As described in torch.fft.fftn.
    spectral_scale : Tensor or number, optional
        if provided, the FFT of the input tensor gets scaled accordingly
        before getting splitted. (Default is None.)
        Note that if the input tensor ``x`` is real, ``spectral_scale`` cannot
        be arbitrary; it must be Hermitian-symmetric like the FFT of ``x``,
        otherwise ``spectral_cat`` and ``spectral_split`` are not each others
        inverse. (To this end, one can use ``conjugate_counterpart`` function.)
    as_nested : Boolean, optional
        if True the output is organized as nested lists. (Default is False.)
    """

    if spectral_scale is None:
        x_tilde = fftn(x, dim=dims, norm=norm)
    else:
        x_tilde = fftn(x, dim=dims, norm=norm) * spectral_scale

    blocks = Splitter.apply(dims, cuts, x_tilde)

    if torch.is_complex(x):
        blocks = [ifftn(blk, dim=dims, norm=norm) for blk in blocks]
    else:
        blocks = [ifftn(blk, dim=dims, norm=norm).real for blk in blocks]

    if as_nested:
        blocks = pack_as_nested(blocks)

    return blocks


def spectral_cat(blocks, *, cuts, dims,
        norm=None, spectral_scale=None, as_nested=False
        ):
    """Reverse of spectral_split."""

    if as_nested:
        blocks = unpack(blocks, len(dims))

    if cuts is None:
        cuts = [0] * len(dims)

    x_tilde = Concatenator.apply(
            dims, cuts, *[fftn(blk, dim=dims, norm=norm) for blk in blocks]
            )

    if spectral_scale is None:
        x = ifftn(x_tilde, dim=dims, norm=norm)
    else:
        x = ifftn(x_tilde / spectral_scale, dim=dims, norm=norm)

    if not torch.is_complex(blocks[0]):
        x = x.real

    return x


# =============================================================================
def splitted_fftn(x, *, cuts, dims,
        norm=None, spectral_scale=None, as_nested=False
        ):
    """Similar to ``spectal_split``, but the blocks are in Fourier space."""

    if spectral_scale is None:
        x_tilde = fftn(x, dim=dims, norm=norm)
    else:
        x_tilde = fftn(x, dim=dims, norm=norm) * spectral_scale

    blocks = list(Splitter.apply(dims, cuts, x_tilde))

    if as_nested:
        blocks = pack_as_nested(blocks)

    return blocks


def splitted_ifftn(blocks, *, cuts, dims,
        norm=None, spectral_scale=None, as_nested=False
        ):
    """Reverse of splitted_ifftn."""

    if as_nested:
        blocks = unpack(blocks, len(dims))

    x_tilde = Concatenator.apply(dims, cuts, *blocks)

    if spectral_scale is None:
        x = ifftn(x_tilde, dim=dims, norm=norm)
    else:
        x = ifftn(x_tilde / spectral_scale, dim=dims, norm=norm)

    return x


# =============================================================================
class Splitter(torch.autograd.Function):
    """Given data in Fourier domain, splits it to low and hight frequency
    blocks. The blocks have a particular property that is useful if the
    original data in the Physical domain are real: the inverse Fourier
    transform of each block is real too.

    Paramerers
    ----------
    dims : Tuple[ind]
        dimensions to be splitted.
    cuts : Tuple[ind]
        The cut between low and high frequencies in corresponding dimensions.
        Each cut must be 0 or a positive, even integer. If 0, the cut is set to
        the middle frequency.
    x_tilde : Tensor
        the input tensor in Fourier domain.
    """

    @staticmethod
    def forward(ctx, dims, cuts, x_tilde):

        pre_blocks = [x_tilde]

        inds = tuple([slice(None)] * x_tilde.ndim)
        coeff = 2**(-1/2)

        def fix_cutlines(x_tilde_n, axis, cut2):
            if cut2 == 0:
                cut = x_tilde_n.shape[axis] // 4
            else:
                cut = cut2 // 2
            inds1, inds3 = list(inds), list(inds)
            inds1[axis] = cut
            inds3[axis] = - cut
            x_tilde_n[inds1], x_tilde_n[inds3] = (
                    (x_tilde_n[inds1] + x_tilde_n[inds3]) * coeff,
                    (x_tilde_n[inds1] - x_tilde_n[inds3]) * (1j * coeff)
                    )
            return x_tilde_n

        def splitcat(x_tilde_n, axis, cut2):
            ell = x_tilde_n.shape[axis]
            if cut2 == 0:
                assert ell > 3, f"{axis} axis is too small to be splitted"
                cut = ell // 4
            else:
                cut = cut2 // 2
            splt = torch.split(x_tilde_n, [cut, ell - 2 * cut, cut], dim=axis)
            return torch.cat([splt[0], splt[2]], dim=axis), splt[1]

        for n, axis in enumerate(dims):
            cut = cuts[n]
            assert cut % 2 == 0
            blocks = [None] * (2 * len(pre_blocks))
            for n, x_tilde_n in enumerate(pre_blocks):
                x_tilde_n = fix_cutlines(x_tilde_n, axis, cut)
                blocks[2 * n: 2 * n + 2] = splitcat(x_tilde_n, axis, cut)
            pre_blocks = blocks

        if ctx is not None:
            ctx.save_for_backward(makesure_tensor(dims), makesure_tensor(cuts))

        # The output is returned as tuple for "torch.autograd.gradcheck"
        return tuple(blocks)

    @staticmethod
    def backward(ctx, *grad_blocks):
        dims = ctx.saved_tensors[0].tolist()
        cuts = ctx.saved_tensors[1].tolist()
        grad_dims, grad_cuts = None, None
        grad_x_tilde = Concatenator.forward(None, dims, cuts, *grad_blocks)
        return grad_dims, grad_cuts, grad_x_tilde


class Concatenator(torch.autograd.Function):
    """Reverse of Splitter."""

    @staticmethod
    def forward(ctx, dims, cuts, *blocks):
        # The input is taken as tuple for "torch.autograd.gradcheck"

        inds = tuple([slice(None)] * blocks[0].ndim)
        coeff = 2**(-1/2)

        def undo_cutlines(x_tilde_n, axis, cut2):
            if cut2 == 0:
                cut = x_tilde_n.shape[axis] // 4
            else:
                cut = cut2 // 2
            inds1, inds3 = list(inds), list(inds)
            inds1[axis] = cut
            inds3[axis] = - cut
            x_tilde_n[inds1], x_tilde_n[inds3] = (
                    (x_tilde_n[inds1] - 1j * x_tilde_n[inds3]) * coeff,
                    (x_tilde_n[inds1] + 1j * x_tilde_n[inds3]) * coeff
                    )
            return x_tilde_n

        def undo_splitcat(splt_0, splt_1, axis, cut2):
            if cut2 == 0:
                cut = splt_0.shape[axis] // 2
            else:
                cut = cut2 // 2
            splt = torch.split(splt_0, [cut, cut], dim=axis)
            return torch.cat([splt[0], splt_1, splt[1]], dim=axis)

        for n, axis in enumerate(dims[::-1]):
            cut = cuts[-1 - n]
            assert cut % 2 == 0
            pre_blocks = [None] * (len(blocks) // 2)
            for n, _ in enumerate(pre_blocks):
                x_tilde_n = undo_splitcat(*blocks[2 * n: 2 * n + 2], axis, cut)
                pre_blocks[n] = undo_cutlines(x_tilde_n, axis, cut)
            blocks = pre_blocks

        assert len(blocks) == 1
        x_tilde = blocks[0]

        if ctx is not None:
            ctx.save_for_backward(makesure_tensor(dims), makesure_tensor(cuts))

        return x_tilde

    @staticmethod
    def backward(ctx, grad_x_tilde):
        dims = ctx.saved_tensors[0].tolist()
        cuts = ctx.saved_tensors[1].tolist()
        grad_dims, grad_cuts = None, None
        grad_blocks = Splitter.forward(None, dims, cuts, grad_x_tilde)
        return (grad_dims, grad_cuts, *grad_blocks)


# =============================================================================
def makesure_tensor(dims):
    if torch.is_tensor(dims):
        return dims
    else:
        return torch.tensor(dims)


def conjugate_counterpart(x, dims=None):
    """Return ``x[:, -i_1, -i_2, ..., -i_n]``.

    Note that the FFT of a real signal is Hermitian-symmetric as
    ``x[:, -i_1, -i_2, ..., -i_n] = conj(x[:, i_1, i_2, ..., i_n])``.
    """
    if dims is None:
        dims = tuple(range(1, x.ndim))
    return torch.flip(torch.roll(x, [-1] * len(dims), dims=dims), dims=dims)


def pack_as_nested(blocks):
    n = len(blocks) // 2
    if n == 1:
        return [blocks[0], blocks[1]]
    else:
        return [pack_as_nested(blocks[:n]), pack_as_nested(blocks[n:])]


def unpack(nested, depth):
    if depth == 1:
        return nested
    else:
        return [*unpack(nested[0], depth - 1), *unpack(nested[1], depth - 1)]


# =============================================================================
def gradcheck(*, shape, cuts, dims, complex_input=False):
    """For sanity check of the implemented backward propagations.
    For example use:
        >>> gradcheck(shape=(2, 8, 3, 9), cuts=(4, 3), dims=(1, 3))
    """

    torch.set_default_dtype(torch.float64)  # double precision

    scale = torch.randn(*shape)
    if not complex_input:
        # then, scale must be Hermitian-symmetric
        scale = (scale + conjugate_counterpart(scale, dims=dims)) / 2.

    kwargs = dict(cuts=cuts, dims=dims, spectral_scale=scale)

    split = lambda x: spectral_split(x, **kwargs)
    cat = lambda *blocks: spectral_cat(blocks, **kwargs)

    if complex_input:
        x = torch.randn(*shape) + 1j * torch.randn(*shape)
    else:
        x = torch.randn(*shape)
    x.requires_grad = True
    blocks = split(x)

    print("spectral_cat(spectral_split(.)) == Identity:", end='\t')
    x_hat = spectral_cat(blocks, **kwargs)
    print(torch.mean((x_hat - x).abs()).item() < 1e-13)

    print("gradcheck(spectral_split):", end='\t')
    print(torch.autograd.gradcheck(split, x))
    print("gradcheck(spectral_cat):", end='\t')
    print(torch.autograd.gradcheck(cat, blocks))

    fftn = lambda x: splitted_fftn(x, **kwargs)
    ifftn = lambda *blks: splitted_ifftn(blks, **kwargs)

    blocks = fftn(x)

    print("splitted_ifftn(splitted_fftn(.)) == Identity:", end='\t')
    x_hat = splitted_ifftn(blocks, **kwargs)
    print(torch.mean((x_hat - x).abs()).item() < 1e-13)

    print("gradcheck(splitted_fftn):", end='\t')
    print(torch.autograd.gradcheck(fftn, x))
    print("gradcheck(splitted_ifftn):", end='\t')
    print(torch.autograd.gradcheck(ifftn, blocks))
