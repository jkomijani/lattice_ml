import torch
from lattice_ml.functions import pow_special_unitary_group

matmul = torch.matmul

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
        link_axis_ = -3
        spatial_start = prefix_dims
        spatial_end   = x.ndim - 3  # up to (but not incl.) link axis
        d = spatial_end - spatial_start # actually D = d but I keep them separate in case we downsample only in some directions.
        # After removing link axis, stack back at boundary between sites and matrices
        stack_dim = prefix_dims + d
        # dims used for torch.roll
        def roll_dim(mu): return prefix_dims + mu
    else:
        # axes: (*prefix, D, L1...Ld, Nc, Nc)
        link_axis_ = prefix_dims
        spatial_start = prefix_dims + 1
        spatial_end   = x.ndim - 2
        d = spatial_end - spatial_start
        stack_dim = prefix_dims  # insert link axis right after *prefix
        def roll_dim(mu): return prefix_dims + 1 + mu

    # Unbind the link-direction axis: list of tensors, one per direction μ
    links = torch.unbind(x, dim=link_axis_)  # length D

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


def gauge_upsampler(
    x_fine_pre: torch.Tensor,          # fine lattice BEFORE transform (contains a,b,...)
    x_coarse_post: torch.Tensor,       # coarse lattice AFTER transform (contains A')
    prefix_dims: int = 1,
    sites_before_link: bool = True,
) -> torch.Tensor:
    """
    Upsample a transformed coarse lattice A' back to fine links (a', b')
    using the original fine lattice before transform (a, b) as 'middle links'.

    For each direction μ and for links whose tails lie on even sites:
        A  = a @ b               (from x_fine_pre)
        Q  = a† @ A @ A'^† @ A @ b† (set to I for now)
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
        link_axis_= -3
        spatial_start = prefix_dims
        spatial_end   = x.ndim - 3
        d = spatial_end - spatial_start
        def roll_dim(mu): return prefix_dims + mu
    else:
        # axes: (*prefix, D, L1...Ld, Nc, Nc)
        link_axis_ = prefix_dims
        spatial_start = prefix_dims + 1
        spatial_end   = x.ndim - 2
        d = spatial_end - spatial_start
        def roll_dim(mu): return prefix_dims + 1 + mu

    # Unbind directions for the fine PRE lattice and coarse POST lattice
    fine_links_pre = list(torch.unbind(x, dim=link_axis_))       # D tensors of shape (*prefix, L..., Nc, Nc)
    coarse_links_post = list(torch.unbind(x_coarse_post, dim=link_axis_))  # D tensors of shape (*prefix, L/2..., Nc, Nc)

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

        set_id = True

        if not set_id:
            # Q = a† @ A @ A'^† @ A @ b†
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

            Sq =  pow_special_unitary_group(Q, 0.5)
        else:
            Nc = A_pre.size(-1)
            # Identity with correct shape/dtype/device, broadcast over batch/lattice dims
            Sq = torch.eye(Nc, dtype=A_pre.dtype, device=A_pre.device).expand(
                A_pre.shape[:-2] + (Nc, Nc)
            )
            
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
    x_fine_post = torch.stack(fine_links_post, dim=link_axis_)
    return x_fine_post

# Test field
B, Lx, Ly, D, Nc = 1, 4, 4, 2, 1
x = torch.zeros((B, Lx, Ly, D, Nc, Nc))
for i in range(Lx):
    for j in range(Ly):
        x[0, i, j, 0, 0, 0] = 10*i + j      # Ux (vertical)
        x[0, i, j, 1, 0, 0] = 100*i + 10*j  # Uy (horizontal)



def _test_gauge_equivaraince():
    """Shows the gauge equivariance of the transformation in gauge_downsampler."""

    import normflow  # pylint: disable=import-outside-toplevel
    shape = (2, 2, 2, 2, 4)  # 2^4 lattice; the last axis is the "mu" axis.
    prior = normflow.prior.SUnPrior(3, shape=shape)

    x = prior.sample(2)

    x_coarse = gauge_downsampler(x, prefix_dims=1, sites_before_link=True)
    y = gauge_upsampler(x, x_coarse)

    # Now gauge transform `x`; only the links connected to the origin
    q = prior.sample(1)[0, 0, 0, 0, 0, 0]
    for i in range(4):
        x[0, 0, 0, 0, 0, i] = q @ x[0, 0, 0, 0, 0, i]
    x[0, -1, 0, 0, 0, 0] = x[0, -1, 0, 0, 0, 0] @ q.adjoint()
    x[0, 0, -1, 0, 0, 1] = x[0, 0, -1, 0, 0, 1] @ q.adjoint()
    x[0, 0, 0, -1, 0, 2] = x[0, 0, 0, -1, 0, 2] @ q.adjoint()
    x[0, 0, 0, 0, -1, 3] = x[0, 0, 0, 0, -1, 3] @ q.adjoint()

    # Use the gauge transformed x & transform it w/ instances of GaugeLinkConv
    z = gauge_upsampler(x, gauge_downsampler(x))

    # Undo the gauge transformation on `z` to check the gauge equivarience.
    for i in range(4):
        z[0, 0, 0, 0, 0, i] = q.adjoint() @ z[0, 0, 0, 0, 0, i]
    z[0, -1, 0, 0, 0, 0] = z[0, -1, 0, 0, 0, 0] @ q
    z[0, 0, -1, 0, 0, 1] = z[0, 0, -1, 0, 0, 1] @ q
    z[0, 0, 0, -1, 0, 2] = z[0, 0, 0, -1, 0, 2] @ q
    z[0, 0, 0, 0, -1, 3] = z[0, 0, 0, 0, -1, 3] @ q

    print(f"Gauge Equivariant if {(z - y).abs().mean()} is approximately 0")

_test_gauge_equivaraince()