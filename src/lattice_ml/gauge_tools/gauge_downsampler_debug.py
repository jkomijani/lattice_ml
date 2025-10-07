import torch
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

# Simple 2D test lattice
B, Lx, Ly, D, Nc = 1, 4, 4, 2, 1
x = torch.zeros((B, Lx, Ly, D, Nc, Nc))

# Fill it with easily distinguishable numbers
for i in range(Lx):
    for j in range(Ly):
        x[0, i, j, 0, 0, 0] = 10*i + j      # Ux(x)
        x[0, i, j, 1, 0, 0] = 100*i + 10*j  # Uy(x)

print("Fine lattice Ux:")
print(x[0, :, :, 0, 0, 0])
print("Fine lattice Uy:")
print(x[0, :, :, 1, 0, 0])

x_coarse = gauge_downsampler(x, prefix_dims=1, sites_before_link=True)

print("\nCoarse lattice shape:", x_coarse.shape)
print("Coarse Ux:")
print(x_coarse[0, :, :, 0, 0, 0])
print("Coarse Uy:")
print(x_coarse[0, :, :, 1, 0, 0])
