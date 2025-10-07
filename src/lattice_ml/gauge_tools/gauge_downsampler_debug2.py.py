# Revised visualization: make directions consistent with the downsampler.
# By convention in this test:
#   - First site axis is x (rows, vertical).
#   - Second site axis is y (columns, horizontal).
# Therefore:
#   - Ux(x,y) goes VERTICALLY (downwards) from (x,y) to (x+1,y).
#   - Uy(x,y) goes HORIZONTALLY (right) from (x,y) to (x,y+1).
#
# We'll reprint BOTH the fine lattice and the coarse lattice with this convention,
# using the SAME original gauge_downsampler (unchanged).

import torch

# Use the same test field
B, Lx, Ly, D, Nc = 1, 4, 4, 2, 1
x = torch.zeros((B, Lx, Ly, D, Nc, Nc))
for i in range(Lx):
    for j in range(Ly):
        x[0, i, j, 0, 0, 0] = 10*i + j      # Ux (vertical)
        x[0, i, j, 1, 0, 0] = 100*i + 10*j  # Uy (horizontal)

# Original downsampler (reuse from previous cell)
# (Assumes it's already defined; redefine quickly for isolation)
matmul = torch.matmul
def gauge_downsampler(
    x: torch.Tensor,
    prefix_dims: int = 1,
    sites_before_link: bool = True,
) -> torch.Tensor:
    if sites_before_link:
        link_axis_in_x = -3
        spatial_start = prefix_dims
        spatial_end   = x.ndim - 3
        d = spatial_end - spatial_start
        stack_dim = prefix_dims + d
        def roll_dim(mu): return prefix_dims + mu
    else:
        link_axis_in_x = prefix_dims
        spatial_start = prefix_dims + 1
        spatial_end   = x.ndim - 2
        d = spatial_end - spatial_start
        stack_dim = prefix_dims
        def roll_dim(mu): return prefix_dims + 1 + mu

    links = torch.unbind(x, dim=link_axis_in_x)
    sample_link = links[0]
    even_idx = [slice(None)] * sample_link.ndim
    for ax in range(spatial_start, spatial_start + d):
        even_idx[ax] = slice(0, None, 2)
    even_idx = tuple(even_idx)

    coarse_links = []
    for mu, u in enumerate(links):
        u_even = u[even_idx]
        u_shift = torch.roll(u, shifts=-1, dims=roll_dim(mu))
        u_shift_even = u_shift[even_idx]
        u_coarse = matmul(u_even, u_shift_even)
        coarse_links.append(u_coarse)

    x_coarse = torch.stack(coarse_links, dim=stack_dim)
    return x_coarse

def extract_Ux_Uy_from_x(x, prefix_dims=1, sites_before_link=True, batch_index=0):
    idx = [slice(None)] * x.ndim
    if prefix_dims >= 1:
        idx[0] = batch_index
    xs = x[tuple(idx)]
    if sites_before_link:
        Ux = xs[..., 0, 0, 0]  # scalar Nc=1
        Uy = xs[..., 1, 0, 0]
    else:
        Ux = xs[0, ..., 0, 0]
        Uy = xs[1, ..., 0, 0]
    return Ux, Uy

def print_lattice_ascii_vertical_Ux_horizontal_Uy(Ux: torch.Tensor, Uy: torch.Tensor, title: str, show_wrap=False):
    """
    Print sites (x,y). Vertical edges labeled by Ux(x,y) (since Ux goes along +x),
    horizontal edges labeled by Uy(x,y) (since Uy goes along +y).
    """
    Lx, Ly = Ux.shape
    print(f"\n=== {title} ===\n")
    w = max(3, len(str(int(torch.max(torch.abs(Ux)).item()))),
               len(str(int(torch.max(torch.abs(Uy)).item()))))

    for x_idx in range(Lx):
        # Row of sites with horizontal Uy to the right
        line = []
        for y_idx in range(Ly):
            line.append(f"({x_idx},{y_idx})")
            if y_idx < Ly - 1:
                line.append(f"--{int(Uy[x_idx,y_idx]):>{w}}--> ")
            elif show_wrap:
                line.append(f"--{int(Uy[x_idx,y_idx]):>{w}}↷ ")
        print("".join(line))

        # Vertical Ux down arrows to next row
        if x_idx < Lx - 1:
            vline = []
            for y_idx in range(Ly):
                pad = " " * (len(f"({x_idx},{y_idx})")//2)
                vline.append(pad + f"|{int(Ux[x_idx,y_idx]):>{w}}" + pad)
                if y_idx < Ly - 1:
                    vline.append(" " * (6 + w))
                elif show_wrap:
                    vline.append(" " * (3 + w))
            print(vline[0] + "".join(vline[1:]))
        elif show_wrap:
            # bottom wrap vertical
            vwrap = []
            for y_idx in range(Ly):
                pad = " " * (len(f"({x_idx},{y_idx})")//2)
                vwrap.append(pad + f"v{int(Ux[x_idx,y_idx]):>{w}}" + pad)
                if y_idx < Ly - 1:
                    vwrap.append(" " * (6 + w))
                else:
                    vwrap.append(" " * (3 + w))
            print(vwrap[0] + "".join(vwrap[1:]))

# ---- Print FINE lattice with corrected orientation ----
Ux_fine, Uy_fine = extract_Ux_Uy_from_x(x, prefix_dims=1, sites_before_link=True)
print_lattice_ascii_vertical_Ux_horizontal_Uy(Ux_fine, Uy_fine, "FINE lattice (Ux vertical, Uy horizontal)")

# ---- Downsample and print COARSE lattice ----
x_coarse = gauge_downsampler(x, prefix_dims=1, sites_before_link=True)
Ux_c, Uy_c = extract_Ux_Uy_from_x(x_coarse, prefix_dims=1, sites_before_link=True)
print_lattice_ascii_vertical_Ux_horizontal_Uy(Ux_c, Uy_c, "COARSE lattice (even sites; Ux vertical, Uy horizontal)")
