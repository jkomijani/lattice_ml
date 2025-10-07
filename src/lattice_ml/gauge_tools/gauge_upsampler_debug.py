import torch
torch.set_printoptions(sci_mode=False, linewidth=140)

# -------------------------------------------------------------------
# Aliases & utilities
# -------------------------------------------------------------------
matmul = torch.matmul

def _dagger(x: torch.Tensor) -> torch.Tensor:
    return x.conj().transpose(-1, -2)

def _sqrtm_unitary(Q: torch.Tensor, *, project_back: bool = True) -> torch.Tensor:
    Sq = torch.linalg.sqrtm(Q)
    if project_back:
        # Polar projection to unitary, det=1
        U, _ = torch.linalg.polar(Sq)
        detU = torch.det(U)
        Nc = U.shape[-1]
        # Avoid divide-by-zero:
        det_factor = torch.pow(detU, -1.0 / Nc).unsqueeze(-1).unsqueeze(-1)
        U = U * det_factor
        return U
    return Sq

# -------------------------------------------------------------------
# Your functions (down- and up-samplers)
# -------------------------------------------------------------------
def gauge_downsampler(
    x: torch.Tensor,
    prefix_dims: int = 1,
    sites_before_link: bool = True,
) -> torch.Tensor:
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

    links = torch.unbind(x, dim=link_axis_in_x)  # length D

    sample_link = links[0]
    even_idx = [slice(None)] * sample_link.ndim
    for ax in range(spatial_start, spatial_start + (spatial_end - spatial_start)):
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

def gauge_upsampler(
    x_fine_pre: torch.Tensor,          # fine BEFORE transform (contains a,b,...)
    x_coarse_post: torch.Tensor,       # coarse AFTER transform (contains A')
    prefix_dims: int = 1,
    sites_before_link: bool = True,
    project_back_to_suN: bool = True,
) -> torch.Tensor:
    x = x_fine_pre
    if sites_before_link:
        link_axis_in_x = -3
        spatial_start = prefix_dims
        spatial_end   = x.ndim - 3
        d = spatial_end - spatial_start
        def roll_dim(mu): return prefix_dims + mu
    else:
        link_axis_in_x = prefix_dims
        spatial_start = prefix_dims + 1
        spatial_end   = x.ndim - 2
        d = spatial_end - spatial_start
        def roll_dim(mu): return prefix_dims + 1 + mu

    fine_links_pre = list(torch.unbind(x, dim=link_axis_in_x))            # D tensors: (*prefix, L..., Nc, Nc)
    # stack_dim for x_coarse_post is exactly where the link axis sits relative to sites:
    stack_dim = (prefix_dims + d) if sites_before_link else prefix_dims
    coarse_links_post = list(torch.unbind(x_coarse_post, dim=stack_dim))  # D tensors: (*prefix, L/2..., Nc, Nc)

    sample = fine_links_pre[0]
    even_idx = [slice(None)] * sample.ndim
    for ax in range(spatial_start, spatial_start + d):
        even_idx[ax] = slice(0, None, 2)
    even_idx = tuple(even_idx)

    def odd_idx_for_mu(mu):
        odd = list(even_idx)
        ax = roll_dim(mu)
        odd[ax] = slice(1, None, 2)  # positions immediately after even tails along μ
        return tuple(odd)

    fine_links_post = [u.clone() for u in fine_links_pre]

    for mu in range(len(fine_links_pre)):
        u_pre  = fine_links_pre[mu]
        A_post = coarse_links_post[mu]

        a = u_pre[even_idx]
        u_shift = torch.roll(u_pre, shifts=-1, dims=roll_dim(mu))
        b = u_shift[even_idx]

        A_pre = matmul(a, b)
        Q = matmul(matmul(A_pre, _dagger(A_post)), A_pre)
        Sq = _sqrtm_unitary(Q, project_back=project_back_to_suN)

        a_post = matmul(matmul(A_post, _dagger(b)), Sq)
        b_post = matmul(matmul(Sq, _dagger(a)), A_post)

        out_mu = fine_links_post[mu]
        out_mu[even_idx]          = a_post
        out_mu[odd_idx_for_mu(mu)] = b_post
        fine_links_post[mu] = out_mu

    x_fine_post = torch.stack(fine_links_post, dim=link_axis_in_x)
    return x_fine_post

# -------------------------------------------------------------------
# Small printable Nc=1 test (real scalars) — easy to eyeball
# -------------------------------------------------------------------
def make_scalar_lattice_2d(B, Lx, Ly, D=2, sites_before_link=True):
    # U_mu(x,y) = 100*mu + 10*x + y + 1
    Nc = 1
    if sites_before_link:
        x = torch.zeros((B, Lx, Ly, D, Nc, Nc), dtype=torch.float64)
        for mu in range(D):
            for i in range(Lx):
                for j in range(Ly):
                    val = 100*mu + 10*i + j + 1
                    x[:, i, j, mu, 0, 0] = val
    else:
        x = torch.zeros((B, D, Lx, Ly, Nc, Nc), dtype=torch.float64)
        for mu in range(D):
            for i in range(Lx):
                for j in range(Ly):
                    val = 100*mu + 10*i + j + 1
                    x[:, mu, i, j, 0, 0] = val
    return x

def print_per_direction_scalar_grid(x, sites_before_link=True, title=""):
    if title: print(f"\n=== {title} ===")
    B = x.shape[0]
    assert B == 1
    if sites_before_link:
        Lx, Ly, D = x.shape[1], x.shape[2], x.shape[3]
        for mu in range(D):
            grid = x[0, :, :, mu, 0, 0]
            print(f"\nDirection μ={mu} (shape {tuple(grid.shape)}):")
            for i in range(Lx):
                print(" ".join(f"{int(v):4d}" for v in grid[i]))
    else:
        D, Lx, Ly = x.shape[1], x.shape[2], x.shape[3]
        for mu in range(D):
            grid = x[0, mu, :, :, 0, 0]
            print(f"\nDirection μ={mu} (shape {tuple(grid.shape)}):")
            for i in range(Lx):
                print(" ".join(f"{int(v):4d}" for v in grid[i]))

# Parameters
B, Lx, Ly, D = 1, 4, 4, 2
sites_before_link = True

# Build fine (pre) lattice and show it
x_fine_pre = make_scalar_lattice_2d(B, Lx, Ly, D=D, sites_before_link=sites_before_link)
print_per_direction_scalar_grid(x_fine_pre, sites_before_link, title="FINE (pre) — Nc=1")

# Downsample to get coarse (pre)
x_coarse_pre = gauge_downsampler(x_fine_pre, prefix_dims=1, sites_before_link=sites_before_link)
print_per_direction_scalar_grid(x_coarse_pre, sites_before_link, title="COARSE (pre) = downsample(FINE pre)")

# Define a target coarse POST: take A' = 2 * A  (nontrivial but simple)
scale = 2.0
x_coarse_post = x_coarse_pre * scale
print_per_direction_scalar_grid(x_coarse_post, sites_before_link, title="COARSE (post) = 2 * COARSE (pre)")

# Upsample back to fine (post) using the provided rule
x_fine_post = gauge_upsampler(
    x_fine_pre=x_fine_pre,
    x_coarse_post=x_coarse_post,
    prefix_dims=1,
    sites_before_link=sites_before_link,
    project_back_to_suN=False,   # exact sqrt for scalars
)

# Downsample the upsampled fine and compare with target coarse_post
x_coarse_check = gauge_downsampler(x_fine_post, prefix_dims=1, sites_before_link=sites_before_link)

print_per_direction_scalar_grid(x_fine_post, sites_before_link, title="FINE (post) — after upsample split")
print_per_direction_scalar_grid(x_coarse_check, sites_before_link, title="COARSE (check) = downsample(FINE post)")

print("\nAllclose check (Nc=1):", torch.allclose(x_coarse_check, x_coarse_post))

# Also show shapes
print("\nShapes:")
print("fine pre:", tuple(x_fine_pre.shape))
print("coarse pre:", tuple(x_coarse_pre.shape))
print("coarse post:", tuple(x_coarse_post.shape))
print("fine post:", tuple(x_fine_post.shape))
print("coarse check:", tuple(x_coarse_check.shape))
