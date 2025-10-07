import torch
torch.set_printoptions(sci_mode=False, linewidth=140)

# -------------------------------------------------------------------
# Aliases & utilities
# -------------------------------------------------------------------
matmul = torch.matmul


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
