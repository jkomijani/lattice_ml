import torch
matmul = torch.matmul

def matsqrt(Q: torch.Tensor) -> torch.Tensor:
    """
    Principal (batched) matrix square root for normal / unitary Q.
    Shapes: (..., n, n) -> (..., n, n)
    """
    assert Q.shape[-1] == Q.shape[-2], "Q must be square"

    # Ensure complex dtype for eig
    if torch.is_complex(Q):
        Qc = Q
    else:
        Qc = Q.to(torch.complex64 if Q.dtype == torch.float32 else torch.complex128)

    # Eig decomposition
    w, V = torch.linalg.eig(Qc)                      # (..., n), (..., n, n)
    Dsqrt = torch.diag_embed(torch.sqrt(w))          # principal branch

    # Prefer solve over explicit inverse: V X = Dsqrt  →  X = V^{-1} Dsqrt
    X = torch.linalg.solve(V, Dsqrt)                 # (..., n, n)
    Sq = V @ X

    return Sq.to(dtype=Q.dtype)


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
        Q  = A @ A'^† @ A
        a' = A' @ b^† @ sqrt(Q)
        b' = sqrt(Q) @ a^† @ A'

    All other fine links are copied unchanged from x_fine_pre.

    Shapes:
      - sites_before_link=True:  (*prefix, L1,...,Ld, D, Nc, Nc)
      - sites_before_link=False: (*prefix, D, L1,...,Ld, Nc, Nc)
    """
    # --- Align dtype & device to avoid Float/Double or CPU/GPU mismatches
    x = x_fine_pre
    x_coarse_post = x_coarse_post.to(dtype=x.dtype, device=x.device)

    if sites_before_link:
        # axes: (*prefix, L1...Ld, D, Nc, Nc)
        link_axis_ = -3
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

    # Unbind directions for fine PRE and coarse POST
    fine_links_pre = list(torch.unbind(x, dim=link_axis_))              # D tensors: (*prefix, L..., Nc, Nc)
    coarse_links_post = list(torch.unbind(x_coarse_post, dim=link_axis_))  # D tensors: (*prefix, L/2..., Nc, Nc)

    # Build even-site index tuple (stride-2) on spatial axes
    sample = fine_links_pre[0]
    even_idx = [slice(None)] * sample.ndim
    for ax in range(spatial_start, spatial_start + d):
        even_idx[ax] = slice(0, None, 2)
    even_idx = tuple(even_idx)

    # Companion index for the second link (odd tail) along μ: same as even_idx but +1 on μ-axis
    def odd_idx_for_mu(mu):
        odd = list(even_idx)
        ax = roll_dim(mu)
        odd[ax] = slice(1, None, 2)
        return tuple(odd)

    # Start from a copy so untouched links remain identical
    fine_links_post = [u.clone() for u in fine_links_pre]

    # Split in each direction μ
    for mu in range(len(fine_links_pre)):
        u_pre = fine_links_pre[mu]     # (*prefix, L..., Nc, Nc)
        A_post = coarse_links_post[mu] # (*prefix, L/2..., Nc, Nc)

        # Even-tail link a, and its immediate neighbor b along μ (via periodic roll)
        a = u_pre[even_idx]                                # (*prefix, L/2..., Nc, Nc)
        u_shift = torch.roll(u_pre, shifts=-1, dims=roll_dim(mu))
        b = u_shift[even_idx]

        # A from fine PRE
        A_pre = matmul(a, b)

        # Q = A A'^† A
        Q = matmul(matmul(A_pre, A_post.mH), A_pre)        # use mH (conj-transpose)

        # sqrt(Q)
        Sq = matsqrt(Q)

        # a' = A' b^† sqrt(Q),  b' = sqrt(Q) a^† A'
        a_post = matmul(matmul(A_post, b.mH), Sq)
        b_post = matmul(matmul(Sq, a.mH), A_post)

        # Scatter back
        out_mu = fine_links_post[mu]
        out_mu[even_idx] = a_post
        out_mu[odd_idx_for_mu(mu)] = b_post
        fine_links_post[mu] = out_mu

    # Re-stack directions
    x_fine_post = torch.stack(fine_links_post, dim=link_axis_)
    return x_fine_post


# === TEST ===
B, Lx, Ly, D, Nc = 1, 4, 4, 2, 1
x_fine_pre = torch.zeros((B, Lx, Ly, D, Nc, Nc), dtype=torch.float64)

# Fill fine lattice: simple numeric pattern
for i in range(Lx):
    for j in range(Ly):
        x_fine_pre[0, i, j, 0, 0, 0] = 10*i + j      # Ux (vertical)
        x_fine_pre[0, i, j, 1, 0, 0] = 100*i + 10*j  # Uy (horizontal)

# Create a dummy coarse lattice with matching dtype/device
coarse = torch.zeros((B, Lx//2, Ly//2, D, Nc, Nc), dtype=x_fine_pre.dtype, device=x_fine_pre.device)
for i in range(Lx//2):
    for j in range(Ly//2):
        coarse[0, i, j, 0, 0, 0] = 0.5 * (20*i + 2*j)
        coarse[0, i, j, 1, 0, 0] = 0.5 * (200*i + 20*j)

# Run upsampler
x_fine_post = gauge_upsampler(x_fine_pre, coarse)

# === PRINT ===
def print_lattice(U: torch.Tensor, name: str):
    print(f"\n{name}:")
    # Works for both fine (Lx,Ly) and coarse (Lx//2,Ly//2)
    print("Ux (vertical):")
    print(U[0, :, :, 0, 0, 0])
    print("Uy (horizontal):")
    print(U[0, :, :, 1, 0, 0])

print_lattice(x_fine_pre,  "Fine lattice (pre)")
print_lattice(coarse,      "Coarse lattice (post)")
print_lattice(x_fine_post, "Fine lattice (post-upsampled)")

def _test_gauge_equivaraince(fine, coarse):
    """Shows the gauge equivariance of the transformation in gauge_downsampler."""

    import normflow  # pylint: disable=import-outside-toplevel
    shape = (4, 4, 4, 4, 4)  # 2^4 lattice; the last axis is the "mu" axis.
    shape2 =  (2, 2, 2,  2,4)
    prior = normflow.prior.SUnPrior(3, shape=shape)
    prior2 = normflow.prior.SUnPrior(3, shape=shape2)

    # Define `x` and transform it with instances of GaugeLinkConv

    x = prior.sample(2)
    print(x.shape)
    coarse = prior2.sample(2)
    print(coarse.shape)
    y = gauge_upsampler(x, coarse)

    # Now gauge transform `x`; only the links connected to the origin
    q = prior.sample(1)[0, 0, 0, 0, 0, 0]
    for i in range(4):
        x[0, 0, 0, 0, 0, i] = q @ x[0, 0, 0, 0, 0, i]
    x[0, -1, 0, 0, 0, 0] = x[0, -1, 0, 0, 0, 0] @ q.adjoint()
    x[0, 0, -1, 0, 0, 1] = x[0, 0, -1, 0, 0, 1] @ q.adjoint()
    x[0, 0, 0, -1, 0, 2] = x[0, 0, 0, -1, 0, 2] @ q.adjoint()
    x[0, 0, 0, 0, -1, 3] = x[0, 0, 0, 0, -1, 3] @ q.adjoint()

    for i in range(4):
        coarse[0, 0, 0, 0, 0, i] = q @ coarse[0, 0, 0, 0, 0, i]
    coarse[0, -1, 0, 0, 0, 0] = coarse[0, -1, 0, 0, 0, 0] @ q.adjoint()
    coarse[0, 0, -1, 0, 0, 1] = coarse[0, 0, -1, 0, 0, 1] @ q.adjoint()
    coarse[0, 0, 0, -1, 0, 2] = coarse[0, 0, 0, -1, 0, 2] @ q.adjoint()
    coarse[0, 0, 0, 0, -1, 3] = coarse[0, 0, 0, 0, -1, 3] @ q.adjoint()

    # Use the gauge transformed x & transform it w/ instances of GaugeLinkConv
    z = gauge_upsampler(x, coarse)

    # Undo the gauge transformation on `z` to check the gauge equivarience.
    for i in range(4):
        z[0, 0, 0, 0, 0, i] = q.adjoint() @ z[0, 0, 0, 0, 0, i]
    z[0, -1, 0, 0, 0, 0] = z[0, -1, 0, 0, 0, 0] @ q
    z[0, 0, -1, 0, 0, 1] = z[0, 0, -1, 0, 0, 1] @ q
    z[0, 0, 0, -1, 0, 2] = z[0, 0, 0, -1, 0, 2] @ q
    z[0, 0, 0, 0, -1, 3] = z[0, 0, 0, 0, -1, 3] @ q

    print(f"Gauge Equivariant if {(z - y).abs().mean()} is approximately 0")

_test_gauge_equivaraince(x_fine_pre, coarse)