import torch
import math

matmul = torch.matmul

def compute_wilson_1x1_loop( 
    x: torch.Tensor,
    mu: int,
    nu: int,
    prefix_dims: int = 1,
    sites_before_link: bool = True
):
    """
    1×1 Wilson loop at each site:
      W(x) = U_mu(x) U_nu(x+mu) U_mu(x+nu)^\dagger U_nu(x)^\dagger
    Layout assumed: sites_before_link=True -> x.shape=(B, Lx, Ly, D, Nc, Nc)
    """
    link_axis = -3 if sites_before_link else prefix_dims
    links = torch.unbind(x, dim=link_axis)

    x_mu = links[mu]  # (*prefix, L..., Nc, Nc)
    x_nu = links[nu]

    # U_nu(x+mu) and U_mu(x+nu)
    y_nu = torch.roll(x_nu, -1, dims=prefix_dims + mu)  # roll along μ-site axis
    z_mu = torch.roll(x_mu, -1, dims=prefix_dims + nu)  # roll along ν-site axis

    # W_11(x) = U_mu(x) U_nu(x+μ) [U_nu(x) U_mu(x+ν)]^\dagger
    w_11 = matmul(matmul(x_mu, y_nu), matmul(x_nu, z_mu).adjoint())
    return w_11  # (*prefix, L..., Nc, Nc)

def print_lattice_ascii_Ux_vertical_Uy_horizontal(Ux: torch.Tensor, Uy: torch.Tensor, title: str):
    """Show sites (x,y); **Ux is vertical**, **Uy is horizontal**."""
    Lx, Ly = Ux.shape
    print(f"\n=== {title} ===\n")
    def maxlen_num(t):
        t_abs = torch.abs(t) if torch.is_complex(t) else torch.abs(t)
        return len(str(int(torch.max(t_abs).item()))) if t.numel() else 1
    w = max(3, maxlen_num(Ux), maxlen_num(Uy))
    for x_idx in range(Lx):
        # Horizontal (Uy) to the right
        line = []
        for y_idx in range(Ly):
            line.append(f"({x_idx},{y_idx})")
            if y_idx < Ly - 1:
                val = f"{Uy[x_idx,y_idx].real:.2f}" if torch.is_complex(Uy) else f"{int(Uy[x_idx,y_idx])}"
                line.append(f"--{val:>{w}}--> ")
        print("".join(line))
        # Vertical (Ux) downwards
        if x_idx < Lx - 1:
            vline = []
            for y_idx in range(Ly):
                val = f"{Ux[x_idx,y_idx].real:.2f}" if torch.is_complex(Ux) else f"{int(Ux[x_idx,y_idx])}"
                pad = " " * (len(f"({x_idx},{y_idx})")//2)
                vline.append(pad + f"|{val:>{w}}" + pad)
                if y_idx < Ly - 1:
                    vline.append(" " * (6 + w))
            print("".join(vline))
    print()

def demo_integer_links():
    print("\n######## Demo A: Integer-valued links (easy to trace indices) ########")
    B, Lx, Ly, D, Nc = 1, 4, 4, 2, 1
    x = torch.zeros((B, Lx, Ly, D, Nc, Nc))
    # Ux(x,y) runs along +x; Uy(x,y) runs along +y
    for i in range(Lx):
        for j in range(Ly):
            x[0, i, j, 0, 0, 0] = 10*i + j      # Ux
            x[0, i, j, 1, 0, 0] = 100*i + 10*j  # Uy
    
    Ux = x[0, :, :, 0, 0, 0]
    Uy = x[0, :, :, 1, 0, 0]

    x_ = x[0, :, :, :, 0, 0]
    print('x', x_)
    print('Ux', Ux)
    print('Uy', Uy)

    print_lattice_ascii_Ux_vertical_Uy_horizontal(
        Ux, Uy, "FINE lattice (Ux vertical, Uy horizontal)"
    )

    W = compute_wilson_1x1_loop(x, mu=0, nu=1, prefix_dims=1, sites_before_link=True)
    W_scalar = W[0, :, :, 0, 0]
    print("W_11 (integers; adjoint=identity for real scalars):")
    print(W_scalar)

    # Manual check at one site
    i, j = 0, 0
    U_mu = Ux[i,j]
    U_nu_xplusmu = Uy[(i+1)%Lx, j]
    U_mu_xplusnu = Ux[i, (j+1)%Ly]
    U_nu = Uy[i,j]
    manual = (U_mu * U_nu_xplusmu) * (U_nu * U_mu_xplusnu).conj()
    print(f"\nManual check at ({i},{j}):")
    print(f"U_mu(x)={U_mu}, U_nu(x+mu)={U_nu_xplusmu}, U_mu(x+nu)={U_mu_xplusnu}, U_nu(x)={U_nu}")
    print(f"W_11({i},{j}) manual = {manual} ; from function = {W_scalar[i,j]}")

def demo_u1_phases():
    print("\n######## Demo B: U(1) phase-valued links (unit modulus) ########")
    torch.set_printoptions(precision=3, sci_mode=False)
    B, Lx, Ly, D, Nc = 1, 4, 4, 2, 1
    x = torch.zeros((B, Lx, Ly, D, Nc, Nc), dtype=torch.complex64)

    # Ux(x,y) = exp(i * (ax*x + bx*y)), Uy(x,y) = exp(i * (cx*x + dy*y))
    ax, bx, cx, dy = 0.20, 0.10, -0.05, 0.15
    for i in range(Lx):
        for j in range(Ly):
            theta_x = ax*i + bx*j
            theta_y = cx*i + dy*j
            x[0, i, j, 0, 0, 0] = complex(math.cos(theta_x), math.sin(theta_x))  # Ux
            x[0, i, j, 1, 0, 0] = complex(math.cos(theta_y), math.sin(theta_y))  # Uy

    Ux = x[0, :, :, 0, 0, 0]
    Uy = x[0, :, :, 1, 0, 0]
    print_lattice_ascii_Ux_vertical_Uy_horizontal(
        Ux, Uy, "FINE lattice (real part of U(1) links shown)"
    )

    W = compute_wilson_1x1_loop(x, mu=0, nu=1, prefix_dims=1, sites_before_link=True)
    Wc = W[0, :, :, 0, 0]
    print("W_11 (complex U(1) values):")
    print(Wc)

    mag = torch.abs(Wc)
    phase = torch.angle(Wc)
    print("\n|W_11|:")
    print(mag)
    print("\narg(W_11) [radians]:")
    print(phase)

# Run both demos
demo_integer_links()
demo_u1_phases()
