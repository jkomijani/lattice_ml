r"""
Jacobian of the spectral-twist map on SU(N).

    U = Q Lambda Q^dagger  (eigendecomposition, eigenvalues sorted by argument)
    Z = Q Lambda Q^dagger Lambda^dagger = U Lambda^dagger

In matched left-invariant frames

    d eta = Q^dagger U^dagger dU Q ,    d xi = Q^dagger Z^dagger dZ Q

the Jacobian is independent of Lambda and equals

    |det(d xi / d eta)| = det(I - B) restricted to  { h : sum_m h_m = 0 },
    B[k, j] = |Q[j, k]|^2      (doubly stochastic)

            = prod_{k=2}^{N} (1 - mu_k); mu_1 = 1, mu_2..mu_N = rest of spec(B)
            = sum_{k=1}^{N} det[(I - B) with row k and column k deleted]
            = N * det[(I - B) with row 1 and column 1 deleted].

The eigenangles themselves never appear. What does matter is the ORDERING RULE:
B pairs eigenvector index k with diagonal slot j, so Q must be the eigenvector
matrix whose columns are already in the chosen (sorted-by-argument) order.
A different ordering gives a different Z and a different J.

All functions take Q (or U) with an arbitrary leading batch shape (..., N, N)
and return a real tensor of shape (...).
"""

# pylint: disable=invalid-name  # for matrices like Q, U, B and dims like N

import torch

__all__ = [
    "log_jacobian_sun",
    "log_jacobian_su2",
    "log_jacobian_su3",
]


# ----------------------------------------------------------------------
# B matrix
# ----------------------------------------------------------------------
def bistochastic_from_Q(Q: torch.Tensor) -> torch.Tensor:
    """B[..., j, k] = |Q[..., j, k]|^2.

    In probability and combinatorics, a doubly stochastic matrix (also called
    bistochastic matrix) is a square matrix with nonnegative entries whose rows
    and columns each sum to 1.
    """
    return (Q.real**2 + Q.imag**2) if Q.is_complex() else Q**2


# ----------------------------------------------------------------------
# General SU(N)
# ----------------------------------------------------------------------
def jacobian_sun(Q: torch.Tensor) -> torch.Tensor:
    """|det dZ/dU| for SU(N), any N >= 2. Q: (..., N, N) complex."""
    B = bistochastic_from_Q(Q)
    N = B.shape[-1]
    V = _helmert(N, B.dtype, B.device)
    eye = torch.eye(N, dtype=B.dtype, device=B.device)
    M = V.transpose(-1, -2) @ (eye - B) @ V  # (..., N-1, N-1)
    return torch.linalg.det(M)


def log_jacobian_sun(Q: torch.Tensor) -> torch.Tensor:
    """log |det dZ/dU|, computed via slogdet (stable for near-singular Q)."""
    B = bistochastic_from_Q(Q)
    N = B.shape[-1]
    V = _helmert(N, B.dtype, B.device)
    eye = torch.eye(N, dtype=B.dtype, device=B.device)
    M = V.transpose(-1, -2) @ (eye - B) @ V  # (..., N-1, N-1)
    _, logabsdet = torch.linalg.slogdet(M)
    return logabsdet


def _helmert(N: int, dtype: torch.dtype, device) -> torch.Tensor:
    """N x (N-1) matrix with orthonormal columns spanning {h : sum h = 0}."""
    V = torch.zeros(N, N - 1, dtype=dtype, device=device)
    for k in range(1, N):
        V[:k, k - 1] = 1.0 / (k * (k + 1.0)) ** 0.5
        V[k, k - 1] = -(k / (k + 1.0)) ** 0.5
    return V


# ----------------------------------------------------------------------
# Closed forms for N = 2, 3
# ----------------------------------------------------------------------
def jacobian_su2(Q: torch.Tensor) -> torch.Tensor:
    """SU(2): |det dZ/dU| = 2 |Q_12|^2 = 2 (1 - |Q_11|^2)."""
    q01 = Q[..., 0, 1]
    return 2.0 * (q01.real**2 + q01.imag**2)


def log_jacobian_su2(Q: torch.Tensor) -> torch.Tensor:
    """SU(2): log |det dZ/dU| = log(2 |Q_12|^2) = log(2 (1 - |Q_11|^2))."""
    q01 = Q[..., 0, 1]
    return torch.log(2.0 * (q01.real**2 + q01.imag**2))


def jacobian_su3(Q: torch.Tensor) -> torch.Tensor:
    """SU(3): |det dZ/dU| = 2 - sum_k |Q_kk|^2 + det[|Q_ij|^2]."""
    P = Q.real**2 + Q.imag**2  # P[..., i, j] = |Q_ij|^2
    e1 = torch.einsum('...ii -> ...', P)
    e3 = torch.linalg.det(P)
    return 2.0 - e1 + e3


def log_jacobian_su3(Q: torch.Tensor) -> torch.Tensor:
    """SU(3): log |det dZ/dU| = log(2 - sum_k |Q_kk|^2 + det[|Q_ij|^2])."""
    P = Q.real**2 + Q.imag**2  # P[..., i, j] = |Q_ij|^2
    e1 = torch.einsum('...ii -> ...', P)
    e3 = torch.linalg.det(P)
    return torch.log(2.0 - e1 + e3)


# ----------------------------------------------------------------------
# From U directly
# ----------------------------------------------------------------------
def eigendecompose_sorted(U: torch.Tensor):
    """
    Eigendecomposition of unitary U with eigenvalues sorted by ascending
    argument.

    Returns (lam, Q) with U = Q diag(lam) Q^dagger.
    """
    lam, Q = torch.linalg.eig(U)
    idx = torch.argsort(torch.angle(lam), dim=-1)
    lam = torch.gather(lam, -1, idx)
    Q = torch.gather(Q, -1, idx.unsqueeze(-2).expand_as(Q))
    return lam, Q


def twist_eigenvalues(U: torch.Tensor) -> torch.Tensor:
    """Z = U Lambda^dagger."""
    lam, _ = eigendecompose_sorted(U)
    return U @ torch.diag_embed(lam.conj())


def jacobian_from_U(U: torch.Tensor) -> torch.Tensor:
    """|det dZ/dU| computed straight from U."""
    _, Q = eigendecompose_sorted(U)
    return jacobian_sun(Q)


# ======================================================================
# self-test helpers
# ======================================================================
def _haar_sun(N, batch, dtype=torch.complex128):
    """Batch of Haar-random SU(N) matrices."""
    z = torch.randn(batch, N, N, dtype=dtype) / 2 ** 0.5
    q, r = torch.linalg.qr(z)
    phase = torch.diagonal(r, dim1=-2, dim2=-1)
    q = q * (phase / phase.abs()).unsqueeze(-2)
    return q * torch.linalg.det(q).unsqueeze(-1).unsqueeze(-1) ** (-1.0 / N)


def _sun_basis(N, cdt=torch.complex128, rdt=torch.float64):
    """Orthonormal basis of su(N): off-diagonal pairs plus Helmert diagonal."""
    basis = []
    for i in range(N):
        for j in range(i + 1, N):
            E = torch.zeros(N, N, dtype=cdt)
            E[i, j], E[j, i] = 1, -1
            basis.append(E / 2 ** 0.5)

            E = torch.zeros(N, N, dtype=cdt)
            E[i, j], E[j, i] = 1j, 1j
            basis.append(E / 2 ** 0.5)
    V = _helmert(N, rdt, None)
    for c in range(N - 1):
        basis.append(1j * torch.diag(V[:, c]).to(cdt))
    return torch.stack(basis)


def _fd_logjac(U, eps=1e-6):
    """FD estimate of log|det d(Z^dag dZ)/d(U^dag dU)| in an su(N) frame."""
    N = U.shape[-1]
    basis = _sun_basis(N)
    Z0 = twist_eigenvalues(U.unsqueeze(0))[0]
    rows = []
    for a in range(basis.shape[0]):
        Up = U @ torch.matrix_exp(eps * basis[a])
        Um = U @ torch.matrix_exp(-eps * basis[a])
        dZ = (twist_eigenvalues(Up.unsqueeze(0))[0]
              - twist_eigenvalues(Um.unsqueeze(0))[0]) / (2 * eps)
        X = Z0.conj().T @ dZ
        row = [(basis[b].conj().T @ X).diagonal().sum().real
               for b in range(basis.shape[0])]
        rows.append(torch.stack(row))
    return torch.linalg.slogdet(torch.stack(rows))[1]


def run_self_tests():  # pylint: disable=too-many-locals,too-many-statements
    """Sanity-check the spectral-twist Jacobian against known formulas."""
    torch.manual_seed(0)
    cdt, rdt = torch.complex128, torch.float64
    tol, fd_tol, mean_tol = 1e-10, 1e-6, 1e-2

    for N in (2, 3, 4, 5):
        U = _haar_sun(N, 6)
        lam, Q = eigendecompose_sorted(U)

        Urec = Q @ torch.diag_embed(lam) @ Q.conj().transpose(-1, -2)
        rec = float((Urec - U).abs().max())
        assert rec < tol, f"N={N}: reconstruction error {rec}"

        Z = twist_eigenvalues(U)
        ZZd = Z @ Z.conj().transpose(-1, -2)
        uni = float((ZZd - torch.eye(N, dtype=cdt)).abs().max())
        det1 = float((torch.linalg.det(Z) - 1).abs().max())
        assert uni < tol, f"N={N}: Z not unitary, err={uni}"
        assert det1 < tol, f"N={N}: det(Z) != 1, err={det1}"

        Jg = jacobian_sun(Q)
        eye = torch.eye(N, dtype=rdt)
        Jm = N * torch.linalg.det((eye - bistochastic_from_Q(Q))[..., 1:, 1:])
        minor_err = float((Jg - Jm).abs().max())
        assert minor_err < tol, f"N={N}: minor-form mismatch {minor_err}"

        from_u_err = float((Jg - jacobian_from_U(U)).abs().max())
        assert from_u_err < tol, f"N={N}: from-U mismatch {from_u_err}"

        assert bool((Jg > 0).all()), f"N={N}: non-positive Jacobian"

        fd = torch.stack([_fd_logjac(U[b]) for b in range(3)])
        fd_err = float((fd - torch.log(Jg[:3])).abs().max())
        assert fd_err < fd_tol, f"N={N}: finite-difference mismatch {fd_err}"

        print(f"N={N}  recon={rec:.1e}  Z unitary={uni:.1e}  "
              f"det Z-1={det1:.1e}  minor-form={minor_err:.1e}  "
              f"from-U={from_u_err:.1e}  fd={fd_err:.1e}")

    U = _haar_sun(2, 6)
    _, Q = eigendecompose_sorted(U)
    su2_err = float((jacobian_sun(Q) - jacobian_su2(Q)).abs().max())
    assert su2_err < tol, f"su2 closed form mismatch {su2_err}"
    print("SU(2) closed-form mismatch:", su2_err)

    U = _haar_sun(3, 6)
    _, Q = eigendecompose_sorted(U)
    su3_err = float((jacobian_sun(Q) - jacobian_su3(Q)).abs().max())
    assert su3_err < tol, f"su3 closed form mismatch {su3_err}"
    print("SU(3) closed-form mismatch:", su3_err)

    # Lambda-independence WITHIN a fixed ordering cell: keep the eigenangles
    # sorted ascending so the sort rule returns the same Q columns.
    N = 3
    Q = _haar_sun(N, 4)
    Jref = jacobian_sun(Q)
    for _ in range(3):
        th, _ = torch.sort(torch.rand(4, N, dtype=rdt) * 2 - 1, dim=-1)
        th = th - th.mean(-1, keepdim=True)
        th, _ = torch.sort(th, dim=-1)
        Lam = torch.diag_embed(torch.exp(1j * th.to(cdt)))
        U = Q @ Lam @ Q.conj().transpose(-1, -2)
        lam_err = float((jacobian_from_U(U) - Jref).abs().max())
        assert lam_err < tol, f"lambda-independence residual {lam_err}"
        print("lambda-independence residual:", lam_err)

    # ordering DOES matter (Z changes if eigenvalues go to other slots)
    U = _haar_sun(2, 1)
    _, Q = eigendecompose_sorted(U)
    J, J_flip = float(jacobian_sun(Q)), float(jacobian_sun(Q.flip(-1)))
    assert abs(J + J_flip - 2) < tol, (J, J_flip)
    print("SU(2) sorted vs flipped ordering:", J, J_flip, "  (sum = 2)")

    # E[J] = 1  (map is a.e. one-to-one)
    for N in (2, 3, 4):
        Q = _haar_sun(N, 200000)
        J = jacobian_sun(Q)
        mean, sem = float(J.mean()), float(J.std() / 200000 ** 0.5)
        assert abs(mean - 1) < mean_tol, f"N={N}: E[J]={mean}"
        print(f"N={N}  E[J] = {mean:.4f} +- {sem:.4f}")


if __name__ == "__main__":
    run_self_tests()
