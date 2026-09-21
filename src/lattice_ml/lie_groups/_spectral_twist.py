r"""
Jacobian of the spectral-twist map on SU(N).

    U = W Lambda W^dagger  (eigendecomposition, eigenvalues sorted by argument)
    Z_tilde = W Lambda W^dagger Lambda^dagger = U Lambda^dagger

In matched left-invariant frames

    d eta = W^dagger U^dagger dU W ,
    d xi  = W^dagger Z_tilde^dagger dZ_tilde W

the Jacobian is independent of Lambda and equals

    |det(d xi / d eta)| = det(I - B) restricted to  { h : sum_m h_m = 0 },
    B[k, j] = |W[j, k]|^2      (doubly stochastic)

            = prod_{k=2}^{N} (1 - mu_k); mu_1 = 1, mu_2..mu_N = rest of spec(B)
            = sum_{k=1}^{N} det[(I - B) with row k and column k deleted]
            = N * det[(I - B) with row 1 and column 1 deleted].

The eigenangles themselves never appear. What does matter is the ORDERING RULE:
B pairs eigenvector index k with diagonal slot j, so W must be the eigenvector
matrix whose columns are already in the chosen (sorted-by-argument) order.
A different ordering gives a different Z_tilde and a different J.

All functions take W (or U) with an arbitrary leading batch shape (..., N, N)
and return a real tensor of shape (...).


Relation to `_sun_group_commutator`
-----------------------------------
Two parallel problems reach the same matrix by different routes:

    spectral twist (this module):    U  ->  Z_tilde = U Lambda_U^dagger
    group commutator (that module):    (X, Y)  ->  Z = X Y X^dagger Y^dagger

where Z_tilde = Q^dagger Z Q, with Q defined below.

They are different maps with different inputs, and the bridge is conjugation
into Y's eigenframe with Q the eigenframe of Y. The computation of log-Jacobian
is in this module.

The word "twist" belongs to this module only: there is no twisting in the
group-commutator problem, and `_sun_group_commutator` speaks of conjugation.
"""

# pylint: disable=invalid-name  # for matrices like W, U, B and dims like N

import torch


__all__ = [
    "log_jacobian_sun",
    "log_jacobian_su2",
    "log_jacobian_su3",
]


# ----------------------------------------------------------------------
# B matrix
# ----------------------------------------------------------------------
def bistochastic_from_W(W: torch.Tensor) -> torch.Tensor:
    """B[..., k, j] = |W[..., j, k]|^2.

    In probability and combinatorics, a doubly stochastic matrix (also called
    bistochastic matrix) is a square matrix with nonnegative entries whose rows
    and columns each sum to 1.
    """
    # For consistency with docstring, we use W^T; it does not change det(I - B)
    Q = torch.transpose(W, -1, -2)
    return (Q.real**2 + Q.imag**2) if Q.is_complex() else Q**2


# ----------------------------------------------------------------------
# General SU(N)
# ----------------------------------------------------------------------
def jacobian_sun(W: torch.Tensor) -> torch.Tensor:
    """|det dZ_tilde/dU| for SU(N), any N >= 2. W: (..., N, N) complex."""
    B = bistochastic_from_W(W)
    N = B.shape[-1]
    V = _helmert(N, B.dtype, B.device)
    eye = torch.eye(N, dtype=B.dtype, device=B.device)
    M = V.transpose(-1, -2) @ (eye - B) @ V  # (..., N-1, N-1)
    return torch.linalg.det(M)


def log_jacobian_sun(W: torch.Tensor) -> torch.Tensor:
    """log |det dZ_tilde/dU|, via slogdet (stable for near-singular W)."""
    B = bistochastic_from_W(W)
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
def jacobian_su2(W: torch.Tensor) -> torch.Tensor:
    """SU(2): |det dZ_tilde/dU| = 2 |W_12|^2 = 2 (1 - |W_11|^2)."""
    w01 = W[..., 0, 1]
    return 2.0 * (w01.real**2 + w01.imag**2)


def log_jacobian_su2(W: torch.Tensor) -> torch.Tensor:
    """SU(2): log |det dZ_tilde/dU| = log(2 |W_12|^2) = log(2 - 2|W_11|^2)."""
    w01 = W[..., 0, 1]
    return torch.log(2.0 * (w01.real**2 + w01.imag**2))


def jacobian_su3(W: torch.Tensor) -> torch.Tensor:
    """SU(3): |det dZ_tilde/dU| = 2 - sum_k |W_kk|^2 + det[|W_ij|^2]."""
    P = W.real**2 + W.imag**2  # P[..., i, j] = |W_ij|^2
    e1 = torch.einsum('...ii -> ...', P)
    e3 = torch.linalg.det(P)
    return 2.0 - e1 + e3


def log_jacobian_su3(W: torch.Tensor) -> torch.Tensor:
    """SU(3): log |det dZ_tilde/dU| = log(2 - sum|W_kk|^2 + det[|W_ij|^2])."""
    P = W.real**2 + W.imag**2  # P[..., i, j] = |W_ij|^2
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

    Returns (lam, W) with U = W diag(lam) W^dagger.
    """
    lam, W = torch.linalg.eig(U)
    idx = torch.argsort(torch.angle(lam), dim=-1)
    lam = torch.gather(lam, -1, idx)
    W = torch.gather(W, -1, idx.unsqueeze(-2).expand_as(W))
    return lam, W


def twist_eigenvalues(U: torch.Tensor) -> torch.Tensor:
    """Z_tilde = U Lambda^dagger."""
    lam, _ = eigendecompose_sorted(U)
    return U @ torch.diag_embed(lam.conj())


def jacobian_from_U(U: torch.Tensor) -> torch.Tensor:
    """|det dZ_tilde/dU| computed straight from U."""
    _, W = eigendecompose_sorted(U)
    return jacobian_sun(W)


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
    """FD estimate of log|det d(Z_tilde^dag dZ_tilde)/d(U^dag dU)|."""
    N = U.shape[-1]
    basis = _sun_basis(N)
    Z_tilde0 = twist_eigenvalues(U.unsqueeze(0))[0]
    rows = []
    for a in range(basis.shape[0]):
        Up = U @ torch.matrix_exp(eps * basis[a])
        Um = U @ torch.matrix_exp(-eps * basis[a])
        dZ_tilde = (twist_eigenvalues(Up.unsqueeze(0))[0]
                    - twist_eigenvalues(Um.unsqueeze(0))[0]) / (2 * eps)
        X = Z_tilde0.conj().T @ dZ_tilde
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
        lam, W = eigendecompose_sorted(U)

        Urec = W @ torch.diag_embed(lam) @ W.conj().transpose(-1, -2)
        rec = float((Urec - U).abs().max())
        assert rec < tol, f"N={N}: reconstruction error {rec}"

        Z_tilde = twist_eigenvalues(U)
        ZZd = Z_tilde @ Z_tilde.conj().transpose(-1, -2)
        uni = float((ZZd - torch.eye(N, dtype=cdt)).abs().max())
        det1 = float((torch.linalg.det(Z_tilde) - 1).abs().max())
        assert uni < tol, f"N={N}: Z_tilde not unitary, err={uni}"
        assert det1 < tol, f"N={N}: det(Z_tilde) != 1, err={det1}"

        Jg = jacobian_sun(W)
        eye = torch.eye(N, dtype=rdt)
        Jm = N * torch.linalg.det((eye - bistochastic_from_W(W))[..., 1:, 1:])
        minor_err = float((Jg - Jm).abs().max())
        assert minor_err < tol, f"N={N}: minor-form mismatch {minor_err}"

        from_u_err = float((Jg - jacobian_from_U(U)).abs().max())
        assert from_u_err < tol, f"N={N}: from-U mismatch {from_u_err}"

        assert bool((Jg > 0).all()), f"N={N}: non-positive Jacobian"

        fd = torch.stack([_fd_logjac(U[b]) for b in range(3)])
        fd_err = float((fd - torch.log(Jg[:3])).abs().max())
        assert fd_err < fd_tol, f"N={N}: finite-difference mismatch {fd_err}"

        print(f"N={N}  recon={rec:.1e}  Z_tilde unitary={uni:.1e}  "
              f"det Z_tilde-1={det1:.1e}  minor-form={minor_err:.1e}  "
              f"from-U={from_u_err:.1e}  fd={fd_err:.1e}")

    U = _haar_sun(2, 6)
    _, W = eigendecompose_sorted(U)
    su2_err = float((jacobian_sun(W) - jacobian_su2(W)).abs().max())
    assert su2_err < tol, f"su2 closed form mismatch {su2_err}"
    print("SU(2) closed-form mismatch:", su2_err)

    U = _haar_sun(3, 6)
    _, W = eigendecompose_sorted(U)
    su3_err = float((jacobian_sun(W) - jacobian_su3(W)).abs().max())
    assert su3_err < tol, f"su3 closed form mismatch {su3_err}"
    print("SU(3) closed-form mismatch:", su3_err)

    # Lambda-independence WITHIN a fixed ordering cell: keep the eigenangles
    # sorted ascending so the sort rule returns the same W columns.
    N = 3
    W = _haar_sun(N, 4)
    Jref = jacobian_sun(W)
    for _ in range(3):
        th, _ = torch.sort(torch.rand(4, N, dtype=rdt) * 2 - 1, dim=-1)
        th = th - th.mean(-1, keepdim=True)
        th, _ = torch.sort(th, dim=-1)
        Lam = torch.diag_embed(torch.exp(1j * th.to(cdt)))
        U = W @ Lam @ W.conj().transpose(-1, -2)
        lam_err = float((jacobian_from_U(U) - Jref).abs().max())
        assert lam_err < tol, f"lambda-independence residual {lam_err}"
        print("lambda-independence residual:", lam_err)

    # ordering DOES matter (Z_tilde changes if eigenvalues go to other slots)
    U = _haar_sun(2, 1)
    _, W = eigendecompose_sorted(U)
    J, J_flip = float(jacobian_sun(W)), float(jacobian_sun(W.flip(-1)))
    assert abs(J + J_flip - 2) < tol, (J, J_flip)
    print("SU(2) sorted vs flipped ordering:", J, J_flip, "  (sum = 2)")

    # E[J] = 1  (map is a.e. one-to-one)
    for N in (2, 3, 4):
        W = _haar_sun(N, 200000)
        J = jacobian_sun(W)
        mean, sem = float(J.mean()), float(J.std() / 200000 ** 0.5)
        assert abs(mean - 1) < mean_tol, f"N={N}: E[J]={mean}"
        print(f"N={N}  E[J] = {mean:.4f} +- {sem:.4f}")


if __name__ == "__main__":
    run_self_tests()
