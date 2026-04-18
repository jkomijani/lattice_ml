import torch


from svd_result import SVDResult, SolveLambdaSUN


torch.set_default_dtype(torch.float64)

# ----------------------------
# 1. Reconstruction test
# ----------------------------
def test_svd_reconstruction(svd_result):
    M_rec = svd_result.reconstruct()
    assert torch.allclose(M, M_rec, atol=1e-10), "1: SVD reconstruction failed"


# ----------------------------
# 2. Unitary factor test
# ----------------------------
def test_unitary_factor(svd_result):
    Q = svd_result.unitary_factor

    I = Q @ Q.adjoint()
    eye = torch.eye(Q.shape[-1], dtype=Q.dtype)

    assert torch.allclose(I, eye, atol=1e-10), "2: Unitary factor failed"


# ----------------------------
# 3. SU(n) constraint test
# ----------------------------
def test_special_unitary(svd_result):
    X = svd_result.special_unitary_factor
    D = svd_result.diagonal_phase_factor
    S = svd_result.S

    I = X @ X.adjoint()
    eye = torch.eye(X.shape[-1], dtype=X.dtype)

    det = torch.det(X)

    assert torch.allclose(I, eye, atol=1e-10), "3: Not unitary"
    assert torch.allclose(det, torch.ones_like(det), atol=1e-10), "3: Det != 1"


# ----------------------------
# 4. Autograd sanity test
# ----------------------------
def test_lambda_grad():
    s = torch.rand(4, requires_grad=True)
    theta = torch.tensor(0.3, requires_grad=True)

    lam = SolveLambdaSUN.apply(s, theta)
    loss = lam**2
    loss.backward()

    assert torch.isfinite(s.grad).all(), "4: NaN in grad_s"
    assert torch.isfinite(theta.grad), "4: NaN in grad_theta"


# ----------------------------
# 5. Finite difference check (gold standard)
# ----------------------------
def test_lambda_grad_fd():
    eps = 1e-6

    s0 = torch.rand(3)
    theta0 = torch.tensor(0.3)

    s0.requires_grad_()

    # autograd
    lam = SolveLambdaSUN.apply(s0, theta0)
    lam.backward()
    grad_auto = s0.grad.clone()

    # finite diff
    grad_fd = torch.zeros_like(s0)

    for i in range(len(s0)):
        s_plus = s0.clone()
        s_minus = s0.clone()

        s_plus[i] += eps
        s_minus[i] -= eps

        lam_plus = SolveLambdaSUN.apply(s_plus, theta0)
        lam_minus = SolveLambdaSUN.apply(s_minus, theta0)

        grad_fd[i] = (lam_plus - lam_minus) / (2 * eps)

    assert torch.allclose(grad_auto, grad_fd, atol=1e-10), "5: FD mismatch"


def test_imaginary_part_DS(svd_result):
    """
    Checks the key SU(n) optimality condition:

        Im(D @ diag(S)) = λ I
    """
    D = svd_result.diagonal_phase_factor

    imag = torch.imag(D * S)

    # expected: λ I, recover λ as average diagonal imaginary part
    lam_est = torch.mean(imag, dim=-1)

    identity = torch.ones(S.shape[-1], dtype=imag.dtype).expand_as(imag)

    target = lam_est[..., None] * identity

    assert torch.allclose(imag, target, atol=1e-10), "6: Im(D S) ≠ λ I"

# ----------------------------
# Run all tests
# ----------------------------
if __name__ == "__main__":
    M = torch.randn(3, 3, dtype=torch.complex128)
    U, S, Vh = torch.linalg.svd(M)
    svd_result = SVDResult(U=U, S=S, Vh=Vh)
    test_svd_reconstruction(svd_result)
    test_unitary_factor(svd_result)
    test_special_unitary(svd_result)
    test_lambda_grad()
    test_lambda_grad_fd()
    test_imaginary_part_DS(svd_result)

    print("All tests passed ")
