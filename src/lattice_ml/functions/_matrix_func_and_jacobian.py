# Copyright (c) 2024 Javad Komijani

"""For computation of matrix functions and their Jacobian."""

from abc import abstractmethod, ABC
import torch


from lattice_ml.linalg import eigh, eigu, inverse_eign, reciprocal


# =============================================================================
class MatrixFunctionTemplate(ABC):
    r"""
    Template class for handling a matrix transformation as :math:`f(M)`, where
    the matrix :math:`M` is supposed to be normal.

    Exploiting the spectral decomposition of :math:`H`, we have

    .. math::

        F = \Omega f(\Lambda) \Omega^\dagger
       dF = \Omega (df(\Lambda) + [d\Gamma, \Lambda]) \Omega^\dagger

    where:
      - :math:`[A, B] = AB - BA` denotes the commutator,
      - :math:`d\Gamma = \Omega^\dagger d\Omega` is anti-Hermitian,
      - :math:`df(\Lambda)` is elementwise derivative applied to eigenvalues.

    Subclasses must implement :meth:`scalar_func`, which:
      1. Applies the scalar function :math:`f` to eigenvalues.
      2. Returns both the transformed eigenvalues and their derivatives.

    Notes
    -----
    - This implementation assumes distinct (or well-behaved) eigenvalues for
      numerical stability in the Jacobian computation.
    """

    # The backend function used for eigendecomposition in forward mode.
    forward_mode_eig = torch.linalg.eig

    def __call__(self, matrix):
        """Alias for :meth:`forward`."""
        return self.forward(matrix)

    def forward(self, matrix):
        """
        Compute the matrix function and its Jacobian.

        Parameters
        ----------
        Args:
            matrix (torch.Tensor): Input square matrix (assumed normal).

        Returns:
            F (torch.Tensor): The transformed matrix :math:`f(M)`.
            J (torch.Tensor): The Jacobian of the transformation.
        """
        # Eigen-decomposition: M = V diag(vals) V^{-1}
        vals, vecs = self.forward_mode_eig(matrix)

        # Apply scalar function to eigenvalues and get derivatives
        f_vals, f_prime = self.scalar_func(vals)

        # Reconstruct f(M) = V diag(f_vals) V^{-1}
        matrix = inverse_eign(f_vals, vecs)

        return matrix, self.calc_jacobian_matrix(vals, vecs, f_vals, f_prime)

    def calc_jacobian_matrix(self, eigvals, eigvecs, f_eigvals, f_prime):
        r"""
        Construct the Jacobian of the matrix function in vectorized form.

        This uses the standard formula for matrix functions based on
        eigenvalue differences:

        .. math::

            (f(\lambda_i) - f(\lambda_j)) / (\lambda_i - \lambda_j)

        with diagonal entries replaced by :math:`f'(\lambda_i)`.

        Args:
            eigvals (torch.Tensor): Eigenvalues of the input matrix.
            eigvecs (torch.Tensor): Corresponding eigenvectors.
            f_eigvals (torch.Tensor): Transformed eigenvalues.
            f_prime (torch.Tensor): Derivatives :math:`f'(\lambda_i)`.

        Returns:
            J (torch.Tensor): The Jacobian matrix.
        """
        # Reciprocal of pairwise eigenvalue differences: 1 / (λ_i - λ_j)
        nabla = reciprocal(calc_eig_delta(eigvals))

        # Pairwise differences of transformed eigenvalues: f(λ_i) - f(λ_j)
        delta_f = calc_eig_delta(f_eigvals)

        # Off-diagonal entries use (f(λ_i) - f(λ_j)) / (λ_i - λ_j)
        # Diagonal entries replaced with f'(λ_i)
        mat = torch.diag_embed(f_prime) + delta_f * nabla

        # Flatten into block-diagonal form
        jac1 = torch.diag_embed(mat.reshape(*nabla.shape[:-2], -1))

        # Change of basis using Kronecker product of eigenvectors
        jac2 = kronecker_product(eigvecs, eigvecs.conj())

        return jac2 @ jac1 @ jac2.adjoint()

    @abstractmethod
    def scalar_func(self, eigvals):
        r"""
        Apply the scalar function to eigenvalues.

        Args:
            eigvals (torch.Tensor): Eigenvalues of the input matrix.

        Returns:
            f_eigvals (torch.Tensor): Transformed eigenvalues.
            f_prime (torch.Tensor): Derivatives :math:`f'(\lambda_i)`.
        """


class MatrixExp1jh(MatrixFunctionTemplate):
    """Compute `U = exp(iH)` for Hermitian matrices H."""

    forward_mode_eig = eigh

    def scalar_func(self, eigvals):
        """Spectral map for `f(λ) = exp(iλ)` and its derivative."""
        f_eigvals = torch.exp(1j * eigvals)
        fp_eigvals = 1j * f_eigvals
        return f_eigvals, fp_eigvals


class MatrixAngleU(MatrixFunctionTemplate):
    """Compute `H = -i log(U)` for unitary matrix U."""

    forward_mode_eig = eigu

    def scalar_func(self, eigvals):
        """Spectral map for `f(λ) = -i log(λ)` and its derivative."""
        f_eigvals = torch.angle(eigvals)
        fp_eigvals = -1j * eigvals.conj()
        return f_eigvals, fp_eigvals


# =============================================================================
def inverse_eign_and_jacobian(eigvals, eigvecs, mode='Gamma'):
    r"""
    Reconstruct the matrix

    .. math::

        M = \Omega \Lambda \Omega^\dagger

    from unitary eigenvectors :math:`\Omega` and corresponding eigenvalues,
    and compute the Jacobian under different parameterizations.

    We use the differential parameterization:

    .. math::

        d\Gamma = \Omega^\dagger d\Omega

    which is anti-Hermitian. We set its diagonal terms to zero, fixing
    redundancy in :math:`\Omega`.

    Modes
    -----
    mode = 'Gamma' (default):
        Jacobian w.r.t. :math:`\Gamma` only (eigenvalues fixed).

    mode = 'Full':
        Jacobian w.r.t. both :math:`[\Lambda, \Gamma]`.

    mode = 'Omega':
        Jacobian w.r.t. :math:`\Omega` directly (no redundancy removed).

    Notes
    -----
    - 'Gamma' and 'Omega' modes yield singular Jacobians:
        * 'Gamma': eigenvalues are fixed
        * 'Omega': unitary redundancy is not removed
    - 'Full' provides a non-redundant parameterization (up to degeneracies).

    Args:
        eigvals (torch.Tensor): Eigenvalues, i.e. diagonals of :math:`\Lambda`.
        eigvecs (torch.Tensor): Eigenvectors forming :math:`\Omega`.
        mode (str): One of {'Gamma', 'Full', 'Omega'}.

    Returns:
        M (torch.Tensor): Reconstructed matrix.
        J (torch.Tensor): Jacobian in vectorized (Kronecker) form.
    """
    # Reconstruct M = Ω Λ Ω†
    matrix = inverse_eign(eigvals, eigvecs)

    # Pairwise eigenvalue differences (λᵢ − λⱼ)
    delta = calc_eig_delta(eigvals)

    # Identity (same shape as delta)
    eye = eyes_like(delta)

    # Basis transform: Ω ⊗ Ω*
    jac2 = kronecker_product(eigvecs, eigvecs.conj())

    shape = [*delta.shape[:-2], -1]

    match mode:
        case 'Gamma':
            # Only non-diagonal terms from the commutator: [dΓ, Λ]
            jac1 = torch.diag_embed(delta.reshape(*shape))  # det(jac1) = 0

        case 'Full':
            # Includes both dΛ (diagonal) and commutator terms
            jac1 = torch.diag_embed((eye + delta).reshape(*shape))

        case 'Omega':
            # Parameterization directly in Ω (redundant)
            jac1 = torch.diag_embed(delta.reshape(*shape))  # det(jac1) = 0
            jac1 = jac1 @ kronecker_product(eigvecs.adjoint(), eye)

    return matrix, jac2 @ jac1


def commutator_and_jacobian(mat1, mat2):
    """
    Compute the commutator and its Jacobian w.r.t. the first argument.

    The commutator of two square matrices P and Q is defined as:

        [P, Q] = PQ − QP

    Using the identity:

        vec(PQ − QP) = (I ⊗ Qᵀ − Q ⊗ I) vec(P),

    where `vec(P)_{iN + j} = P_{ij}` with indices starting from 0, the Jacobian
    is:

        J = I ⊗ Qᵀ − Q ⊗ I

    To prove the above identity, one can use

        (AB)_{ij} = A_{il} B_{lk} I_{jk} = (A ⊗ I)_{iN+j, lN+k} B_{lk}
        (AB)_{ij} = I_{il} A_{lk} B_{kj} = (I ⊗ Bᵀ)_{iN+j, lN+k} A_{lk}

    that can be written as

        vec(AB)_{iN+j} = (A ⊗ I)_{iN+j, lN+k} vec(B)_{lN+k}
        vec(AB)_{iN+j} = (I ⊗ Bᵀ)_{iN+j, lN+k} vec(A)_{lN+k}

    Interpretation (real vs complex, constrained spaces)
    ----------------------------------------------------
    - The Jacobian is algebraically valid for both real and complex matrices.

    - For complex matrices, the full Jacobian depends on the intended notion of
      derivative: complex (holomorphic) view vs real view, where there are two
      real degrees of freedom for each complex variable.

    - For constrained matrices (e.g., symmetric, Hermitian), the true degrees
      of freedom are reduced. Then, quantities like det(J) must be computed on
      a restricted Jacobian to avoide double-counted volume factors.

    - This function was primarily developed with Hermitian P and Q in mind;
      care is required when applying it outside this setting.

    Args:
        mat1 (torch.Tensor): Tensor P of shape (..., n, n).
        mat2 (torch.Tensor): Tensor Q of shape (..., n, n).

    Returns:
        torch.Tensor: The commutator [P, Q].
        torch.Tensor: Jacobian ∂vec([P, Q]) / ∂vec(P).
    """
    # [P, Q] = PQ − QP
    mat = mat1 @ mat2 - mat2 @ mat1

    # Identity and transpose needed for vec identity
    eye = eyes_like(mat1)
    mat2_t = mat2.transpose(-2, -1)

    # J = I ⊗ Qᵀ − Q ⊗ I
    jac = kronecker_product(eye, mat2_t) - kronecker_product(mat2, eye)

    return mat, jac


# =============================================================================
def kronecker_product(mat1, mat2):
    """
    Compute the (batched) Kronecker product of two matrices.

    For matrices A ∈ ℂ^{m×n} and B ∈ ℂ^{p×q}, the Kronecker product is:

        A ⊗ B =
        [ a_{11} B  a_{12} B  ...  a_{1n} B
          a_{21} B  a_{22} B  ...  a_{2n} B
          ...
          a_{m1} B  a_{m2} B  ...  a_{mn} B ]

    resulting in a matrix of shape (m·p, n·q).

    This implementation supports batched inputs, requiring that the batch
    dimensions of mat1 and mat2 match.

    Args:
        mat1 (torch.Tensor): Tensor of shape (..., m, n).
        mat2 (torch.Tensor): Tensor of shape (..., p, q).

    Returns:
        torch.Tensor: Kronecker product with shape (..., m·p, n·q).
    """
    shp1 = mat1.shape
    shp2 = mat2.shape

    # Ensure batch dimensions match
    assert shp1[:-2] == shp2[:-2], f"{shp1[:-2]} != {shp2[:-2]}"

    # Expand A by repeating each entry into p×q blocks
    mat1 = mat1.repeat_interleave(shp2[-2], -2).repeat_interleave(shp2[-1], -1)

    # Tile B across the block structure of A
    mat2 = mat2.repeat(*[1] * (len(shp1) - 2) + list(shp1[-2:]))

    return mat1 * mat2


def eyes_like(matrix):
    """Return identity matrices of the same size of the input matrix."""
    eye = torch.zeros_like(matrix)
    for k in range(matrix.shape[-1]):
        eye[..., k, k] = 1
    return eye


def calc_eig_delta(u):
    """u is the list of eigenvalues"""
    n = u.shape[-1]
    delta = u.view(-1, 1, n).repeat(1, n, 1) - u.view(-1, n, 1).repeat(1, 1, n)
    return delta.view(*u.shape[:-1], n, n)
