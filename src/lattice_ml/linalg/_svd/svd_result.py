# Copyright (c) 2026 Javad Komijani

"""
Singular Value Decomposition utilities and SU(n) projection tools.

This module provides a structured interface around the Singular Value
Decomposition (SVD) of complex matrices, along with utilities for projecting
unitary matrices onto the special unitary group SU(n).

Overview
--------
Given a matrix M ∈ ℂ^{n×n}, its SVD is:

    M = U @ diag(S) @ Vh

where U and Vh are unitary and S contains non-negative singular values.
This decomposition is used to construct:

- The polar (unitary) factor:
      U_polar = U @ Vh

- The special unitary projection:
      U_{SU(n)} = argmax_{X ∈ SU(n)} ReTr (X† M)

  which is obtained by applying a diagonal phase correction matrix D:

      U_{SU(n)} = U @ D @ Vh

  where D is a diagonal unitary matrix chosen such that det(U_{SU(n)}) = 1,
  while minimally adjusting phase.

Features
--------
- Lazy, cached access to commonly used derived quantities:
    * Polar unitary factor (U @ Vh); this one is not cached
    * Special unitary projection (SU(n))
    * Diagonal phase correction matrix
    * Sigma matrix

"""

# pylint: disable=invalid-name, arguments-differ

from dataclasses import dataclass, field
import torch


__all__ = ["SVDResult", "project_to_special_unitary_from_svd"]


# =============================================================================
@dataclass
class SVDResult:
    """
    Structured container for Singular Value Decomposition (SVD).

    A matrix M is decomposed as:

        M = U @ diag(S) @ Vh

    where:
        - U  ∈ ℂ^{n×n} is unitary (left singular vectors)
        - S  ∈ ℝ⁺^{n} is the vector of singular values
        - Vh ∈ ℂ^{n×n} is unitary (the dagger of right signular vectors V)

    This class also provides cached derived quantities, such SU(n) projections.

    Notes
    -----
    - All tensors may be batched (..., n, n) or (..., n)
    - Lazy properties cache expensive matrix constructions after first access
    - Designed for use in differentiable linear algebra pipelines

    Cached Attributes
    ------------------
    _s_unitary : torch.Tensor
        Projection of unitary factor onto SU(n)
    _diagonal_phase: torch.Tensor
        The diagonal phase matrix used in projection onto SU(n). We olny save
        the diagonal terms and we use `D` to denote the full matrix.
    _sigma_matrix : torch.Tensor
        The Σ matrix defined as `Σ = V @ diag(D @ S) @ V†`.

    Attributes
    ----------
    U : torch.Tensor
        Left singular vectors (..., n, n)
    S : torch.Tensor
        Singular values (..., n)
    Vh : torch.Tensor
        Right singular vectors (Hermitian transpose) (..., n, n)

    Example
    -------
    >>> import torch
    >>> M = torch.randn(3, 3, dtype=torch.complex128)
    >>> U, S, Vh = torch.linalg.svd(M)
    >>> svd_result = SVDResult(U=U, S=S, Vh=Vh)
    >>> M_reconstructed = svd_result.reconstruct()
    >>> Q = svd_result.special_unitary_factor
    >>> D = svd_result.diagonal_phase_factor
    """

    U: torch.Tensor
    S: torch.Tensor
    Vh: torch.Tensor

    _s_unitary: torch.Tensor = field(default=None, init=False, repr=False)
    _diagonal_phase: torch.Tensor = field(default=None, init=False, repr=False)
    _sigma_matrix: torch.Tensor = field(default=None, init=False, repr=False)

    def __repr__(self) -> str:
        """
        Return a full string representation of the tensors that are defined.

        Attributes with value None are skipped.
        """
        items = [f"{key}={value}" for key, value in self.__dict__.items()
                 if value is not None]
        return "SVDResult(\n" + ",\n".join(items) + ")"

    @property
    def shape(self) -> dict:
        """Return the shape of tensors as a dictionary (skip if None)."""
        return {
            key: value.shape
            for key, value in self.__dict__.items()
            if value is not None
        }

    @property
    def dtype(self) -> dict:
        """Return the dtypes of tensors as a dictionary (skip if None)."""
        return {
            key: value.dtype
            for key, value in self.__dict__.items()
            if value is not None
        }

    def reconstruct(self) -> torch.Tensor:
        """Reconstruct the original matrix."""
        return self.U @ (self.S[..., None] * self.Vh)

    @property
    def unitary_factor(self):
        """Return the projection onto U(n), i.e., U @ Vh."""
        return self.U @ self.Vh  # no need to be cached

    @property
    def special_unitary_factor(self):
        """Return the projection onto SU(n)."""
        if self._s_unitary is None:
            s_unitary, d_phase = project_to_special_unitary_from_svd(self)
            self._s_unitary = s_unitary
            self._diagonal_phase = d_phase
        return self._s_unitary

    @property
    def diagonal_phase_factor(self):
        """Return the diagonal phase matrix used in projection onto SU(n)."""
        if self._diagonal_phase is None:
            _ = self.special_unitary_factor  # fills _d_diagonal too
        return self._diagonal_phase

    @property
    def sigma_matrix_factor(self):
        """Return the Σ factor: `Σ = V @ diag(D @ S) @ V†`.

        Notes:
            1. Using X to denote the projection onto SU(n), the original matrix
               M can be factorized as `M = X Σ`
            2. Because `Im(D S) = λ I`, where λ is the Lagrange multiplier
               enforcing det = 1 in SU(n) projection.
            3. From (2), we conclude Σ is Hermitian up to an additive imaginary
               component as: `Σ = H + i λ I`, where H is a Hermitian matrix.
        """
        if self._sigma_matrix is None:
            D = self.diagonal_phase_factor
            self._sigma_matrix = (
                self.Vh.adjoint() @ ((D * self.S)[..., None] * self.Vh)
            )
        return self._sigma_matrix

    def clear_cache(self) -> None:
        """Clear cached quantities, but keep core SVD components (U, S, Vh)."""
        self._s_unitary = None
        self._diagonal_phase = None
        self._sigma_matrix = None


# =============================================================================
def project_to_special_unitary_from_svd(svd_result: SVDResult, n_iter=8):
    """
    Project the input onto SU(n) by solving `argmax_{X ∈ SU(n)} Re Tr(X† M)`.

    The solution is obtained by correcting the unitary (polar) factor with an
    optimal diagonal phase matrix D:
        X = U @ D @ Vh

    where:
        D = diag(e^{iθ_1}, ..., e^{iθ_n}) ∈ U(1)^n

    is chosen such that:
        - det(X) = 1 (special unitary constraint)
        - X is optimal for the above variational problem

    The phases θ_j are determined via a scalar Lagrange multiplier λ solving:
        sum_j arcsin(λ / σ_j) = θ

    where σ_j are the singular values and θ = -arg det(U @ Vh).

    The optimal phases satisfy:
        sin(θ_j) = λ / σ_j

    which implies the uniformity condition:
        Im(D @ S) = λ I

    i.e. σ_j sin(θ_j) = λ for all j. This expresses that the determinant
    constraint is enforced uniformly across all singular directions via
    the scalar Lagrange multiplier λ.

    Notes
    -----
    - D lies in the maximal torus U(1)^n ⊂ U(n)
    - This is an exact projection (not a simple determinant normalization)
    - The construction is compatible with automatic differentiation

    Args:
        svd_result (SVDResult): Precomputed SVD container of the input matrix
        n_iter (int): Number of Newton iterations used to solve for λ

    Returns:
        Tuple[torch.Tensor, torch.Tensor]
            - Projected matrix in SU(n), shape (..., n, n)
            - Diagonal elemetns of phase correction D matrix, shape (..., n)
    """
    # Step 1: unitary projection
    U_polar = svd_result.unitary_factor

    if not torch.is_complex(U_polar):
        raise ValueError("Real input detected. Use SO(n) projection instead.")

    # Step 2: compute required phase correction
    det_phase = torch.angle(torch.det(U_polar))

    # Step 3: solve for angles
    angles = compute_optimal_phase_angles(svd_result.S, -det_phase, n_iter)

    # Step 4: construct diagonal correction
    D = torch.exp(1j * angles)

    # Step 5: reconstruct SU(N) matrix
    return svd_result.U @ (D[..., None] * svd_result.Vh), D


def naive_project_to_special_unitary_from_svd(svd_result: SVDResult):
    """
    Fast and approximate projection onto SU(n).

    This method removes only the global phase:
        U → U / (det U)^(1/n)

    It is exact only when singular values are equal.

    Args:
        svd_result (SVDResult): Precomputed SVD container of the input matrix

    Returns:
        torch.Tensor: Naively projected SU(n) matrix.
    """

    U_polar = svd_result.unitary_factor

    n = U_polar.shape[-1]
    phase = torch.angle(torch.det(U_polar)) / n

    phase_factor = torch.exp(-1j * phase)[..., None, None]

    return U_polar * phase_factor


# =============================================================================
def compute_optimal_phase_angles(singular_values, target_phase, n_iter=8):
    """
    Compute optimal phase angles θ_j for SU(n) projection.

    The phases satisfy:
        sum_j θ_j = θ
        sin(θ_j) = λ / σ_j  for j ≠ argmin σ

    For all but the smallest singular value, θ_j is taken from the principal
    branch of arcsin (i.e. θ_j ∈ [-π/2, π/2]). The remaining phase is assigned
    to the smallest singular value, which resolves the branch ambiguity when
    the true phase exceeds this range.

    Args:
        singular_values (torch.Tensor): Singular values σ_j, sorted descending.
        target_phase (torch.Tensor): Required total phase θ.
        n_iter (int): Number of Newton iterations.

    Returns:
        torch.Tensor: Phase angles θ_j (..., n)
    """
    lam = SolveLambdaSUN.apply(singular_values, target_phase, n_iter)

    # principal angles for all but smallest singular value
    angles = torch.asin(lam[..., None] / singular_values[..., :-1])

    # last angle enforces sum constraint
    last_angle = target_phase - angles.sum(dim=-1)

    return torch.cat([angles, last_angle[..., None]], dim=-1)


# =============================================================================
def solve_lambda_sun(singular_values, target_phase, n_iter=8, eps=1e-16):
    # This is the API version that is wrapped with safe AD.
    """
    Solve for the Lagrange multiplier λ enforcing det = 1 in SU(n) projection.

    Solves `sum_j arcsin(λ / σ_j) = θ` using Newton iterations.

    Args:
        singular_values (torch.Tensor): Singular values σ_j, sorted descending.
        target_phase (torch.Tensor): Required total phase θ.
        n_iter (int): Number of Newton iterations.
        eps (float): Numerical stability constant.

    Returns:
        torch.Tensor: Solution λ.
    """
    return SolveLambdaSUN.apply(singular_values, target_phase, n_iter, eps)


def _solve_lambda_sun(singular_values, target_phase, n_iter=8, eps=1e-16):
    # This is the core version that is NOT wrapped with safe AD.
    """
    Solve for the Lagrange multiplier λ enforcing det = 1 in SU(n) projection.

    Solves `sum_j arcsin(λ / σ_j) = θ` using Newton iterations.

    Args:
        singular_values (torch.Tensor): Singular values σ_j, sorted descending.
        target_phase (torch.Tensor): Required total phase θ.
        n_iter (int): Number of Newton iterations.
        eps (float): Numerical stability constant.

    Returns:
        torch.Tensor: Solution λ.
    """
    # ----------------------------------------------------------
    # PARAMETRIZATION (important)
    #
    # We avoid directly constraining λ by introducing:
    #
    #     λ = σ_min * sin(η)
    #
    # This guarantees:
    #     |λ| ≤ σ_min  ⇒  λ / σ_j ∈ [-1, 1]
    #
    # The constraint becomes a scalar equation in η:
    #
    #     f(η) = sum_j arcsin( (σ_min / σ_j) * sin(η) ) - θ = 0
    #
    # Note that η is indeed the phase of the smallest singular value:
    #     f(η) = sum_{j ≠ min} arcsin( (σ_min / σ_j) * sin(η) ) - θ + η = 0
    # ----------------------------------------------------------

    # Numerical safety:
    singular_values = singular_values.clamp_min(eps)

    # The code operates assuming σ_j are sorted from largest to smallest.
    s_min = singular_values[..., -1]

    # Define a convenient ratio r_j = σ_min / σ_j, exclusing the last one
    r = s_min[..., None] / singular_values[..., :-1]

    # Initial guess from linear approximation: η ≈ θ / (σ_min \sum_j 1/σ_j).
    # This guess is exact if all singular values are equal or one vanishes
    eta = target_phase / (r.sum(dim=-1) + 1)

    for _ in range(n_iter):
        # Function value and its derivative
        x = r * torch.sin(eta)[..., None]
        denom = torch.sqrt(torch.clamp(1 - x**2, min=eps**2))
        f = torch.asin(x).sum(dim=-1) - target_phase + eta
        fp = (r / denom).sum(dim=-1) * torch.cos(eta) + 1

        # Newton update
        eta = eta - f / fp

    lam = s_min * torch.sin(eta)

    return lam


class SolveLambdaSUN(torch.autograd.Function):
    """
    Solve for the Lagrange multiplier λ enforcing det = 1 in SU(n) projection.

    Solves `sum_j arcsin(λ / σ_j) = θ` using Newton iterations.

    Args:
        singular_values (torch.Tensor): Singular values σ_j that are postive.
        target_phase (torch.Tensor): Required total phase θ.
        n_iter (int): Number of Newton iterations.
        eps (float): Numerical stability constant.

    Returns:
        torch.Tensor: Solution λ.
    """

    @staticmethod
    def forward(ctx, singular_values, target_phase, n_iter=8, eps=1e-16):
        """Forward pass."""
        lam = _solve_lambda_sun(singular_values, target_phase, n_iter, eps)

        # Save for backward
        ctx.save_for_backward(lam, singular_values)
        ctx.eps = eps

        return lam

    @staticmethod
    def backward(ctx, grad_lam):
        """
        Implicit differentiation:

            f(λ, σ, θ) = 0

        where `f(λ, σ, θ) = sum_j arcsin(λ / σ_j) - θ`.
        Starting from:

            dλ df/dλ + dσ df/dσ + dθ df/dθ = 0

        and using AD we obtain:

            σ̄ = - λ̄ (df/dσ) / (df/dλ)
            θ̄ = - λ̄ (dλ/dθ) / (df/dλ)

        Returns
            σ̄ and θ̄
        """

        lam, singular_values = ctx.saved_tensors
        eps = ctx.eps

        # invert with numerical safety
        x = lam[..., None] / singular_values.clamp_min(eps)
        denom = torch.sqrt(torch.clamp(1 - x**2, min=eps**2))

        # Let us scale the derivatives for clarity, the scaling drops
        lam_df_dlam = (x / denom).sum(dim=-1)  # indeed: λ ∂f/∂λ
        lam_df_dsigma = -x**2 / denom  # indeed: λ ∂f/∂σ_j
        lam_df_dtheta = -lam  # indeed λ ∂f/∂θ

        grad_sigma = - (grad_lam / lam_df_dlam)[..., None] * lam_df_dsigma  # σ̄
        grad_theta = - (grad_lam / lam_df_dlam) * lam_df_dtheta  # θ̄

        return grad_sigma, grad_theta, None, None
