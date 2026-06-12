# Copyright (c) 2023-2026 Javad Komijani

"""
A module for computing singular value decomposition of complex square matrices.
"""

# pylint: disable=invalid-name

from dataclasses import dataclass
from typing import Optional
import torch

from .._eig import eigh
from .svd_result import SVDResult

__all__ = ["svd"]


# =============================================================================
def svd(matrix: torch.Tensor, backend: str = "custom") -> SVDResult:
    """
    Compute the singular value decomposition of the complex square matrix.

    Args:
        matrix (torch.Tensor): Input tensor of shape (..., n, n).
        backend (str, optional): Which implementation to use:
            - "custom": Uses a custom SVD implementation (default).
            - "torch": Uses torch.linalg.svd.

    Returns:
        SVDResult: Structured SVD result.
    """
    if backend == "custom":
        return custom_svd(matrix)
    elif backend == "torch":
        U, S, Vh = torch.linalg.svd(matrix)
        return SVDResult(U=U, S=S, Vh=Vh)
    else:
        raise ValueError(f"Unknown backend: {backend}")


# =============================================================================
@dataclass
class DeprecatedSVDResult:
    """
    Container for the outputs of `svd`.

    Attributes:
        U (torch.Tensor): Left singular vectors.
        S (torch.Tensor): Singular values.
        Vh (torch.Tensor): Right singular vectors.

    Optional Attributes:
        rdet_angle (torch.Tensor): The angle of rooted determinant of input.
        sU (torch.Tensor): Scaled U ensuring special unitarity of `sU Vh`.
        sUVh (torch.Tensor): Product of sU and Vh (special-unitary).
        Sigma (torch.Tensor): Hermitian matrix computed as `Vh† diag(S) Vh`.

    Optional fields default to None, allowing partial initialization.
    """
    U: torch.Tensor
    S: torch.Tensor
    Vh: torch.Tensor
    rdet_angle: Optional[torch.Tensor] = None
    sU: Optional[torch.Tensor] = None
    sUVh: Optional[torch.Tensor] = None
    Sigma: Optional[torch.Tensor] = None

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
        """
        Return the shapes of all defined tensors as a dictionary.
        Fields with None are skipped.
        """
        return {
            key: value.shape
            for key, value in self.__dict__.items()
            if value is not None
        }

    @property
    def dtype(self) -> dict:
        """
        Return the dtypes of all defined tensors as a dictionary.
        Fields with None are skipped.
        """
        return {
            key: value.dtype
            for key, value in self.__dict__.items()
            if value is not None
        }


# =============================================================================
def eigh_with_descending_eigvals(matrix: torch.Tensor):
    """
    Compute the eigenvalue decomposition of a Hermitian matrix with eigenvalues
    ordered in descending order.

    Falls back to manual reordering if the backend does not support
    the `descending` argument.
    """
    try:
        w, v = eigh(matrix, descending=True)
    except TypeError:
        w, v = eigh(matrix)
        w = torch.flip(w, dims=(-1,))
        v = torch.flip(v, dims=(-1,))
    return w, v


# =============================================================================
def custom_svd(matrix: torch.Tensor) -> SVDResult:
    """
    Compute the singular value decomposition of the complex square matrix.

    Args:
        matrix (torch.Tensor): Input tensor of shape (..., n, n).

    Returns:
        SVDResult: Structured SVD result.
    """

    # First obtain S^2 and U
    s_sq, u = eigh_with_descending_eigvals(matrix @ matrix.adjoint())

    # V can be obtained by multiplying S^{-1} U^\dagger and matrix
    s = torch.sqrt(s_sq)
    s[s_sq < 0] = 0  # to remove possible roundoff error
    inv_s = 1 / s
    inv_s[s == 0] = 0

    vh = (inv_s.unsqueeze(-1) * u.adjoint()) @ matrix

    # The method fails if S^{-1} diverges, which will be taken care separately.
    # cond = (torch.sum(s == 0, dim=-1) > 0).ravel()  # not precise (roundoff)
    cond = (torch.linalg.matrix_norm(vh) < 0.99 * vh.shape[-1]**0.5).ravel()
    if torch.sum(cond) > 0:
        n = matrix.shape[-1]
        vh.view(-1, n, n)[cond] = slow_svd(matrix.view(-1, n, n)[cond]).Vh

    return SVDResult(U=u, S=s, Vh=vh)


def slow_svd(matrix):
    r"""
    Return singular value decomposition of the input complex, square matrix.

    The singular value decomposition of matrix :math:`M` is

    .. math::

         M = U S V^\dagger

    If :math:`S^{-1}` exists, then :math:`U V^\dagger` is unique, otherwise
    it is not.
    We explain it now. let us introduce unitary matrices :math:`D_u` and
    :math:`D_v` that satisfy :math:`[D_u, S] = [D_v, S] = 0`.
    Then, one can show that

    .. math::

         M = (U D_u) S (V D_v)^\dagger

    is another valid decomposition only if :math:`S D_u D_v^\dagger = S`.
    When :math:`S` is invertible, the condition indicates :math:`D_u = D_v`.
    This then implies that

    .. math::

        U V^\dagger

    is unique. If some elements of :math:`S` are zero, the above constraint
    does not fully relate :math:`D_v` to :math:`D_u` and :math:`U V^\dagger` is
    not unique anymore. We use a particular presciption to handle this
    situation.
    """
    s_sq, u = eigh_with_descending_eigvals(matrix @ matrix.adjoint())
    _, naive_v = eigh_with_descending_eigvals(matrix.adjoint() @ matrix)
    # Note: V = naive_v @ D.adjoint()

    s = torch.sqrt(s_sq)
    s[s_sq < 0] = 0  # to remove possible roundoff error
    inv_s = 1 / s
    inv_s[s == 0] = 0

    # If all singular values are nonzero, the following expression yields `D`
    # such that `vh = D @ naive_v.adjoint()`.
    # Note that D is block diagonal, each block corresponds to a unique
    # singular value and the block is unitary itself.
    # We replace the block correspoding to vanishing sigular values to I; it is
    # numerically more precise to look at the vanishing diagonal terms in
    # s_times_d than s.

    naive_d = inv_s.unsqueeze(-1) * (u.adjoint() @ matrix @ naive_v)

    fixer = torch.zeros_like(s)
    # fixer[ s == 0] = 1  # not precise because of round off errors
    fixer[torch.linalg.vector_norm(naive_d, dim=-1) < 0.01] = 1

    vh = (naive_d + torch.diag_embed(fixer)) @ naive_v.adjoint()

    return SVDResult(U=u, S=s, Vh=vh)


# =============================================================================
def extract_row_unitary(matrix):
    """
    Recover a unitary matrix by orthonormalizing the rows of a row-scaled
    unitary matrix.

    The input is assumed to have the form

        A = S Vh

    where

        S  – diagonal matrix (typically singular values)
        Vh – unitary matrix (V†)

    This function removes the scaling by performing a Gram–Schmidt
    orthonormalization on the rows of `matrix`, returning an estimate of Vh.

    Rows are processed from bottom to top because the singular values are
    assumed to be ordered from smallest to largest. Rows associated with
    larger singular values typically have better numerical precision, so
    they are fixed first and used to orthogonalize the less reliable rows.

    Parameters
    ----------
    matrix : torch.Tensor
        Tensor of shape (..., n, n) whose rows are assumed to be scaled
        orthogonal vectors.

    Returns
    -------
    torch.Tensor
        Tensor of the same shape containing a unitary matrix approximating Vh.
    """
    n = matrix.shape[-1]

    for ind_a in reversed(range(n)):
        a = matrix[..., ind_a, :]

        for ind_b in reversed(range(ind_a + 1, n)):
            b = matrix[..., ind_b, :]

            coef_ab = torch.sum(a.conj() * b, dim=-1, keepdim=True)
            coef_bb = torch.sum(b.conj() * b, dim=-1, keepdim=True)

            a = a - (coef_ab / coef_bb) * b

        norm = torch.linalg.vector_norm(a, dim=-1, keepdim=True)
        matrix[..., ind_a, :] = a / norm

    return matrix


# =============================================================================
# The following ones are used for test and will be removed in future.

def append_suvh(svd_):
    r"""Return a new svd_ object that also includes the produce of U and Vh
    projected to special unitary matrices as

    .. math::

         (U @ V^\dagger) * phase_factor

    where the phase factor is constructed such that the matrix turns to SU(n).
    We call this matrix `sUVh`.
    It also returns determinant of `(U @ V^\dagger)`.
    """
    uvh = svd_.U @ svd_.Vh
    rdet = torch.det(uvh)**(1 / uvh.shape[-1])  # root of determinant
    # We now make determinant of uvh unity:
    uvh = uvh / rdet.reshape(*rdet.shape, 1, 1)
    return DeprecatedSVDResult(
        U=svd_.U, S=svd_.S, Vh=svd_.Vh, rdet_angle=rdet, sUVh=uvh
    )


def append_su(svd_, matrix=None):
    r"""
    Return a new svd_ object, in which U is scaled by a phase, and called sU,
    such that sU @ Vh is special unitary
    """
    det = torch.det(svd_.U @ svd_.Vh if matrix is None else matrix)
    rdet_angle = torch.angle(det) / svd_.U.shape[-1]  # r: rooted
    s_u = svd_.U * torch.exp(-1j * rdet_angle.reshape(*rdet_angle.shape, 1, 1))
    return DeprecatedSVDResult(
        U=svd_.U, S=svd_.S, Vh=svd_.Vh, rdet_angle=rdet_angle, sU=s_u
        )
