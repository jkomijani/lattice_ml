# Copyright (c) 2024-2025 Javad Komijani

"""
This module provides various functions, especially for mapping between unitary
matrices and their Lie algebra elements.
"""

# pylint: disable=invalid-name  # for matrices like U & A

import itertools
import math
import torch

from lattice_ml.linalg import eigh, eigu, inverse_eign


__all__ = [
    "exp_unitary_algebra",
    "log_unitary_group",
    "log_special_unitary_group",
    "enumerate_sun_preimages",
    "pow_special_unitary_group",
    "pow_special_unitary_group_",
    "matrix_exp1jh",
    "matrix_angleu",
    "matrix_cumprod",
    "matrix_prod",
    "enforce_zero_sum",
    "kronecker_product",
    "eye_like"
]


# =============================================================================
def exp_unitary_algebra(A: torch.Tensor) -> torch.Tensor:
    r"""
    Compute the exponential map from the Lie algebra `u(n)` to its lie group.

    Given an anti-Hermitian matrix `A`, this function returns the unitary
    matrix `exp(A)`.  If `A` is also traceless, `U` will be in SU(n).

    Parameters
    ----------
    A : torch.Tensor
        Anti-Hermitian input matrix of shape `(..., n, n)`.

    Returns
    -------
    U : torch.Tensor
        The corresponding unitary matrix (same shape as `A`).

    Notes
    -----
    This function uses the spectral decomposition of `A`:

    .. math::

        A = \Omega \Lambda \Omega^\dagger, \quad
        U = \Omega \exp(\Lambda) \Omega^\dagger.
    """
    vals, vecs = eigh(-1j * A)
    f_vals = torch.exp(1j * vals)
    return inverse_eign(f_vals, vecs)


def log_unitary_group(U: torch.Tensor) -> torch.Tensor:
    r"""
    Compute the logarithm map from the lie group `U(n)` to its Lie algebra.

    Given a unitary matrix `U`, this function returns the anti-Hermitian
    matrix `log(U)`.

    Parameters
    ----------
    U : torch.Tensor
        Unitary input matrix of shape `(..., n, n)`.

    Returns
    -------
    A : torch.Tensor
        The corresponding anti-Hermitian matrix.

    Notes
    -----
    This function uses the spectral decomposition of `U`:

    .. math::

        U = \Omega \Lambda \Omega^\dagger, \quad
        A = \Omega \log(\Lambda) \Omega^\dagger.
    """
    vals, vecs = eigu(U)
    f_vals = 1j * torch.angle(vals)
    return inverse_eign(f_vals, vecs)


def log_special_unitary_group(U: torch.Tensor) -> torch.Tensor:
    r"""
    Compute the logarithm map from the lie group `SU(n)` to its Lie algebra.

    Given a special unitary matrix `U`, this function returns the traceless,
    anti-Hermitian matrix `log(U)`.

    Parameters
    ----------
    U : torch.Tensor
        Speical Unitary input matrix of shape `(..., n, n)`.

    Returns
    -------
    A : torch.Tensor
        The corresponding traceless anti-Hermitian matrix.

    Notes
    -----
    This function uses the spectral decomposition of `U`:

    .. math::

        U = \Omega \Lambda \Omega^\dagger, \quad
        A = \Omega \log(\Lambda) \Omega^\dagger,

    and enssures that the matrix `A` is traceless.
    """
    vals, vecs = eigu(U)
    f_vals = 1j * enforce_zero_sum(torch.angle(vals))
    return inverse_eign(f_vals, vecs)


def enumerate_sun_preimages(logU: torch.Tensor, max_branch_shift: int = 1):
    """Enumerate preimages of the exponential map `exp(logU)` given `logU`.

    Each preimage is obtained by adding 2π times integers to the eigenvalues
    of `logU`, generating different branches of the logarithm.

    Args:
        logU (torch.Tensor): AntiHermitian su(n) matrix of shape `(..., n, n)`.
        max_branch_shift (int): Maximum integer shift for eigenvalues.

    Returns:
        List[torch.Tensor]: Matrices corresponding to different 2π branch
        shifts of the input logU eigenvalues.
    """

    vals, vecs = eigh(1j * logU)
    vals = enforce_zero_sum(vals)  # make logU traceless
    # Note: a traceless logU is not necessarily in the principal branch.

    all_vals = [vals]

    n_c = vals.shape[-1]  # number of colors
    rng = range(-max_branch_shift, max_branch_shift + 1)

    for shift_tuple in itertools.product(rng, repeat=n_c):
        if all(s == 0 for s in shift_tuple):
            continue
        if sum(s**2 for s in shift_tuple) > max_branch_shift**2:
            continue
        shift = torch.tensor(shift_tuple, dtype=vals.dtype, device=vals.device)
        shift = shift * (2.0 * math.pi)
        all_vals.append(vals + shift)

    # Convert eigenvalues into matrices in su(N)
    preimgs = [inverse_eign(-1j * v, vecs) for v in all_vals]
    return preimgs


def pow_special_unitary_group(
    U: torch.Tensor,
    t: float | torch.Tensor
) -> torch.Tensor:
    r"""
    Computes `U^t` for a special unitary matrix using its eigen-decomposition.

    Parameters
    ----------
    U : torch.Tensor
        Special unitary input matrix of shape (..., n, n).
    t : float or torch.Tensor
        Exponent. Can be a float or a tensor broadcastable to the shape
        of the eigenvalues.

    Returns
    -------
    torch.Tensor
        Matrix raised to the power t (same shape as U).

    Notes
    -----
    This function uses the spectral decomposition of `U`:

    .. math::

        U = \Omega \Lambda \Omega^\dagger, \quad
        U^t = \Omega \Lambda^t \Omega^\dagger,

    and enssures that `U^t` remains special unitary.
    """
    vals, vecs = eigu(U)
    log_vals = 1j * enforce_zero_sum(torch.angle(vals))
    f_vals = torch.exp(log_vals * t)
    return inverse_eign(f_vals, vecs)


def pow_special_unitary_group_(
    U: torch.Tensor,
    t: float | torch.Tensor
) -> torch.Tensor:
    r"""
    Computes `U^t` for a special unitary matrix using its eigen-decomposition.
    In addition to `U^t`, this function also returns `log(U)` that is needed
    for taking derivatices with respect to the exponent.

    Parameters
    ----------
    U : torch.Tensor
        Special unitary input matrix of shape (..., n, n).
    t : float or torch.Tensor
        Exponent. Can be a float or a tensor broadcastable to the shape
        of the eigenvalues.

    Returns
    -------
    torch.Tensor
        Matrix raised to the power t (same shape as U) and logarithm of U.

    Notes
    -----
    This function uses the spectral decomposition of `U`:

    .. math::

        U = \Omega \Lambda \Omega^\dagger, \quad
        U^t = \Omega \Lambda^t \Omega^\dagger,

    and enssures that `U^t` remains special unitary.
    """
    vals, vecs = eigu(U)
    log_vals = 1j * enforce_zero_sum(torch.angle(vals))
    f_vals = torch.exp(log_vals * t)
    return inverse_eign(f_vals, vecs), inverse_eign(log_vals, vecs)


def matrix_exp1jh(H: torch.Tensor) -> torch.Tensor:
    r"""
    Return :math:`U = \exp(i H)` with :math:`H` being a Hermitian matrix.

    This is a variant of `exp_unitary_algebra` with Hermitian input matrix.
    """
    vals, vecs = eigh(H)
    f_vals = torch.exp(1j * vals)
    return inverse_eign(f_vals, vecs)


def matrix_angleu(U: torch.Tensor) -> torch.Tensor:
    r"""
    Return :math:`H = -i \log(U)` with :math:`U` being a unitary matrix.

    This is a variant of `log_unitary_group` with Hermitian output matrix.
    """
    vals, vecs = eigu(U)
    f_vals = torch.angle(vals)
    return inverse_eign(f_vals, vecs)


# =============================================================================
def matrix_cumprod(
    W: torch.Tensor,
    dim: int,
    prepend_identity: bool = False,
    use_scan: bool = False,
) -> torch.Tensor:
    """
    Cumulative ordered matrix product along a given axis.

    Computes, for each index k along `dim`:

        Z_0 = W_0                     (or Z_0 = I if prepend_identity=True)
        Z_k = Z_{k-1} @ W_k

    Parameters
    ----------
    W : torch.Tensor
        Tensor of square matrices with shape [..., L, ..., N, N].

    dim : int
        Axis along which to accumulate the product (any axis except the
        last two, which are reserved for the matrix indices).

    prepend_identity : bool, default=False
        If False, output has L entries along `dim`, with Z_0 = W_0.
        If True, prepends Z_0 = I instead, giving L + 1 entries.

    use_scan : bool, default=False
        If False (default), accumulates with a plain sequential loop: O(L)
        sequential matmuls, O(L) total work.  If True, uses a Hillis-Steele
        parallel scan: O(log L) sequential batched-matmul steps instead of
        O(L), at the cost of O(L log L) total work instead of O(L).

    Returns
    -------
    torch.Tensor
        Cumulative products, same shape as `W` except along `dim`, which has
        size L (prepend_identity=False) or L + 1 (prepend_identity=True).
    """
    L = W.shape[dim]

    if use_scan:
        Z = W
        offset = 1
        while offset < L:
            n = L - offset
            prefix = Z.narrow(dim, 0, n)
            suffix = Z.narrow(dim, offset, n)
            updated = prefix @ suffix
            unchanged = Z.narrow(dim, 0, offset)
            Z = torch.cat([unchanged, updated], dim=dim)
            offset *= 2
    else:
        outs = [W.select(dim, 0)]
        for k in range(1, L):
            outs.append(outs[-1] @ W.select(dim, k))
        Z = torch.stack(outs, dim=dim)

    if prepend_identity:
        eye = eye_like(Z.select(dim, 0)).unsqueeze(dim)
        Z = torch.cat([eye, Z], dim=dim)

    return Z


# =============================================================================
def matrix_prod(W: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Ordered product of matrices along a given axis: W_0 @ W_1 @ ... @ W_{L-1}.

    Parameters
    ----------
    W : torch.Tensor
        Tensor of square matrices with shape [..., L, ..., N, N].

    dim : int
        Axis along which to compute the product (any axis except the last two).

    Returns
    -------
    torch.Tensor
        The ordered product; same shape as `W` with `dim` removed.
    """
    result = W.select(dim, 0)
    for i in range(1, W.shape[dim]):
        result = result @ W.select(dim, i)
    return result


# =============================================================================
def enforce_zero_sum(
    x: torch.Tensor,
    dim: int = -1,
    period: float = 2 * math.pi
):
    """
    Adjust a single entry along `dim` so the total sum becomes exactly zero.

    Assumes that the sum along `dim` is already an integer multiple of the
    period (e.g., 2π). The function removes the winding number by modifying
    one entry.

    SU(3) eigen-angle example:
    - If sum = -2π, the smallest angle is increased by 2π.
    - If sum = 2π, the largest angle is decreased by 2π.
    - If sum = 0, no change is needed.

    Algorithm
    ---------
    1. Compute the winding number as the rounded sum divided by period.
    2. Sort entries along `dim` and pick the entry at (winding + n//2) % n.
    3. Map that index back to the original tensor and subtract period*winding.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor (e.g., eigen-angles), can have batch dimensions.
    dim : int, optional
        Dimension along which to enforce zero-sum (default: -1).
    period : float, optional
        Period of the values (default: 2π).

    Returns
    -------
    torch.Tensor
        Copy of `x` with one entry modified so the sum along `dim` equals 0.
    """
    # Compute integer winding number along dim
    n_w = roundint(torch.sum(x, dim=dim) / period).unsqueeze(dim)

    n_d = x.size(dim)

    # Sort values and get original indices
    indices = torch.argsort(x, dim=dim)

    # Comment: For SU(3) eigen-angles, sorting is unnecessary since x is sorted
    # when n_w ≠ 0. We sort anyway for consistency and generality.

    # Determine which entry to adjust in sorted order
    designated_idx = (n_w + n_d // 2) % n_d

    # Map back to original tensor
    designated_orig_idx = torch.gather(indices, dim=dim, index=designated_idx)

    # Subtract winding number * period from the chosen entry
    correction = -n_w.to(x.dtype) * period
    y = x.clone()
    y.scatter_add_(dim, designated_orig_idx, correction)
    return y


def roundint(x, dtype=torch.int64):
    """Return the closest integer to `x`."""
    return torch.round(x).to(dtype)


def kronecker_product(mat1, mat2):
    """Return the Kronecker product of two input matrices."""
    shp1 = mat1.shape
    shp2 = mat2.shape
    assert shp1[:-2] == shp2[:-2], f"{shp1[:-2]} != {shp2[:-2]}"
    mat1 = mat1.repeat_interleave(shp1[-2], -2).repeat_interleave(shp1[-1], -1)
    mat2 = mat2.repeat(*[1]*(len(shp1) - 2) + list(shp1[-2:]))
    return mat1 * mat2


def eye_like(x: torch.Tensor) -> torch.Tensor:
    """
    Return identity matrices matching x's shape, dtype, and device.
    The last two dimensions of x must be square.
    """
    eye = torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
    return eye.repeat(*x.shape[:-2], 1, 1)


# =============================================================================
def _test_enforce_zero_sum(n_samples):
    saved_dtype = torch.get_default_dtype()
    # pylint: disable=import-outside-toplevel
    from normflow.prior import UniformSUnPrior
    torch.set_default_dtype(saved_dtype)  # importing normflow may change dtype
    samples = UniformSUnPrior(n=3, shape=(1,)).sample(n_samples)
    vals, _ = eigu(samples)
    angs = torch.angle(vals)
    angs_p = enforce_zero_sum(angs)
    for x, y in zip(angs, angs_p):
        print(x[0], f"{x.sum():.4f}", y[0])
