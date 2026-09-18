# Copyright (c) 2026 Javad Komijani

# `repr` is the natural name for the output-representation option, and
# shadowing the builtin in a keyword argument is standard practice.
# pylint: disable=redefined-builtin

r"""Lie-algebra coordinates for SU(N) group elements.

Every SU(N) element can be written as `U = exp(i H)` with `H` traceless
Hermitian, and `H` expanded in a basis of su(N) generators normalized to
`Tr(T_a T_b) = delta_ab`:

.. math::

    U = e^{i H}, \qquad H = \sum_a \theta_a T_a .

`sun_to_algebra` is invertible **by construction**: it takes the eigenangles
from `torch.angle`, which land in `(-pi, pi]`, and shifts them to sum to zero,
which places them in the principal cell (the fundamental alcove). For SU(2)
that is `|theta| <= pi`; for SU(3) it is `w = (max - min) / (2 pi) <= 1`. No
compactification is applied, and none is needed.

`algebra_to_sun` is the exact inverse **only on that cell**. Outside it the
exponential map is many-to-one, so a round trip returns a different
representative of the same group element. If you need coordinates on an
unbounded space (a Gaussian prior, say), compose an explicit squashing map of
your own; deliberately not built in here, so that the plain functions stay a
clean bijection.

Note that `functions/_matrix_func_and_jacobian.py` also exponentiates
Hermitian matrices, but returns the *full* `d^2 x d^2` Jacobian matrix. The
functions here return a per-sample `log|det J|` instead, as normalizing flows
need. The two are not interchangeable.
"""

import torch

from ._ordering import ZeroSumOrder
from ._su3_eigangles_parametrization import su3_sorted_angles_to_rectangle
from .._autograd import eigh, eigu, inverse_eigh, inverse_eign


__all__ = ['sun_to_algebra', 'algebra_to_sun']


ALGEBRA_REPRS = ('matrix', 'coeffs', 'axis_angle')
_SQRT2 = 2 ** 0.5


# =============================================================================
def resolve_repr(repr_):
    """Validate `repr`, mapping None onto the default."""
    if repr_ is None:
        return 'coeffs'
    if repr_ not in ALGEBRA_REPRS:
        raise ValueError(
            f"repr must be one of {ALGEBRA_REPRS}, got {repr_!r}"
        )
    return repr_


# =============================================================================
# su(N) generator bases
# =============================================================================
def su2_coeffs_to_matrix(coeffs):
    """Coefficients (..., 3) -> traceless Hermitian (..., 2, 2)."""
    dtype = torch.promote_types(coeffs.dtype, torch.complex64)
    matrix = torch.zeros(
        (*coeffs.shape[:-1], 2, 2), device=coeffs.device, dtype=dtype
    )
    matrix[..., 0, 0] = coeffs[..., 2]
    matrix[..., 1, 1] = -coeffs[..., 2]
    matrix[..., 1, 0] = coeffs[..., 0] + 1j * coeffs[..., 1]
    matrix[..., 0, 1] = matrix[..., 1, 0].conj()
    return matrix / _SQRT2


def su2_matrix_to_coeffs(matrix):
    """Inverse of `su2_coeffs_to_matrix`."""
    coeffs = torch.zeros(
        (*matrix.shape[:-2], 3), device=matrix.device, dtype=matrix.real.dtype
    )
    coeffs[..., 0] = matrix[..., 1, 0].real
    coeffs[..., 1] = matrix[..., 1, 0].imag
    coeffs[..., 2] = matrix[..., 0, 0].real
    return coeffs * _SQRT2


def su3_coeffs_to_matrix(coeffs):
    """Coefficients (..., 8) -> traceless Hermitian (..., 3, 3)."""
    dtype = torch.promote_types(coeffs.dtype, torch.complex64)
    matrix = torch.zeros(
        (*coeffs.shape[:-1], 3, 3), device=coeffs.device, dtype=dtype
    )
    matrix[..., 0, 0] = coeffs[..., 2] + coeffs[..., 7] / 3**0.5
    matrix[..., 1, 1] = -coeffs[..., 2] + coeffs[..., 7] / 3**0.5
    matrix[..., 2, 2] = coeffs[..., 7] * (-2 / 3**0.5)
    matrix[..., 1, 0] = coeffs[..., 0] + 1j * coeffs[..., 1]
    matrix[..., 2, 0] = coeffs[..., 3] + 1j * coeffs[..., 4]
    matrix[..., 2, 1] = coeffs[..., 5] + 1j * coeffs[..., 6]
    matrix[..., 0, 1] = matrix[..., 1, 0].conj()
    matrix[..., 0, 2] = matrix[..., 2, 0].conj()
    matrix[..., 1, 2] = matrix[..., 2, 1].conj()
    return matrix / _SQRT2


def su3_matrix_to_coeffs(matrix):
    """Inverse of `su3_coeffs_to_matrix`."""
    coeffs = torch.zeros(
        (*matrix.shape[:-2], 8), device=matrix.device, dtype=matrix.real.dtype
    )
    coeffs[..., 2] = (matrix[..., 0, 0].real - matrix[..., 1, 1].real) / 2
    coeffs[..., 7] = matrix[..., 2, 2].real / (-2 / 3**0.5)
    coeffs[..., 0] = matrix[..., 1, 0].real
    coeffs[..., 1] = matrix[..., 1, 0].imag
    coeffs[..., 3] = matrix[..., 2, 0].real
    coeffs[..., 4] = matrix[..., 2, 0].imag
    coeffs[..., 5] = matrix[..., 2, 1].real
    coeffs[..., 6] = matrix[..., 2, 1].imag
    return coeffs * _SQRT2


_COEFFS_TO_MATRIX = {2: su2_coeffs_to_matrix, 3: su3_coeffs_to_matrix}
_MATRIX_TO_COEFFS = {2: su2_matrix_to_coeffs, 3: su3_matrix_to_coeffs}


def _dispatch(table, n):
    try:
        return table[n]
    except KeyError:
        raise ValueError(f"N = {n} is not supported") from None


# =============================================================================
# eigenangles: zero sum, and the direction-independent radius
# =============================================================================
def _zero_sum_eigangs(eigangs):
    """Shift eigenangles to sum to exactly zero, keeping them aligned with
    the eigenvector columns they came from.

    `torch.angle` returns each angle in `(-pi, pi]` independently, so for
    N > 2 the sum can be a non-zero multiple of `2 pi` and the reconstructed
    `H` would not be traceless. For SU(2) the two angles are exact negatives
    and this is already a no-op, but it is applied uniformly.
    """
    order = ZeroSumOrder(eigangs)
    return order.revert(order.sorted_val)


def _modified_theta(eigangs, n):
    r"""The direction-independent rescaling of `theta` used by
    `repr='axis_angle'`.

    A unit-`Tr H^2` generator still has one shape degree of freedom for
    N > 2, so `theta` alone does not bound the alcove. For SU(2) there is no
    such freedom and this is `theta` itself. For SU(3) it is
    `theta cos(phi) = pi w`, with `w` the rectangle coordinate of
    `su3_sorted_angles_to_rectangle`.
    """
    if n == 2:
        return eigangs.norm(dim=-1)
    sorted_angs, _ = torch.sort(eigangs, dim=-1)
    w_r, _ = su3_sorted_angles_to_rectangle(sorted_angs)
    return torch.pi * w_r[..., 0]


def log_conjugacy_vol(eigvals):
    r"""`log prod_{k<l} |lambda_k - lambda_l|^2`, up to an additive constant.

    This is the volume of the conjugacy class, i.e. the Jacobian of
    recomposing a normal matrix from its spectrum and eigenframe.
    """
    log_vol = torch.zeros(eigvals.shape[:-1], dtype=eigvals.real.dtype,
                          device=eigvals.device)
    for k in range(eigvals.shape[-1] - 1):
        diff = eigvals[..., k:k+1] - eigvals[..., k+1:]
        log_vol = log_vol + 2 * torch.sum(torch.log(torch.abs(diff)), dim=-1)
    return log_vol


# =============================================================================
def sun_to_algebra(matrix, repr=None, return_logj=False):
    r"""Map an SU(N) element to its Lie-algebra content.

    `U -> H` with `U = exp(iH)`, `H` traceless Hermitian, returned in one of
    the representations below. The eigenangles are placed in the principal
    cell, so this is a bijection onto that cell (see the module docstring).

    Parameters
    ----------
    matrix : tensor
        SU(N) matrices of shape `(..., N, N)`; N in {2, 3}.

    repr : {'matrix', 'coeffs', 'axis_angle'}, optional
        'matrix'      the Hermitian generator `H`, shape `(..., N, N)`.
        'coeffs'      the flat coefficient vector `theta_a`, shape
                      `(..., N^2 - 1)`. The default.
        'axis_angle'  `(theta, t, modified_theta)` with
                      `theta = |coeffs|`, `t = coeffs / theta` a unit vector,
                      and `modified_theta` a direction-independent radius
                      (see `_modified_theta`).

    return_logj : bool, optional
        Also return `log|det d(repr)/d(su(N))|`, of shape `matrix.shape[:-2]`,
        up to an additive constant. SIGN: this is the FORWARD map;
        `algebra_to_sun` returns its negative, so the two cancel. Not summed
        over field axes -- that is the caller's job.
    """
    repr_ = resolve_repr(repr)
    n = matrix.shape[-1]
    to_coeffs = _dispatch(_MATRIX_TO_COEFFS, n)

    eigvals, eigvecs = eigu(matrix)
    eigangs = _zero_sum_eigangs(torch.angle(eigvals))
    herm = inverse_eigh(eigangs, eigvecs)

    if repr_ == 'matrix':
        out = herm
    else:
        coeffs = to_coeffs(herm)
        if repr_ == 'coeffs':
            out = coeffs
        else:
            theta = coeffs.norm(dim=-1)
            out = (theta, coeffs / theta.unsqueeze(-1),
                   _modified_theta(eigangs, n))

    if not return_logj:
        return out

    # eigu contributes -log(conjugacy volume) of U's spectrum; rebuilding the
    # Hermitian generator contributes +log(conjugacy volume) of the angles.
    logj = log_conjugacy_vol(eigangs) - log_conjugacy_vol(eigvals)
    if repr_ == 'axis_angle':
        # polar coordinates on R^dim: coeffs -> (theta, t)
        logj = logj - (coeffs.shape[-1] - 1) * torch.log(theta)
    return out, logj


# =============================================================================
def algebra_to_sun(x, repr=None, return_logj=False):
    """Inverse of `sun_to_algebra`; `repr` must match the value used there.

    Exact only when the algebra element lies in the principal cell, which
    `sun_to_algebra`'s output always does. Outside it, `exp` is many-to-one
    and the round trip returns a different representative of the same group
    element.
    """
    repr_ = resolve_repr(repr)

    if repr_ == 'axis_angle':
        theta, t, _ = x
        coeffs = t * theta.unsqueeze(-1)
        herm = _coeffs_to_matrix(coeffs)
    elif repr_ == 'coeffs':
        coeffs = x
        herm = _coeffs_to_matrix(coeffs)
    else:
        herm = x

    eigangs, eigvecs = eigh(herm)
    matrix = inverse_eign(torch.exp(1j * eigangs), eigvecs)

    if not return_logj:
        return matrix

    logj = log_conjugacy_vol(torch.exp(1j * eigangs)) \
        - log_conjugacy_vol(eigangs)
    if repr_ == 'axis_angle':
        logj = logj + (coeffs.shape[-1] - 1) * torch.log(theta)
    return matrix, logj


def _n_from_dim(dim):
    """N from `dim(su(N)) = N^2 - 1`."""
    n = round((dim + 1) ** 0.5)
    if n * n - 1 != dim:
        raise ValueError(f"cannot infer N from {dim} coefficients")
    return n


def _coeffs_to_matrix(coeffs):
    """Coefficients -> traceless Hermitian, with N inferred from the shape."""
    return _dispatch(_COEFFS_TO_MATRIX, _n_from_dim(coeffs.shape[-1]))(coeffs)
