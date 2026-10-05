# Copyright (c) 2023 Javad Komijani

"""Computes eigenvalues and eigenvectors of 2x2 normal matrices."""

import torch


# =============================================================================
def eigh2x2(matrix):
    r"""
    Return eigenvalues and eigenvectors of 2x2 hermitian matrices using closed
    form expressions.

    We use following parametrization of hermitian matrices

    .. math::

        H = [[t + z,    x - i y]
             [x + i y,   t - z ]]

    Then the eigenvalues are :math:`(t - r, t + r)` where
    :math:`r = \sqrt{x^2 + y^2 + z^2}` and the matrix of eigenvectors is
    computed by `_stable_eigvecs_2x2`; see that function for the closed
    form and why it is evaluated the way it is.
    """
    a = torch.real(matrix[..., 0, 0])
    b = torch.real(matrix[..., 1, 1])
    x = torch.real(matrix[..., 1, 0])
    y = torch.imag(matrix[..., 1, 0])

    t = (a + b) * (1/2.)
    z = a - t
    r = (z**2 + x**2 + y**2)**0.5

    eigvals = torch.stack([t - r, t + r], dim=-1)
    eigvecs = _stable_eigvecs_2x2(r, z, x, y, matrix.dtype)

    return eigvals, eigvecs


# =============================================================================
def eigsu2x2(matrix):
    r"""
    Return eigenvalues and eigenvectors of 2x2 special unitary matrices using
    closed form expressions.

    We use following parametrization of 2x2 special unitray matrices

    .. math::

        U = [[t + i z,   i x + y]
             [i x - y,   t - i z]]

    where the deteriminant, i.e., :math:`t^2 + x^2 + y^2 + z^2` is unity.
    (Here we do not check whether the determinat is in fact unity.)
    Then the eigenvalues are :math:`(t - i r, t + i r)` where
    :math:`r = \sqrt{x^2 + y^2 + z^2}` and the matrix of eigenvectors is
    computed by `_stable_eigvecs_2x2`, exactly as in `eigh2x2` (only the
    meaning of `t, z, x, y` differs).
    """
    t = torch.real(matrix[..., 0, 0])
    z = torch.imag(matrix[..., 0, 0])
    x = torch.imag(matrix[..., 0, 1])
    y = torch.real(matrix[..., 0, 1])

    r = (z**2 + x**2 + y**2)**0.5

    eigvals = torch.stack([t - r * 1j, t + r * 1j], dim=-1)
    eigvecs = _stable_eigvecs_2x2(r, z, x, y, matrix.dtype)

    return eigvals, eigvecs


# =============================================================================
def eigu2x2(matrix):
    """Compute eigendecomposition of Unitary 2x2 matrices."""
    root_det = torch.det(matrix).unsqueeze(-1)**0.5
    u, v = eigsu2x2(matrix / root_det.unsqueeze(-1))
    return root_det * u, v


# =============================================================================
def _stable_eigvecs_2x2(r, z, x, y, dtype):
    """
    Numerically stable, gradient-safe eigenvector matrix

        Omega = [[z - r, x - iy], [x + iy, r - z]] / c,  c = sqrt(2 r (r - z))

    for `r = sqrt(z^2 + x^2 + y^2)`, shared by `eigh2x2` and `eigsu2x2`
    (only the meaning of `t, z, x, y` differs between them).

    The denominator `c` vanishes as `z -> r`. One can multiply the first
    column by the phase `(x - iy) / sqrt(x^2 + y^2)` and the second
    column by the phase `(x + iy) / sqrt(x^2 + y^2)`, giving the equivalent
    expression

        Omega = [[iy - x, r + z], [r + z, x + iy]] / d,  d = sqrt(2 r (r + z))

    This does not blow up as `z -> r`; instead it blows up as `z -> -r`.
    We use the first expression when `z <= 0` and the second when `z > 0`.
    For the singularity at `r = 0`, where no canonical choice of eigenvectors
    exists; `eye(2)` is returned there by convention.
    """
    eigvecs = torch.empty(*r.shape, 2, 2, dtype=dtype, device=r.device)

    is_pos = z > 0
    for mask, sign in [(~is_pos, -1), (is_pos, +1)]:
        if not torch.any(mask):
            continue
        zm, rm, xm, ym = z[mask], r[mask], x[mask], y[mask]
        denom = (2 * rm * (rm + sign * zm))**0.5
        if sign < 0:
            eigvecs[mask, 0, 0] = (zm - rm).to(dtype) / denom
            eigvecs[mask, 0, 1] = (xm - 1j * ym) / denom
            eigvecs[mask, 1, 0] = (xm + 1j * ym) / denom
            eigvecs[mask, 1, 1] = (rm - zm).to(dtype) / denom
        else:
            eigvecs[mask, 0, 0] = (1j * ym - xm) / denom
            eigvecs[mask, 0, 1] = (zm + rm).to(dtype) / denom
            eigvecs[mask, 1, 0] = (zm + rm).to(dtype) / denom
            eigvecs[mask, 1, 1] = (xm + 1j * ym) / denom

    # The only remaining singularity is r == 0 (x=y=z=0): Return eye(2).
    origin = (r == 0)
    if torch.any(origin):
        eigvecs[origin] = torch.eye(2, dtype=dtype, device=r.device)

    return eigvecs
