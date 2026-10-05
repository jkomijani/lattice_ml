# Copyright (c) 2026 Claude, under supervision of Javad Komijani

"""SU(3) Euler-angle decomposition,

    U = D1(alpha1, alpha2)
        @ R23(theta23)
        @ U13(theta13, delta)
        @ R12(theta12)
        @ D2(beta1, beta2)

with `D1 = diag(e^{i a1}, e^{i a2}, e^{-i(a1 + a2)})` and likewise `D2`;
`R12`, `R23` real rotations in the (1,2) and (2,3) planes; `U13` a rotation in
the (1,3) plane carrying the only complex phase. The middle three factors are
the PDG standard parametrization of the CKM matrix (Review 12, Eq. 12.3); the
diagonal factors restore the four phases CKM discards by rephasing, giving
`4 + 4 = 8 = dim(SU(3))`. One dimension down this is `su2_to_euler_angles`'s
`D(phi) R(theta) D(psi)`.

The parametrization is naively 3-to-1: shifting `(alpha1, alpha2)` by `-gamma`
and `(beta1, beta2)` by `+gamma`, `gamma = 2 pi k / 3`, multiplies `D1` and
`D2` by the central elements `omega^-k` and `omega^k` and leaves the product
unchanged. `alpha1` is therefore canonicalized into that Z_3 action's
fundamental domain `(-pi/3, pi/3]` -- but only *after* a branch search that
reconstructs all three candidates and keeps the one matching the input matrix.
Canonicalizing first breaks even `matrix -> angles -> matrix`. The same
structural redundancy is documented for SU(N) in Tilma & Sudarshan,
arXiv:math-ph/0205016.

Both directions are then exact, `matrix -> angles -> matrix` and
`angles -> matrix -> angles` on canonical input. The branch selection is the
one non-differentiable step: gradients through a chosen branch are fine, the
selection itself is not smooth at branch boundaries.
"""

# pylint: disable=invalid-name, too-many-locals
# pylint: disable=too-many-arguments, too-many-positional-arguments

import torch
import numpy as np

from ._euler_angles_su2 import resolve_coords, pack, unpack

TWO_PI = 2 * np.pi
TWO_PI_OVER_3 = TWO_PI / 3


__all__ = ["su3_to_euler_angles", "euler_angles_to_su3"]


# =============================================================================
def su3_to_euler_angles(
    matrix,
    coords=None,
    channel_axis=None,
    return_logj=False
):
    """Euler decomposition of SU(3) matrices.

    Returns `(alpha1, alpha2, theta23, theta13, delta, theta12, beta1, beta2)`
    for `coords='angles'`; for `coords='uniform'`, the same eight in the same
    order, each mapped onto [0, 1] such that all are exactly uniform under Haar
    (`sin^2` for `theta12`/`theta23`, `1 - cos^4` for `theta13`, an affine
    rescaling for the five phases), which makes `su3_log_jacobian` vanish.

    The order follows the factorization left to right, so the diagonal
    factors' phases sit on the outside and the four parameters of the CKM
    core -- `theta23, theta13, delta, theta12`, channels 2..5 -- sit
    contiguously in the middle, mirroring SU(2)'s modulus-in-the-middle.

    Parameters
    ----------
    matrix : tensor
        the SU(3) matrix (or batch of matrices) to decompose.

    coords : {'angles', 'uniform'} or None (optional)
        Which coordinates to return; None means 'angles'. See the
        `_euler_angles_su2` module docstring for the two conventions.

    channel_axis : int or None (optional)
        If integer, the 8 coordinates are stacked along this axis; if None
        (default), they are returned as a tuple instead.

    return_logj : bool (optional)
        Also return `su3_log_jacobian(...)`, of shape `matrix.shape[:-2]`
        (default False). It is NOT summed over any field axes.
    """
    coords = resolve_coords(coords)
    angles = _su3_to_raw_angles(matrix, channel_axis=None)
    out = pack(_to_coords(angles, coords), channel_axis)

    if not return_logj:
        return out
    return out, su3_log_jacobian(*_mixing_angles(angles), coords)


# =============================================================================
def euler_angles_to_su3(
    param,
    coords=None,
    channel_axis=None,
    return_logj=False
):
    """Inverse of `su3_to_euler_angles`; `coords` must match the value used
    there. With `return_logj=True` the log-Jacobian is the negative of the
    forward one, so the two cancel.
    """
    coords = resolve_coords(coords)
    angles = _from_coords(unpack(param, channel_axis, 8), coords)
    matrix = _euler_angles_to_su3_matrix(*angles)

    if not return_logj:
        return matrix
    return matrix, -su3_log_jacobian(*_mixing_angles(angles), coords)


# =============================================================================
def su3_log_jacobian(theta12, theta13, theta23, coords):
    r"""Return `log |det d(coords)/d(su(3))|`, up to an ADDITIVE CONSTANT.

    Same convention as `su2_log_jacobian`: this is the FORWARD map
    `matrix -> coords`, and the constant is meaningful only in differences.

    For `coords='angles'` the Haar measure factorizes into the kernel
    `sin(t12) cos(t12) * sin(t13) cos(t13)^3 * sin(t23) cos(t23)` times a flat
    measure in the five phases -- the extra power of cosine being the same
    (1,3)-plane asymmetry that makes `U13` carry `delta`. Verified against
    finite differences in left-invariant coordinates.
    """
    if coords == 'uniform':
        return torch.zeros_like(theta12)
    return -(
        torch.log(torch.sin(theta12)) + torch.log(torch.cos(theta12))
        + torch.log(torch.sin(theta13)) + 3 * torch.log(torch.cos(theta13))
        + torch.log(torch.sin(theta23)) + torch.log(torch.cos(theta23))
    )


# =============================================================================
# Coordinate conventions
# =============================================================================
def _mixing_angles(angles):
    """`(theta12, theta13, theta23)` -- all `su3_log_jacobian` needs.

    They are NOT the first three coordinates; pulling them out by name here
    keeps the call sites from silently picking up phases if the order moves.
    """
    return angles[5], angles[3], angles[2]


def _to_coords(angles, coords):
    """Raw Euler angles -> the requested coordinates."""
    if coords == 'angles':
        return angles

    (alpha1, alpha2, theta23, theta13, delta, theta12, beta1, beta2) = angles

    return (
        alpha1 * 3 / TWO_PI + 0.5,      # alpha1 lives in (-pi/3, pi/3]
        alpha2 / TWO_PI + 0.5,
        torch.sin(theta23)**2,          # kernel sin(t) cos(t)
        1 - torch.cos(theta13)**4,      # kernel sin(t) cos(t)^3
        delta / TWO_PI + 0.5,
        torch.sin(theta12)**2,          # kernel sin(t) cos(t)
        beta1 / TWO_PI + 0.5,
        beta2 / TWO_PI + 0.5,
    )


def _from_coords(param, coords):
    """Inverse of `_to_coords`."""
    if coords == 'angles':
        return param
    return (
        (param[0] - 0.5) * TWO_PI / 3,                    # alpha1
        (param[1] - 0.5) * TWO_PI,                        # alpha2
        torch.asin(param[2].clamp(0, 1).sqrt()),          # theta23
        torch.acos(((1 - param[3]).clamp(0, 1))**0.25),   # theta13
        (param[4] - 0.5) * TWO_PI,                        # delta
        torch.asin(param[5].clamp(0, 1).sqrt()),          # theta12
        (param[6] - 0.5) * TWO_PI,                        # beta1
        (param[7] - 0.5) * TWO_PI,                        # beta2
    )


# =============================================================================
# The decomposition itself
# =============================================================================
def _wrap_pi(x):
    """Wrap an angle to (-pi, pi]."""
    return torch.atan2(torch.sin(x), torch.cos(x))


def _su3_to_raw_angles(matrix, channel_axis=-1):
    """SU(3) matrix -> the 8 raw Euler angles, in factorization order.

    `theta12, theta13, theta23` are read off matrix-entry magnitudes, so they
    are unambiguous and lie in `[0, pi/2]`; `delta, alpha2, beta1, beta2` lie
    in `(-pi, pi]`; `alpha1` lies in `(-pi/3, pi/3]` after the Z_3 branch
    search and canonicalization the module docstring describes.

    Returns `(alpha1, alpha2, theta23, theta13, delta, theta12, beta1, beta2)`,
    stacked along `channel_axis` if that is an int, else as a tuple.
    """
    U = matrix
    abs11 = U[..., 0, 0].abs()
    abs12, abs13 = U[..., 0, 1].abs(), U[..., 0, 2].abs()
    abs23, abs33 = U[..., 1, 2].abs(), U[..., 2, 2].abs()

    theta13 = torch.asin(abs13.clamp(-1, 1))
    theta12 = torch.atan2(abs12, abs11)
    theta23 = torch.atan2(abs23, abs33)

    A = torch.angle(U[..., 0, 0])
    B = torch.angle(U[..., 0, 1])
    C = torch.angle(U[..., 1, 2])
    D = torch.angle(U[..., 2, 2])
    E = torch.angle(U[..., 0, 2])

    base = (2 * A + 2 * B + C + D) / 3  # true alpha1, up to a 2*pi/3 branch

    best_err, best = None, None
    for k in (-1, 0, 1):
        alpha1_k = _wrap_pi(base + k * TWO_PI / 3)
        alpha2_k = _wrap_pi((-alpha1_k + C - D) / 2)
        beta1_k = _wrap_pi(A - alpha1_k)
        beta2_k = _wrap_pi(B - alpha1_k)
        delta_k = _wrap_pi(A + B + C + D - E)

        U_k = _euler_angles_to_su3_matrix(
            alpha1_k, alpha2_k, theta23, theta13,
            delta_k, theta12, beta1_k, beta2_k,
        )
        err_k = (U_k - U).abs().amax(dim=(-2, -1))

        if best_err is None:
            best_err = err_k
            best = (alpha1_k, alpha2_k, beta1_k, beta2_k, delta_k)
        else:
            better = err_k < best_err
            best_err = torch.where(better, err_k, best_err)
            best = tuple(
                torch.where(better, cand_k, cand_best)
                for cand_k, cand_best in zip(
                    (alpha1_k, alpha2_k, beta1_k, beta2_k, delta_k), best
                )
            )

    alpha1, alpha2, beta1, beta2, delta = best

    # Canonicalize away the residual Z_3 redundancy: shift (alpha1, alpha2)
    # by -gamma and (beta1, beta2) by +gamma (a shift that provably leaves
    # the matrix unchanged -- see module docstring) so that alpha1 lands in
    # its fundamental domain (-pi/3, pi/3].
    gamma = torch.round(alpha1 / TWO_PI_OVER_3) * TWO_PI_OVER_3
    alpha1 = _wrap_pi(alpha1 - gamma)
    alpha2 = _wrap_pi(alpha2 - gamma)
    beta1 = _wrap_pi(beta1 + gamma)
    beta2 = _wrap_pi(beta2 + gamma)

    # Ordered along the factorization itself, D1 R23 U13 R12 D2, matching
    # `su2_to_euler_angles`'s `(phi, theta, psi)` for `D(phi) R(theta) D(psi)`.
    out = (alpha1, alpha2, theta23, theta13, delta, theta12, beta1, beta2)
    if isinstance(channel_axis, int):
        return torch.stack(out, channel_axis)
    return out


def _euler_angles_to_su3_matrix(
    alpha1, alpha2, theta23, theta13, delta, theta12, beta1, beta2
):
    """The 8 angles -> the matrix. Argument order is the canonical channel
    order, so callers holding the coordinates can splat them in directly."""
    # Argument order is the canonical channel order, so that callers holding
    # the 8 coordinates can splat them in directly.
    c12, s12 = torch.cos(theta12), torch.sin(theta12)
    c13, s13 = torch.cos(theta13), torch.sin(theta13)
    c23, s23 = torch.cos(theta23), torch.sin(theta23)

    is_double = theta12.dtype == torch.float64
    cdtype = torch.complex128 if is_double else torch.complex64
    c12, s12, c13, s13, c23, s23 = (
        t.to(cdtype) for t in (c12, s12, c13, s13, c23, s23)
    )
    ephase = torch.exp(1j * delta.to(cdtype))
    emphase = torch.exp(-1j * delta.to(cdtype))

    V = torch.stack([
        torch.stack([c12 * c13, s12 * c13, s13 * emphase], dim=-1),
        torch.stack([
            -s12 * c23 - c12 * s23 * s13 * ephase,
            c12 * c23 - s12 * s23 * s13 * ephase,
            s23 * c13,
        ], dim=-1),
        torch.stack([
            s12 * s23 - c12 * c23 * s13 * ephase,
            -c12 * s23 - s12 * c23 * s13 * ephase,
            c23 * c13,
        ], dim=-1),
    ], dim=-2)

    alpha3, beta3 = -alpha1 - alpha2, -beta1 - beta2
    alphas = torch.stack([alpha1, alpha2, alpha3], dim=-1).to(cdtype)
    betas = torch.stack([beta1, beta2, beta3], dim=-1).to(cdtype)
    d1, d2 = torch.exp(1j * alphas), torch.exp(1j * betas)
    return d1.unsqueeze(-1) * V * d2.unsqueeze(-2)
