# Copyright (c) 2026 Claude, under supervision of Javad Komijani

"""SU(3) Euler-angle decomposition -- resolved, kept in its own file
(separate from the SU(2) decomposition in `_euler_angles.py`) rather than
merged into it.

Status: `su3_to_euler_angles` / `euler_angles_to_su3` below round-trip
*both* directions to machine precision: `matrix -> angles -> matrix`, and
`angles -> matrix -> angles` provided the input angles are already
canonical (see below) -- any input angles are first canonicalized, and it
is the canonicalized form that round-trips exactly.

Background: the parametrization `(theta12, theta13, theta23, delta,
alpha1, alpha2, beta1, beta2)` is naively a 3-to-1 map onto SU(3), tied to
SU(3)'s Z_3 center -- `(alpha1, alpha2, beta1, beta2)` can be shifted by
`(-gamma, -gamma, +gamma, +gamma)` for `gamma = 2*pi*k/3`, `k = 0, 1, 2`,
without changing the reconstructed matrix at all (this is exactly
left-multiplying `D1` and right-multiplying `D2` by the central elements
`omega^{-k}` and `omega^{k}`, `omega = e^{2 pi i / 3}`, which cancel in the
product). This is the same flavor of ambiguity documented for the
"generalized Euler angle" parametrization of SU(N) in Tilma & Sudarshan,
*Generalized Euler Angle Parametrization for SU(N)*, arXiv:math-ph/0205016
-- there, one specific Cartan angle's range must be exactly N times the
"naive" range to make the map a genuine covering of SU(N) rather than
SU(N)/Z_N; the redundancy is structural, not a bug in either derivation.

Fix: restrict `alpha1` to its own fundamental domain of that Z_3 action,
`(-pi/3, pi/3]`, by finding the unique `k` that brings it there and
applying the (algebraically exact, matrix-preserving) shift above to
`alpha2, beta1, beta2` as well -- *not* by directly wrapping a freshly
computed `alpha1` into `(-pi/3, pi/3]` before verifying it against the
target matrix, which was tried first and made things worse (broke even
`matrix -> angles -> matrix`, for reasons not fully tracked down -- likely
that the raw pre-branch-selection formula for `alpha1` isn't otherwise
guaranteed consistent with `alpha2, beta1, beta2`'s formulas without the
reconstruction-verified branch search happening first). Verified (see
`dev_only/dev_linalg/euler_anlges/su3_euler_angles.ipynb`): two different
input angle-tuples that reconstruct the same matrix canonicalize to the
identical representative (to ~1e-15), not merely "some" valid one.

See `su2_to_euler_angles` in `_euler_angles.py` for the SU(2) sibling this
generalizes (which has no analogous ambiguity: SU(2)'s center is only
Z_2, and the existing `alt_param` construction already sidesteps it).
"""

# pylint: disable=invalid-name, too-many-locals
# pylint: disable=too-many-arguments, too-many-positional-arguments

import torch
import numpy as np

from ._euler_angles import resolve_coords, pack, unpack

TWO_PI = 2 * np.pi


__all__ = ["su3_to_euler_angles", "euler_angles_to_su3"]


_TWO_PI = 2 * np.pi
_TWO_PI_OVER_3 = 2 * np.pi / 3


def _wrap_pi(x):
    """Wrap an angle to (-pi, pi]."""
    return torch.atan2(torch.sin(x), torch.cos(x))


def _su3_to_raw_angles(matrix, channel_axis=-1):
    r"""Perform Euler decomposition of SU(3) matrices and return the angles.

    Generalizes `su2_to_euler_angles` so that the first and last factors are
    diagonal:

        U = D1(alpha1, alpha2) @ R23(theta23) @ U13(theta13, delta)
              @ R12(theta12) @ D2(beta1, beta2)

    where `D1 = diag(e^{i alpha1}, e^{i alpha2}, e^{-i(alpha1 + alpha2)})`,
    similarly `D2` with `(beta1, beta2)`; `R12`, `R23` are real rotations in
    the (1,2) and (2,3) planes; `U13` is a rotation in the (1,3) plane
    carrying the single complex phase `delta`. This is exactly the PDG
    "standard parametrization" of the CKM matrix -- `4 + 4 = 8 = dim(SU(3))`,
    and it reduces to `su2_to_euler_angles`'s `D(phi) @ R(theta) @ D(psi)`
    structure one dimension down.

    `theta12, theta13, theta23` are read off matrix-entry magnitudes (hence
    always in `[0, pi/2]`, unambiguous); `delta, alpha2, beta1, beta2` are in
    `(-pi, pi]`; `alpha1` is in `(-pi/3, pi/3]` (a third of the naive range
    -- see module docstring for why). Extracting `alpha1` involves dividing
    an angle by 3, which has an inherent 3-fold branch ambiguity (SU(3)'s
    center is Z_3 -- the same ambiguity
    `lattice_ml.random.sample_sun_group_commutator`'s `solve_lambda_su3` has
    to resolve for its cube root). All 3 candidate branches are reconstructed
    and compared against `matrix`, and the one that matches is kept and then
    canonicalized into `alpha1`'s fundamental domain. This makes the map
    exact in both directions (module docstring), at the cost of one
    non-differentiable branch selection -- gradients through the *chosen*
    branch are fine, but the selection itself isn't smooth at branch
    boundaries.

    Parameters
    ----------
    matrix : tensor
        The SU(3) matrix (or batch of matrices) to be mapped to Euler
        angles.

    channel_axis : int or None (optional)
        If integer, specifies the axis along which the 8 angles are stacked
        (default is -1); if None, returns them as a tuple instead.

    Returns
    -------
    (theta12, theta13, theta23, delta, alpha1, alpha2, beta1, beta2), or a
    tensor stacking them along `channel_axis`.
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
        alpha1_k = _wrap_pi(base + k * _TWO_PI / 3)
        alpha2_k = _wrap_pi((-alpha1_k + C - D) / 2)
        beta1_k = _wrap_pi(A - alpha1_k)
        beta2_k = _wrap_pi(B - alpha1_k)
        delta_k = _wrap_pi(A + B + C + D - E)

        U_k = _euler_angles_to_su3_matrix(
            theta12, theta13, theta23, delta_k,
            alpha1_k, alpha2_k, beta1_k, beta2_k,
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
    gamma = torch.round(alpha1 / _TWO_PI_OVER_3) * _TWO_PI_OVER_3
    alpha1 = _wrap_pi(alpha1 - gamma)
    alpha2 = _wrap_pi(alpha2 - gamma)
    beta1 = _wrap_pi(beta1 + gamma)
    beta2 = _wrap_pi(beta2 + gamma)

    out = (theta12, theta13, theta23, delta, alpha1, alpha2, beta1, beta2)
    if isinstance(channel_axis, int):
        return torch.stack(out, channel_axis)
    return out


def _euler_angles_to_su3_matrix(
    theta12, theta13, theta23, delta, alpha1, alpha2, beta1, beta2
):
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


def _raw_angles_to_su3(angles, channel_axis=-1):
    """Performing the opposite of su3_to_euler_angles.

    For details see su3_to_euler_angles.

    Parameters
    ----------
    angles : tensor or a tuple of tensors
        Euler decomposed angles: `(theta12, theta13, theta23, delta,
        alpha1, alpha2, beta1, beta2)`.

    channel_axis : int or None (optional)
        If integer, `angles` is a tensor with the 8 angles stacked along
        this axis (default is -1); if None, `angles` is a tuple instead.
    """
    if isinstance(channel_axis, int):
        (theta12, theta13, theta23, delta,
         alpha1, alpha2, beta1, beta2) = torch.unbind(angles, channel_axis)
    else:
        theta12, theta13, theta23, delta, alpha1, alpha2, beta1, beta2 = angles
    return _euler_angles_to_su3_matrix(
        theta12, theta13, theta23, delta, alpha1, alpha2, beta1, beta2
    )


# =============================================================================
def su3_log_jacobian(theta12, theta13, theta23, coords):
    r"""Return `log |det d(coords)/d(su(3))|`, up to an ADDITIVE CONSTANT.

    Same sign convention and same basis-dependence caveat as
    `su2_log_jacobian`: this is the FORWARD map `matrix -> coords`, and the
    constant is only meaningful through differences.

    For `coords='angles'` the Haar measure factorizes into a kernel
    `sin(t12) cos(t12) * sin(t13) cos(t13)^3 * sin(t23) cos(t23)` times a
    flat measure in the five phases -- the extra power of cosine being the
    same (1,3)-plane asymmetry that makes `U13` carry the phase `delta`.
    Verified against finite differences of `matrix -> coords` in
    left-invariant coordinates.
    """
    if coords == 'uniform':
        return torch.zeros_like(theta12)
    return -(
        torch.log(torch.sin(theta12)) + torch.log(torch.cos(theta12))
        + torch.log(torch.sin(theta13)) + 3 * torch.log(torch.cos(theta13))
        + torch.log(torch.sin(theta23)) + torch.log(torch.cos(theta23))
    )


# =============================================================================
def _to_coords(angles, coords):
    """Raw Euler angles -> the requested coordinates."""
    if coords == 'angles':
        return angles
    theta12, theta13, theta23, delta, alpha1, alpha2, beta1, beta2 = angles
    return (
        torch.sin(theta12)**2,          # kernel sin(t) cos(t)
        1 - torch.cos(theta13)**4,      # kernel sin(t) cos(t)^3
        torch.sin(theta23)**2,          # kernel sin(t) cos(t)
        delta / TWO_PI + 0.5,
        alpha1 * 3 / TWO_PI + 0.5,      # alpha1 lives in (-pi/3, pi/3]
        alpha2 / TWO_PI + 0.5,
        beta1 / TWO_PI + 0.5,
        beta2 / TWO_PI + 0.5,
    )


def _from_coords(param, coords):
    """Inverse of `_to_coords`."""
    if coords == 'angles':
        return param
    return (
        torch.asin(param[0].clamp(0, 1).sqrt()),
        torch.acos(((1 - param[1]).clamp(0, 1))**0.25),
        torch.asin(param[2].clamp(0, 1).sqrt()),
        (param[3] - 0.5) * TWO_PI,
        (param[4] - 0.5) * TWO_PI / 3,
        (param[5] - 0.5) * TWO_PI,
        (param[6] - 0.5) * TWO_PI,
        (param[7] - 0.5) * TWO_PI,
    )


# =============================================================================
def su3_to_euler_angles(matrix, channel_axis=None, coords=None,
                        return_logj=False):
    """Perform Euler decomposition of SU(3) matrices and return the
    coordinates.

    See `_su3_to_raw_angles` for the decomposition itself, including the
    Z_3-center redundancy it resolves. The coordinates are

        (theta12, theta13, theta23, delta, alpha1, alpha2, beta1, beta2)

    for `coords='angles'`, and for `coords='uniform'`

        (sin^2(theta12), 1 - cos^4(theta13), sin^2(theta23), and the five
         phases rescaled onto [0, 1]),

    each of which is uniform on [0, 1] under the Haar measure.

    Parameters
    ----------
    matrix : tensor
        the SU(3) matrix (or batch of matrices) to be decomposed.
    coords : {'angles', 'uniform'} or None (optional)
        Which coordinates to return; None means 'angles'. See the
        `_euler_angles` module docstring for the two conventions.

    channel_axis : int or None (optional)
        If integer, the 8 coordinates are stacked along this axis (default
        is -1); if None, they are returned as a tuple instead.

    return_logj : bool (optional)
        Also return `su3_log_jacobian(...)`, of shape `matrix.shape[:-2]`
        (default is False). It is NOT summed over any field axes.
    """
    coords = resolve_coords(coords)
    angles = _su3_to_raw_angles(matrix, channel_axis=None)
    out = pack(_to_coords(angles, coords), channel_axis)

    if not return_logj:
        return out
    return out, su3_log_jacobian(angles[0], angles[1], angles[2], coords)


# =============================================================================
def euler_angles_to_su3(param, channel_axis=None, coords=None,
                        return_logj=False):
    """Perform the opposite of `su3_to_euler_angles`.

    `coords` must match the value used there. With `return_logj=True` the
    returned log-Jacobian is the negative of the forward one, so the two
    cancel.
    """
    coords = resolve_coords(coords)
    angles = _from_coords(unpack(param, channel_axis, 8), coords)
    matrix = _euler_angles_to_su3_matrix(*angles)

    if not return_logj:
        return matrix
    return matrix, -su3_log_jacobian(angles[0], angles[1], angles[2], coords)
