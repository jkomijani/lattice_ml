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


__all__ = [
    "su3_to_euler_angles",
    "euler_angles_to_su3",
]


_TWO_PI = 2 * np.pi
_TWO_PI_OVER_3 = 2 * np.pi / 3


def _wrap_pi(x):
    """Wrap an angle to (-pi, pi]."""
    return torch.atan2(torch.sin(x), torch.cos(x))


def su3_to_euler_angles(matrix, channel_axis=-1):
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


def euler_angles_to_su3(angles, channel_axis=-1):
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
