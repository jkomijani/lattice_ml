# Copyright (c) 2023-2026 Javad Komijani

r"""
Euler decomposition of SU(2) matrices.

Two coordinate conventions are available everywhere, selected with `coords`:

    'angles'   the raw Euler angles, in radians.
    'uniform'  the raw Euler angles are transformed such that the distribution
               of each coordinate is uniform on [0, 1] and the log-Jacobian is
               identically zero.
"""

import torch
import numpy as np

TWO_PI = 2 * np.pi

__all__ = ['su2_to_euler_angles', 'euler_angles_to_su2']


# =============================================================================
def su2_to_euler_angles(
    matrix,
    channel_axis=None,
    coords=None,
    return_logj=False
):
    r"""
    Perform Euler decomposition of SU(2) matrices and return the coordinates.

    Here we closely follow [Ref](https://ncatlab.org/nlab/show/Euler+angle);
    see (3.3.21) of Sakurai's book with a slightly different parametrization.
    The decomposition is `M = D(phi) R(theta) D(psi)`, and

        >>> phi, theta, psi = su2_to_euler_angles(matrix)
        >>> a, b, c = su2_to_euler_angles(matrix, coords='uniform')

    where

        >>> a = angle(M_00) / (2 pi) + 0.5      # angl(M_00)   = (phi + psi) /2
        >>> b = |M_00|^2                        # |M_00| = cos(theta/2)
        >>> c = angle(-1j M_01) / (2 pi) + 0.5  # angl(M_01/i) = (phi - psi) /2

    all three of which are uniform on [0, 1]. The modulus channel `b` sits in
    the middle, matching `theta` in the `angles` convention. The outer two
    carry the same information in a different basis -- `phi` and `psi` are
    the sum and difference of the two phases that `a` and `c` give directly.

    Parameters
    ----------
    matrix : tensor
        the matrix to be decomposed.

    channel_axis : int or None (optional)
        If integer, the coordinates are stacked along this axis;
        if None (default), they are returned as a tuple instead.

    coords : {'angles', 'uniform'} or None (optional)
        Which coordinates to return; None means 'angles'. See the module
        docstring for what the two conventions are and why there are only two.

    return_logj : bool (optional)
        Also return `su2_log_jacobian(...)`, of shape `matrix.shape[:-2]`
        (default is False). It is NOT summed over any field axes -- that is
        the caller's job.
    """
    coords = resolve_coords(coords)

    abs00 = torch.abs(matrix[..., 0, 0])        # \in [0, 1]
    angle00 = torch.angle(matrix[..., 0, 0])    # \in (-pi, pi]
    angle01 = torch.angle(-1j * matrix[..., 0, 1])

    if coords == 'angles':
        out = (
            angle00 + angle01,                      # phi \in (-2 pi, 2 pi]
            2 * torch.acos(abs00.clamp(max=1.0)),   # theta \in [0, pi]
            angle00 - angle01,                      # psi \in (-2 pi, 2 pi]
        )
    else:
        out = (
            angle00 / TWO_PI + 0.5,
            abs00**2,
            angle01 / TWO_PI + 0.5,
        )

    out = pack(out, channel_axis)

    if not return_logj:
        return out
    return out, su2_log_jacobian(abs00, coords)


# =============================================================================
def euler_angles_to_su2(
    param,
    channel_axis=None,
    coords=None,
    return_logj=False
):
    """Perform the opposite of `su2_to_euler_angles`.

    For details see `su2_to_euler_angles`; `coords` must match the value
    used there. With `return_logj=True` the returned log-Jacobian is the
    negative of the forward one, so that the two cancel.
    """
    coords = resolve_coords(coords)
    c_0, c_1, c_2 = unpack(param, channel_axis, 3)

    if coords == 'angles':
        phi, theta, psi = c_0, c_1, c_2
        abs00 = torch.cos(theta / 2)
        angle00 = (phi + psi) / 2
        angle01 = (phi - psi) / 2
    else:
        angle00 = TWO_PI * (c_0 - 0.5)
        abs00 = c_1.clamp(0, 1).sqrt()
        angle01 = TWO_PI * (c_2 - 0.5)

    m00 = abs00 * torch.exp(1j * angle00)
    m01 = 1j * torch.sqrt((1 - abs00**2).clamp_min(0)) * torch.exp(1j*angle01)

    matrix = torch.stack([m00, m01, -m01.conj(), m00.conj()], dim=-1)
    matrix = matrix.reshape(*m00.shape, 2, 2)

    if not return_logj:
        return matrix
    return matrix, -su2_log_jacobian(abs00, coords)


# =============================================================================
def su2_log_jacobian(abs00, coords):
    r"""
    Return `log |det d(coords)/d(su(2))|`, up to an additive constant, which
    cancels anyway between the forward and inverse maps.

    SIGN: this is the Jacobian of the FORWARD map `matrix -> coords`;
    `euler_angles_to_su2` returns its negative. A sign error here silently
    breaks every flow built on this handle, so it is pinned by a test.

    Note: See eq (4.8) in arXiv:math-ph/0210033.

    Args:
        abs00: `|M_00| = cos(theta/2)`, of shape `matrix.shape[:-2]`.
        coords: see `su2_to_euler_angles`.
    """
    if coords == 'uniform':
        return torch.zeros_like(abs00)
    # 'angles': the Haar kernel is sin(theta) = 2 abs00 sqrt(1 - abs00^2)
    sin_theta = 2 * abs00 * torch.sqrt((1 - abs00**2).clamp_min(0))
    return -torch.log(sin_theta)


# =============================================================================
# Coordinate conventions, shared with `_euler_angles_su3.py`
# =============================================================================
COORDS = ('angles', 'uniform')


def resolve_coords(coords):
    """Validate `coords`, mapping None onto the default."""
    if coords is None:
        return 'angles'
    if coords not in COORDS:
        raise ValueError(f"coords must be one of {COORDS}, got {coords!r}")
    return coords


def pack(out, channel_axis):
    """Stack coordinates along `channel_axis`, or leave them as a tuple."""
    if isinstance(channel_axis, int):
        return torch.stack(out, channel_axis)
    return out


def unpack(param, channel_axis, n_expected):
    """Inverse of `pack`, with an arity check."""
    if isinstance(channel_axis, int):
        out = torch.unbind(param, channel_axis)
    else:
        out = tuple(param)
    if len(out) != n_expected:
        raise ValueError(f"expected {n_expected} coordinates, got {len(out)}")
    return out
