# Copyright (c) 2026 Javad Komijani

"""
Maps SU(3) eigenangles onto a rectangle (w, r), concatenated into
a single tensor of shape `(..., 2)`; both w and r are in [0, 1].
"""

import torch

from ._ordering import ZeroSumOrder


__all__ = [
    "su3_eigangs_to_rectangle",
    "rectangle_to_su3_eigangs",
]


_TWO_PI = 2 * torch.pi
_UNIT_SCALE = 3 / (4 * torch.pi)
_UNIT_SHIFT = 0.5


def su3_eigangs_to_rectangle(eigangs):
    r"""
    Map raw SU(3) eigenangles onto a rectangle `(w, r)`:

    .. math::

        w = (z - x) / (2 \pi) \in [0, 1] \\
        r = (y / w) * 3 / (4 \pi) + 1/2 \in [0, 1]

    where `(x, y, z)` is `eigangs` sorted ascending and shifted to sum to zero.
    The `r` rescaling is a constant (`w`-independent) Jacobian factor, dropped
    along with the other additive constants below.

    Parameters
    ----------
    eigangs : tensor
        Shape `(..., 3)`, three eigenangles in any order, summing to a
        multiple of `2 pi` (e.g. `torch.angle(eigvals)`).

    Returns
    -------
    w_r : tensor
        Shape `(..., 2)`, the concatenation of `(w, r)`.

    sorted_ind : tensor
        Sorting permutation; pass back into `rectangle_to_su3_eigangs` to
        recover the original order.

    logj : tensor
        Log-Jacobian up to an additive constant, summed over all but the
        batch axis. The zero-sum-and-sort step contributes none: sorting
        is a volume-preserving relabeling, and shifting by an exact
        multiple of `2 pi` doesn't move the represented point, only its
        coordinate representative.
    """
    order = ZeroSumOrder(eigangs)
    w_r, logj = su3_sorted_angles_to_rectangle(order.sorted_val)
    return w_r, order.sorted_ind, logj


def rectangle_to_su3_eigangs(w_r, sorted_ind):
    r"""
    Map a `(w, r)` rectangle point back to SU(3) eigenangles.

    Inverse of `su3_eigangs_to_rectangle`.

    Parameters
    ----------
    w_r : tensor
        Shape `(..., 2)`, as returned by `su3_eigangs_to_rectangle`.

    sorted_ind : tensor
        The permutation returned by `su3_eigangs_to_rectangle`, used to
        restore the original eigenangle order.

    Returns
    -------
    eigangs : tensor
        Shape `(..., 3)`, in `sorted_ind`'s original order. If an angle
        needed a winding correction going forward, it comes back the same
        physical angle mod `2 pi`, not the same real number.

    logj : tensor
        Log-Jacobian up to an additive constant.
    """
    sorted_angles, logj = rectangle_to_su3_sorted_angles(w_r)
    eigangs = sorted_angles.gather(-1, torch.argsort(sorted_ind, dim=-1))
    return eigangs, logj


def su3_sorted_angles_to_rectangle(sorted_angles):
    """
    Map an already sorted, zero-sum eigenangle triple onto `(w, r)`.

    The pure, stateless step `su3_eigangs_to_rectangle` composes with
    `ZeroSumOrder` -- see there for the formula. Not in `__all__` for
    now, but kept public-looking since it may become part of the public
    API later.
    """
    x, y, z = sorted_angles.split((1, 1, 1), dim=-1)
    w = (z - x) / _TWO_PI
    w_safe = torch.where(w == 0, torch.ones_like(w), w)
    r = torch.where(w == 0, torch.zeros_like(y), y / w_safe)
    r = r * _UNIT_SCALE + _UNIT_SHIFT
    logj = -sum_density(torch.log(w_safe))
    # additive constant `log(8/3 * pi**2)` dropped, as elsewhere here
    return torch.cat((w, r), dim=-1), logj


def rectangle_to_su3_sorted_angles(w_r):
    """
    Map `(w, r)` back to a sorted, zero-sum eigenangle triple.

    Inverse of `su3_sorted_angles_to_rectangle`.
    """
    w, r = w_r.split((1, 1), dim=-1)
    r = (r - _UNIT_SHIFT) / _UNIT_SCALE
    y = w * r
    z = -y / 2 + w * torch.pi
    x = -y / 2 - w * torch.pi
    w_safe = torch.where(w == 0, torch.ones_like(w), w)
    logj = sum_density(torch.log(w_safe))
    # additive constant `log(8/3 * pi**2)` dropped, as elsewhere here
    return torch.cat((x, y, z), dim=-1), logj


def sum_density(x: torch.Tensor):
    """Compute the sum over all, but the batch, axes."""
    ndim = x.dim()
    return x if ndim < 2 else torch.sum(x, dim=list(range(1, ndim)))
