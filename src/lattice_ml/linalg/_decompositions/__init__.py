# Created by Javad Komijani (2026)

"""General-purpose matrix decompositions and parametrizations, not tied to
any particular flow architecture (moved here from `normflow.lib.linalg`,
which now re-exports these for backward compatibility)."""

import torch

from ._qr import haar_qr, haar_sqr, qr_on_cpu

from ._euler_angles_su2 import *
from ._euler_angles_su3 import *


# =============================================================================
def sun_to_euler_angles(
    matrix, coords=None, channel_axis=None, return_logj=False
):
    """Euler decomposition of SU(N) matrices, dispatched on `matrix.shape[-1]`.

    N = 2 and 3 are implemented; the signature is N-agnostic, so further N
    only need their own `suN_to_euler_angles` registered below.

    Note that `sun` here means "dispatches to whichever N is implemented",
    as in `encode_sun_group_commutator`; not one closed form valid for every N.
    """
    kwargs = dict(
        channel_axis=channel_axis, coords=coords, return_logj=return_logj
    )
    n = matrix.shape[-1]
    if n == 2:
        return su2_to_euler_angles(matrix, **kwargs)
    if n == 3:
        return su3_to_euler_angles(matrix, **kwargs)
    raise ValueError(f"N = {n} is not supported")


# =============================================================================
def euler_angles_to_sun(
    param, coords=None, channel_axis=None, return_logj=False
):
    """Inverse of `sun_to_euler_angles`.

    N is inferred from the number of coordinates, `dim(SU(N)) = N^2 - 1`:
    3 -> SU(2), 8 -> SU(3).
    """
    kwargs = dict(
        channel_axis=channel_axis, coords=coords, return_logj=return_logj
    )
    if isinstance(channel_axis, int):
        n_coords = param.shape[channel_axis]
    else:
        n_coords = len(param)
    if n_coords == 3:
        return euler_angles_to_su2(param, **kwargs)
    if n_coords == 8:
        return euler_angles_to_su3(param, **kwargs)
    raise ValueError(f"cannot infer N from {n_coords} coordinates")
