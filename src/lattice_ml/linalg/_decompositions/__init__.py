# Created by Javad Komijani (2026)

"""General-purpose matrix decompositions and parametrizations, not tied to
any particular flow architecture (moved here from `normflow.lib.linalg`,
which now re-exports these for backward compatibility)."""

from ._qr import haar_qr, haar_sqr, qr_on_cpu

from ._euler_angles import su2_to_euler_angles, euler_angles_to_su2
