# Created by Javad Komijani (2026)

"""Coordinates on compact Lie groups.

Charts on SU(N) -- Euler angles, Lie-algebra coordinates and the
parametrizations they rest on -- as invertible maps between a group element
and real coordinates, each carrying the log-Jacobian of the change of
variables.
"""

from ._euler_angles_su2 import su2_to_euler_angles, euler_angles_to_su2
from ._euler_angles_su3 import su3_to_euler_angles, euler_angles_to_su3
from ._euler_angles_sun import sun_to_euler_angles, euler_angles_to_sun

from ._sun_group_commutator import *
from ._sun_group_commutator_density import *
from ._sun_group_commutator_solve import *
from ._su3_group_commutator_eigangles_dist import *
