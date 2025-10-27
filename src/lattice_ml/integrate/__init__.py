# Created By Javad Komijani (2024)

# ======================
# Standard ODE solvers
from ._odeint import odeint

# ODE evolution modules
from ._odeflow import ODEFlow  # Evolves state variables
from ._odeflow import ODEFlow_  # Also returns log-Jacobian

# Adjoint-based solvers
from ._adjoint import AdjODEFlow_  # Adjoint method for backprop ODEFlow_
from ._adjoint import AdjModule  # Utility for custom adjoint-based modules
from ._adjoint import TupleVar  # Helper for handling tuple of variables

# ==========================
# Lie group–aware ODE solvers
from ._lie_group_odeint import lie_odeint

from ._lie_group_odeflow import LieODEFlow
from ._lie_group_odeflow import LieODEFlow_

from ._lie_group_adjoint import AdjLieODEFlow_
from ._lie_group_adjoint import AdjLieModule

# ==========================
# Symplectic ODE solvers
from ._symplectic_odeint import symplectic_odeint
from ._symplectic_odeint import lie_symplectic_odeint
from ._symplectic_odeint import u1_symplectic_odeint

from ._symplectic_adjoint import adjoint_symplectic_odeint

from ._symplectic_odeflow import SymplecticODEFlow
from ._symplectic_odeflow import SymplecticODEFlow_
