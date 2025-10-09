# Created by Javad Komijani (2024)

# functions from functions **reliably** support algorithmic differentiation

from ._spectral_split_cat import spectral_split
from ._spectral_split_cat import spectral_cat
from ._spectral_split_cat import splitted_fftn
from ._spectral_split_cat import splitted_ifftn

from ._matrix_func import *
from ._project import *

from ._matrix_func_and_jacobian import commutator_and_jacobian
from ._matrix_func_and_jacobian import inverse_eign_and_jacobian

from . import _matrix_func_and_jacobian

inverse_eig_and_jacobian = inverse_eign_and_jacobian  # for legacy

matrix_exp1jh_and_jacobian = _matrix_func_and_jacobian.MatrixExp1jh()
matrix_angleu_and_jacobian = _matrix_func_and_jacobian.MatrixAngleU()
