# Created by Javad Komijani (2024)

# functions from functions **reliably** support algorithmic differentiation

from ._spectral_split_cat import spectral_split
from ._spectral_split_cat import spectral_cat
from ._spectral_split_cat import splitted_fftn
from ._spectral_split_cat import splitted_ifftn

from ._matrix_func import *
from ._project import *

from ._matrix_func_and_jacobian import *
