# *** Functions imported here do not have reliable autograd. ***

import torch

from . import generic

from .eig_2x2 import eigh2x2, eigu2x2
from .eigh_3x3 import eigh3x3, eigvalsh3x3
from .eig_3x3 import eign3x3, eigvals3x3
from .eigsu_3x3 import eigvalssu3x3
from .eigh_jacobi import jacobi_diagonalization


def eigh(matrix, **kwargs):
    match matrix.shape[-1]:
        case 2:
            return eigh2x2(matrix, **kwargs)
        case 3:
            return eigh3x3(matrix, **kwargs)
        case _:
            return torch.linalg.eigh(matrix, **kwargs)


def eigu(matrix, **kwargs):
    match matrix.shape[-1]:
        case 2:
            return eigu2x2(matrix, **kwargs)
        case 3:
            return eign3x3(matrix, **kwargs)  # using eign3x3
        case _:
            return torch.linalg.eig(matrix, **kwargs)
