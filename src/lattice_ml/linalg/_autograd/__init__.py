
import torch

from . import eig_autograd
from . import svd_autograd
from .reciprocal import Reciprocal

reciprocal = Reciprocal.apply

eigh = eig_autograd.Eigh.apply
eigu = eig_autograd.Eigu.apply
inverse_eign = eig_autograd.InverseEign.apply
inverse_eigh = eig_autograd.InverseEigh.apply

svd = svd_autograd.SVD.apply_wrapper
svd_with_simplified_ad = svd_autograd.ADSimplifiedSVD.apply_wrapper

from .project_grad_sun import project_grad_sun, project_data_and_grad_sun
