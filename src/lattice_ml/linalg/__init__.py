# Created by Javad Komijani (2024)

# functions from _autograd **reliably** support algorithmic differentiation
from ._autograd import eigh
from ._autograd import eigu
from ._autograd import inverse_eign
from ._autograd import svd
from ._autograd import svd_with_simplified_ad
from ._autograd import reciprocal

inverse_eig = inverse_eign  # for legacy

from ._autograd import project_grad_sun
