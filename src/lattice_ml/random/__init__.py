# Created by Javad Komijani (2026)

"""
Functions and classes for generating random matrices from named distributions
(Haar on SU(n)/U(n), uniform on the maximal torus, Ginibre, ...).

Distinct from `lattice_ml.stats`, which analyzes samples you already have
(importance-sampling diagnostics, resampling, KDE, ...); this package
generates them in the first place. Plain one-shot draws follow `numpy.random`'s
naming (`rand_*_like`, mirroring `torch.rand_like`); the `torch.distributions`-
style classes (`UnGroup`, `SUnGroup`, `U1Group`, `GinibreCMatrixDist`) live
here too, since they exist to generate the same kind of random matrices.
"""

from ._ginibre_dist import *
from ._sample_sun_group_commutator import *
from ._unitary_group import *
