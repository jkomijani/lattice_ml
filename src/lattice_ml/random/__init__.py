# Created by Javad Komijani (2026)

"""Functions for generating random matrices from named distributions
(Haar on SU(n)/U(n), uniform on the maximal torus, ...).

Distinct from `lattice_ml.stats`, which analyzes samples you already have
(importance-sampling diagnostics, resampling, KDE, ...); this package
generates them in the first place -- matching `numpy.random`'s naming, not
`torch.distributions`'s, since the content here is plain functions
(`rand_*_like`, mirroring `torch.rand_like`), not distribution objects.
"""

from ._unitary_group import rand_sun_group_like, rand_diagonal_sun_group_like
from ._unitary_group import UnGroup, SUnGroup, U1Group

from ._ginibre_dist import GinibreCMatrixDist
