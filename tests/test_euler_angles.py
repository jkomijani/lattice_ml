#!/usr/bin/env python3
# Copyright (c) 2026 Javad Komijani

# pylint does not resolve torch's dynamically populated `linalg`
# submodule, so every call into it is reported as not-callable.
# pylint: disable=not-callable

# `haar_sun` / `su_generators` are repeated in each test module on purpose,
# so that every module runs standalone; do not factor them out.
# pylint: disable=duplicate-code

"""Tests for the SU(2)/SU(3) Euler decompositions and their log-Jacobians.

The checks that matter, in order of how badly a regression would hurt:

1. the log-Jacobian SIGN convention -- a flip here silently breaks every
   flow built on these coordinates, with no exception raised and no
   obviously wrong number printed;
2. the log-Jacobian VALUE, against finite differences of the map itself;
3. round trips in BOTH directions -- `matrix -> coords -> matrix` and
   `coords -> matrix -> coords`, the latter being the canonicalization
   property a flow depends on, and the one usually left unchecked;
4. that `coords='uniform'` really is uniform under the Haar measure.

Run with:  python3 tests/test_euler_angles.py -v
      or:  python3 -m unittest tests.test_euler_angles -v
"""

import unittest

import numpy as np
import torch

from lattice_ml.linalg import (
    su2_to_euler_angles, euler_angles_to_su2,
    sun_to_euler_angles, euler_angles_to_sun,
)
# The set of valid conventions is not public API, and the module defining it
# has already been renamed once, so keep a local copy rather than reaching into
# a private path. Extend this when a convention is added.
EULER_COORDS = ('angles', 'uniform')

torch.set_default_dtype(torch.float64)

N_SAMPLES = 4096
N_COORDS = {2: 3, 3: 8}


def haar_sun(n, batch, seed=0):
    """Batch of Haar-random SU(n) matrices."""
    torch.manual_seed(seed)
    z = torch.randn(batch, n, n, dtype=torch.complex128) / 2**0.5
    q, r = torch.linalg.qr(z)
    phase = torch.diagonal(r, dim1=-2, dim2=-1)
    q = q * (phase / phase.abs()).unsqueeze(-2)
    return q * torch.linalg.det(q)[..., None, None] ** (-1.0 / n)


def su_generators(n):
    """Orthonormal basis of su(n): off-diagonal pairs plus Helmert diagonal.

    Normalized to `Tr(T_a T_b) = delta_ab`, the same convention the
    coefficient maps use, which is what lets the finite-difference constant
    below be checked against zero rather than merely against itself.
    """
    basis = []
    for i in range(n):
        for j in range(i + 1, n):
            e = torch.zeros(n, n, dtype=torch.complex128)
            e[i, j], e[j, i] = 1, -1
            basis.append(e / 2**0.5)
            e = torch.zeros(n, n, dtype=torch.complex128)
            e[i, j], e[j, i] = 1j, 1j
            basis.append(e / 2**0.5)
    v = torch.zeros(n, n - 1)
    for k in range(1, n):
        v[:k, k - 1] = 1.0 / (k * (k + 1.0)) ** 0.5
        v[k, k - 1] = -(k / (k + 1.0)) ** 0.5
    for c in range(n - 1):
        basis.append(1j * torch.diag(v[:, c]).to(torch.complex128))
    return torch.stack(basis)


def fd_log_jacobian(matrix, n, coords, eps=1e-6):
    """log |det d(coords)/d(su(n))| by central differences, at one matrix."""
    basis = su_generators(n)
    rows = []
    for k in range(basis.shape[0]):
        plus = matrix @ torch.matrix_exp(eps * basis[k])
        minus = matrix @ torch.matrix_exp(-eps * basis[k])
        kwargs = {'coords': coords, 'channel_axis': -1}
        diff = (sun_to_euler_angles(plus.unsqueeze(0), **kwargs)[0]
                - sun_to_euler_angles(minus.unsqueeze(0), **kwargs)[0])
        rows.append(diff / (2 * eps))
    return float(torch.linalg.slogdet(torch.stack(rows))[1])


class EulerAnglesTest(unittest.TestCase):
    """Checks on the Euler coordinate maps; see the module docstring."""

    def test_log_jacobian_sign_cancels(self):
        """Forward and inverse log-Jacobians must cancel exactly."""
        for n in (2, 3):
            matrix = haar_sun(n, N_SAMPLES)
            for coords in EULER_COORDS:
                with self.subTest(n=n, coords=coords):
                    param, logj_fwd = sun_to_euler_angles(
                        matrix, coords=coords, channel_axis=-1,
                        return_logj=True
                    )
                    _, logj_bwd = euler_angles_to_sun(
                        param, coords=coords, channel_axis=-1,
                        return_logj=True
                    )
                    self.assertEqual(logj_fwd.shape, matrix.shape[:-2])
                    self.assertLess(
                        float((logj_fwd + logj_bwd).abs().max()), 1e-12
                    )

    def test_log_jacobian_matches_finite_differences(self):
        """The closed form must equal the finite-difference Jacobian up to
        ONE additive constant. That constant is basis dependent, so only its
        constancy across points carries meaning."""
        for n in (2, 3):
            for coords in EULER_COORDS:
                with self.subTest(n=n, coords=coords):
                    residuals = []
                    for seed in range(6):
                        matrix = haar_sun(n, 1, seed=100 + seed)[0]
                        _, logj = sun_to_euler_angles(
                            matrix.unsqueeze(0), coords=coords,
                            channel_axis=-1, return_logj=True
                        )
                        residuals.append(
                            fd_log_jacobian(matrix, n, coords) - float(logj[0])
                        )
                    self.assertLess(float(np.std(residuals)), 1e-6)

    def test_matrix_round_trip(self):
        """matrix -> coords -> matrix."""
        for n in (2, 3):
            matrix = haar_sun(n, N_SAMPLES)
            for coords in EULER_COORDS:
                with self.subTest(n=n, coords=coords):
                    param = sun_to_euler_angles(
                        matrix, coords=coords, channel_axis=-1
                    )
                    self.assertEqual(param.shape[-1], N_COORDS[n])
                    rebuilt = euler_angles_to_sun(
                        param, coords=coords, channel_axis=-1
                    )
                    self.assertLess(
                        float((matrix - rebuilt).abs().max()), 1e-12
                    )

    def test_coord_round_trip(self):
        """coords -> matrix -> coords: the canonicalization property a flow
        relies on, since it applies a transform between the two halves."""
        for n in (2, 3):
            matrix = haar_sun(n, N_SAMPLES)
            for coords in EULER_COORDS:
                with self.subTest(n=n, coords=coords):
                    param = sun_to_euler_angles(
                        matrix, coords=coords, channel_axis=-1
                    )
                    again = sun_to_euler_angles(
                        euler_angles_to_sun(
                            param, coords=coords, channel_axis=-1
                        ),
                        coords=coords, channel_axis=-1
                    )
                    self.assertLess(float((param - again).abs().max()), 1e-10)

    def test_uniform_coords_are_uniform_and_flat(self):
        """`coords='uniform'` must be uniform on [0, 1] under the Haar
        measure, which is what makes its log-Jacobian identically zero."""
        for n in (2, 3):
            with self.subTest(n=n):
                matrix = haar_sun(n, 200000)
                param, logj = sun_to_euler_angles(
                    matrix, coords='uniform', channel_axis=-1, return_logj=True
                )
                self.assertEqual(float(logj.abs().max()), 0.0)
                self.assertTrue(bool(((param >= 0) & (param <= 1)).all()))
                # Loose enough not to flake, tight enough to catch a wrong
                # CDF (that would give a KS statistic of order 0.01-0.1).
                band = 5.0 / np.sqrt(param.shape[0])
                for k in range(N_COORDS[n]):
                    x, _ = torch.sort(param[:, k])
                    grid = torch.arange(
                        1, len(x) + 1, dtype=torch.float64
                    ) / len(x)
                    self.assertLess(float((x - grid).abs().max()), band)

    def test_tuple_interface(self):
        """channel_axis=None is the default, so this is the plain call."""
        for n in (2, 3):
            with self.subTest(n=n):
                matrix = haar_sun(n, 128)
                param = sun_to_euler_angles(matrix, coords='uniform')
                self.assertIsInstance(param, tuple)
                self.assertEqual(len(param), N_COORDS[n])
                rebuilt = euler_angles_to_sun(param, coords='uniform')
                self.assertLess(float((matrix - rebuilt).abs().max()), 1e-12)

    def test_invalid_arguments(self):
        """Every rejected argument must raise, not silently coerce."""
        matrix = haar_sun(2, 8)
        with self.assertRaises(ValueError):
            su2_to_euler_angles(matrix, coords='unit')
        with self.assertRaises(ValueError):
            su2_to_euler_angles(matrix, coords='haar')
        with self.assertRaises(TypeError):        # alt_param is gone
            # pylint: disable-next=unexpected-keyword-arg
            su2_to_euler_angles(matrix, alt_param=True)
        with self.assertRaises(ValueError):
            euler_angles_to_su2(torch.rand(4, 5), coords='uniform')
        with self.assertRaises(ValueError):
            sun_to_euler_angles(torch.eye(4, dtype=torch.complex128))

    def test_su2_uniform_is_the_square_of_abs00(self):
        """Migration anchor: the removed `alt_param` returned abs00;
        'uniform' returns abs00**2 and leaves the phase channels alone.

        Channel order anchor too: the modulus channel is the MIDDLE one, so
        that `(s, a, d)` lines up with `(phi, theta, psi)`. It used to be
        first; a silent revert would flip which channel a flow transforms."""
        matrix = haar_sun(2, N_SAMPLES)
        a, b, c = su2_to_euler_angles(matrix, coords='uniform')
        abs00 = matrix[..., 0, 0].abs()
        self.assertLess(float((b - abs00**2).abs().max()), 1e-15)

        # and the phases really are the outer two, in this order
        phase_sum = torch.angle(matrix[..., 0, 0]) / (2 * np.pi) + 0.5
        phase_diff = torch.angle(-1j * matrix[..., 0, 1]) / (2 * np.pi) + 0.5
        self.assertLess(float((a - phase_sum).abs().max()), 1e-15)
        self.assertLess(float((c - phase_diff).abs().max()), 1e-15)


if __name__ == '__main__':
    unittest.main()
