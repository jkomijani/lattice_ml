#!/usr/bin/env python3
# Copyright (c) 2026 Javad Komijani

# pylint does not resolve torch's dynamically populated `linalg`
# submodule, so every call into it is reported as not-callable.
# pylint: disable=not-callable

# `haar_sun` / `su_generators` are repeated in each test module on purpose,
# so that every module runs standalone; do not factor them out.
# pylint: disable=duplicate-code

"""Tests for `sun_to_algebra` / `algebra_to_sun` and their log-Jacobians.

The checks that matter, in order of how badly a regression would hurt:

1. the log-Jacobian SIGN convention -- a flip here silently breaks every
   flow built on these coordinates, with no exception raised and no
   obviously wrong number printed;
2. the log-Jacobian VALUE, against finite differences of the map itself.
   `dim(su(N)) = N^2 - 1` equals the number of coefficients, so the map is
   square and a finite-difference log-det is exact, not merely indicative;
3. the `theta^(dim-1)` polar factor of `repr='axis_angle'`, which needs its
   OWN finite-difference check: the round trip and the sign cancellation
   pass whatever that term is, because the forward and inverse maps share
   the expression;
4. round trips in BOTH directions, the second being the canonicalization
   property a flow depends on;
5. that the eigenangles really do land in the principal cell, which is the
   whole reason `algebra_to_sun` can be called an inverse.

Run with:  python3 tests/test_lie_algebra.py -v
      or:  python3 -m unittest tests.test_lie_algebra -v
"""

import unittest

import numpy as np
import torch

from lattice_ml.lie_groups._lie_algebra import (
    sun_to_algebra, algebra_to_sun
)

# The set of valid representations is not public API (the analogous Euler
# `COORDS` is not exported either), so keep a local copy rather than reaching
# into a private path. Extend this when a representation is added.
ALGEBRA_REPRS = ('matrix', 'coeffs', 'axis_angle')

torch.set_default_dtype(torch.float64)

N_SAMPLES = 4096
N_COEFFS = {2: 3, 3: 8}


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


def _chart(matrix, repr_, tangent=None):
    """The map evaluated at one matrix, flattened to a plain coordinate
    vector so that a log-det can be taken.

    For `repr='axis_angle'` the raw output `(theta, t, modified_theta)` is
    not a chart: `t` is a unit vector, so it carries `dim - 1` degrees of
    freedom in `dim` components, and `modified_theta` is a function of the
    other two. Projecting `t` onto an orthonormal basis of its own tangent
    space at the base point restores a square map.
    """
    out = sun_to_algebra(matrix.unsqueeze(0), repr=repr_)
    if repr_ != 'axis_angle':
        return out[0]
    theta, t, _ = out
    return torch.cat([theta[0].reshape(1), t[0] @ tangent])


def _tangent_basis(t):
    """Orthonormal basis of the complement of `t` in R^dim."""
    dim = t.shape[0]
    stacked = torch.cat([t.unsqueeze(-1), torch.eye(dim, dtype=t.dtype)], -1)
    q, _ = torch.linalg.qr(stacked)
    return q[:, 1:]


def fd_log_jacobian(matrix, n, repr_, eps=1e-6):
    """log |det d(repr)/d(su(n))| by central differences, at one matrix."""
    tangent = None
    if repr_ == 'axis_angle':
        _, t, _ = sun_to_algebra(matrix.unsqueeze(0), repr=repr_)
        tangent = _tangent_basis(t[0])
    basis = su_generators(n)
    rows = []
    for k in range(basis.shape[0]):
        plus = matrix @ torch.matrix_exp(eps * basis[k])
        minus = matrix @ torch.matrix_exp(-eps * basis[k])
        rows.append(
            (_chart(plus, repr_, tangent) - _chart(minus, repr_, tangent))
            / (2 * eps)
        )
    return float(torch.linalg.slogdet(torch.stack(rows))[1])


class LieAlgebraTest(unittest.TestCase):
    """Checks on the su(N) coordinate maps; see the module docstring."""

    def test_log_jacobian_sign_cancels(self):
        """Forward and inverse log-Jacobians must cancel exactly."""
        for n in (2, 3):
            matrix = haar_sun(n, N_SAMPLES)
            for repr_ in ALGEBRA_REPRS:
                with self.subTest(n=n, repr=repr_):
                    x, logj_fwd = sun_to_algebra(
                        matrix, repr=repr_, return_logj=True
                    )
                    _, logj_bwd = algebra_to_sun(
                        x, repr=repr_, return_logj=True
                    )
                    self.assertEqual(logj_fwd.shape, matrix.shape[:-2])
                    self.assertLess(
                        float((logj_fwd + logj_bwd).abs().max()), 1e-11
                    )

    def test_log_jacobian_matches_finite_differences(self):
        """The closed form must equal the finite-difference Jacobian.

        Unlike the Euler case there is no free additive constant to absorb:
        `su_generators` uses the same `Tr(T_a T_b) = delta_ab` normalization
        as the coefficient maps, so the residual must vanish outright. The
        constancy check is the primary one; the zero-mean check additionally
        pins the `1/sqrt(2)` in the generator bases.
        """
        for n in (2, 3):
            for repr_ in ('coeffs', 'axis_angle'):
                with self.subTest(n=n, repr=repr_):
                    residuals = []
                    for seed in range(6):
                        matrix = haar_sun(n, 1, seed=100 + seed)[0]
                        _, logj = sun_to_algebra(
                            matrix.unsqueeze(0), repr=repr_, return_logj=True
                        )
                        residuals.append(
                            fd_log_jacobian(matrix, n, repr_) - float(logj[0])
                        )
                    self.assertLess(float(np.std(residuals)), 1e-6)
                    self.assertLess(abs(float(np.mean(residuals))), 1e-6)

    def test_axis_angle_polar_factor(self):
        """The two representations must differ by exactly the polar factor
        of `R^dim -> (radius, direction)`, and by nothing else."""
        for n in (2, 3):
            with self.subTest(n=n):
                matrix = haar_sun(n, N_SAMPLES)
                _, logj_polar = sun_to_algebra(
                    matrix, repr='axis_angle', return_logj=True
                )
                coeffs, logj_flat = sun_to_algebra(
                    matrix, repr='coeffs', return_logj=True
                )
                dim = coeffs.shape[-1]
                theta = coeffs.norm(dim=-1)
                residual = logj_polar - logj_flat + (dim - 1) * theta.log()
                self.assertLess(float(residual.abs().max()), 1e-12)

    def test_matrix_round_trip(self):
        """matrix -> algebra -> matrix."""
        for n in (2, 3):
            matrix = haar_sun(n, N_SAMPLES)
            for repr_ in ALGEBRA_REPRS:
                with self.subTest(n=n, repr=repr_):
                    x = sun_to_algebra(matrix, repr=repr_)
                    rebuilt = algebra_to_sun(x, repr=repr_)
                    self.assertLess(
                        float((matrix - rebuilt).abs().max()), 1e-11
                    )

    def test_coord_round_trip(self):
        """algebra -> matrix -> algebra: the canonicalization property a flow
        relies on, since it applies a transform between the two halves."""
        for n in (2, 3):
            matrix = haar_sun(n, N_SAMPLES)
            for repr_ in ('matrix', 'coeffs'):
                with self.subTest(n=n, repr=repr_):
                    x = sun_to_algebra(matrix, repr=repr_)
                    again = sun_to_algebra(
                        algebra_to_sun(x, repr=repr_), repr=repr_
                    )
                    self.assertLess(float((x - again).abs().max()), 1e-10)

    def test_generator_is_traceless_hermitian(self):
        """`repr='matrix'` must return an su(N) element, and `exp(iH)` must
        give back the group element it came from."""
        for n in (2, 3):
            with self.subTest(n=n):
                matrix = haar_sun(n, N_SAMPLES)
                herm = sun_to_algebra(matrix, repr='matrix')
                self.assertLess(
                    float((herm - herm.adjoint()).abs().max()), 1e-11
                )
                trace = torch.einsum('...ii->...', herm)
                self.assertLess(float(trace.abs().max()), 1e-11)
                self.assertLess(
                    float((torch.matrix_exp(1j * herm) - matrix).abs().max()),
                    1e-11
                )

    def test_coefficients_are_orthonormal(self):
        """`Tr(H^2) = sum_a theta_a^2`, i.e. the generator basis really is
        normalized to `Tr(T_a T_b) = delta_ab`."""
        for n in (2, 3):
            with self.subTest(n=n):
                matrix = haar_sun(n, N_SAMPLES)
                herm = sun_to_algebra(matrix, repr='matrix')
                coeffs = sun_to_algebra(matrix, repr='coeffs')
                self.assertEqual(coeffs.shape[-1], N_COEFFS[n])
                trace_sq = torch.einsum(
                    '...ij,...ji->...', herm, herm
                ).real
                residual = trace_sq - coeffs.pow(2).sum(-1)
                self.assertLess(float(residual.abs().max()), 1e-11)

    def test_eigenangles_lie_in_the_principal_cell(self):
        """The property that makes `algebra_to_sun` an inverse at all.

        SU(2): `|theta| <= pi`. SU(3): `w = (max - min) / (2 pi) <= 1`.
        Both are the same statement about the width of the eigenangle
        spectrum, so both are checked that way.
        """
        for n in (2, 3):
            with self.subTest(n=n):
                herm = sun_to_algebra(haar_sun(n, 200000), repr='matrix')
                angles = torch.linalg.eigvalsh(herm)
                width = (angles.amax(-1) - angles.amin(-1)) / (2 * np.pi)
                self.assertLessEqual(float(width.max()), 1.0 + 1e-12)

    def test_modified_theta(self):
        """The direction-independent radius: `theta` itself for SU(2), where
        there is no shape freedom, and `(max - min) / 2` for SU(3)."""
        for n in (2, 3):
            with self.subTest(n=n):
                matrix = haar_sun(n, N_SAMPLES)
                theta, t, modified = sun_to_algebra(
                    matrix, repr='axis_angle'
                )
                self.assertLess(float((t.norm(dim=-1) - 1).abs().max()), 1e-12)
                if n == 2:
                    expected = theta
                else:
                    angles = torch.linalg.eigvalsh(
                        sun_to_algebra(matrix, repr='matrix')
                    )
                    expected = (angles.amax(-1) - angles.amin(-1)) / 2
                self.assertLess(
                    float((modified - expected).abs().max()), 1e-11
                )

    def test_invalid_arguments(self):
        """Every rejected argument must raise, not silently coerce."""
        matrix = haar_sun(2, 8)
        with self.assertRaises(ValueError):
            sun_to_algebra(matrix, repr='angles')
        with self.assertRaises(ValueError):
            sun_to_algebra(matrix, repr='hermitian')
        su5 = torch.eye(5, dtype=torch.complex128).expand(2, 5, 5)
        with self.assertRaises(ValueError):       # N = 5 is not implemented
            sun_to_algebra(su5)
        with self.assertRaises(ValueError):       # 5 != N^2 - 1
            algebra_to_sun(torch.rand(4, 5), repr='coeffs')


if __name__ == '__main__':
    unittest.main()
