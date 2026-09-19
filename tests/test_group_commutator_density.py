#!/usr/bin/env python3
# Copyright (c) 2026 Claude, under supervision of Javad Komijani

# pylint does not resolve torch's dynamically populated `linalg`
# submodule, so every call into it is reported as not-callable.
# pylint: disable=not-callable
# pylint: disable=invalid-name

"""Tests for the group commutator density `J(Z)` of SU(2) and SU(3).

The check that matters is Schur orthogonality: for Haar-random, independent
X, Y in SU(N),

    <chi_R(X Y X† Y†)> = 1 / dim_R

for *every* irrep R. The characters are a complete basis for class functions,
so matching this across a range of R pins `J` down completely -- far more
than normalization plus a moment or two, which many wrong shapes also pass.

The remaining checks are the symmetries `J(g Z g†) = J(Z)` and `J(Z†) = J(Z)`,
and that SU(2)'s textbook `(pi - theta) / (2 sin theta)` is the same
one-factor-per-pair formula that SU(3) uses.

Run with:  python3 tests/test_group_commutator_density.py -v
      or:  python3 -m unittest tests.test_group_commutator_density -v
"""

import math
import unittest

import numpy as np
import torch

from lattice_ml.functions import compute_sun_group_commutator_log_density
from lattice_ml.functions._sun_group_commutator_density import (
    su3_character, su3_dim, su3_log_density_heat_kernel
)

torch.set_default_dtype(torch.float64)

TWO_PI = 2 * math.pi


def haar_sun(n, batch, seed=0):
    """Batch of Haar-random SU(n) matrices."""
    torch.manual_seed(seed)
    z = torch.randn(batch, n, n, dtype=torch.complex128) / 2**0.5
    q, r = torch.linalg.qr(z)
    phase = torch.diagonal(r, dim1=-2, dim2=-1)
    q = q * (phase / phase.abs()).unsqueeze(-2)
    return q * torch.linalg.det(q)[..., None, None] ** (-1.0 / n)


def alcove_quadrature(n_points=120):
    """Gauss-Legendre nodes `(u, v)` and weights for the SU(3) Weyl alcove
    `{u, v > 0, u + v < 2 pi}`, in root coordinates `u = theta_1 - theta_2`,
    `v = theta_2 - theta_3`.

    Split into the two triangles `u < v` and `u > v`: `J` has a kink on
    `u = v`, and straddling it would cap the accuracy at a few digits, which
    is exactly the regime where a near-miss formula still looks right.
    """
    x, w = np.polynomial.legendre.leggauss(n_points)
    x, w = (x + 1) / 2, w / 2
    s, t = np.meshgrid(x, x, indexing='ij')
    weight = np.meshgrid(w, w, indexing='ij')
    weight = weight[0] * weight[1] * s

    nodes, weights = [], []
    for B, C in [((0, TWO_PI), (math.pi, math.pi)),
                 ((TWO_PI, 0), (math.pi, math.pi))]:
        B, C = np.asarray(B), np.asarray(C)
        edge = C - B
        point = s[..., None] * B + (s * t)[..., None] * edge
        nodes.append(point.reshape(-1, 2))
        weights.append((weight * abs(B[0] * edge[1] - B[1] * edge[0])).ravel())

    uv = torch.from_numpy(np.concatenate(nodes))
    return uv[:, 0], uv[:, 1], torch.from_numpy(np.concatenate(weights))


def character_moment(p, q):
    """`<chi_{p,q}(Z)>` under `J`, by quadrature over the Weyl alcove."""
    u, v, weight = alcove_quadrature()
    theta = torch.stack(
        [(2 * u + v) / 3, (v - u) / 3, -(u + 2 * v) / 3], dim=-1
    )
    eigvals = torch.exp(1j * theta)

    J = torch.exp(
        compute_sun_group_commutator_log_density(torch.diag_embed(eigvals))
    )
    chi = su3_character(p, q, eigvals).real
    sines = torch.sin(u / 2) * torch.sin(v / 2) * torch.sin((u + v) / 2)

    # Weyl integration formula:
    #   <f> = (1 / 4 pi^2) int_alcove f J |Delta|^2 dtheta_1 dtheta_2,
    # with |Delta|^2 = 64 (prod sin)^2 and dtheta_1 dtheta_2 = du dv / 3.
    integrand = J * chi * 64 * sines**2 / 3
    return float((weight * integrand).sum()) / (4 * math.pi**2)


class GroupCommutatorDensityTest(unittest.TestCase):
    """See the module docstring for what each group of checks buys."""

    def test_su3_character_moments(self):
        """`<chi_{p,q}> = 1 / dim_{p,q}` for every irrep checked."""
        for p, q in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0),
                     (2, 1), (3, 0), (2, 2), (3, 3)]:
            with self.subTest(irrep=(p, q)):
                self.assertAlmostEqual(
                    character_moment(p, q), 1 / su3_dim(p, q), places=10
                )

    def test_class_function_symmetries(self):
        """`J(g Z g†) = J(Z)` and `J(Z†) = J(Z)`, for SU(2) and SU(3)."""
        for n in (2, 3):
            Z, g = haar_sun(n, 16, seed=1), haar_sun(n, 16, seed=2)
            log_J = compute_sun_group_commutator_log_density(Z)
            with self.subTest(n=n, symmetry='conjugation'):
                conjugated = compute_sun_group_commutator_log_density(
                    g @ Z @ g.adjoint()
                )
                self.assertTrue(torch.allclose(log_J, conjugated, atol=1e-10))
            with self.subTest(n=n, symmetry='dagger'):
                daggered = compute_sun_group_commutator_log_density(Z.adjoint())
                self.assertTrue(torch.allclose(log_J, daggered, atol=1e-10))

    def test_su2_is_the_same_pairwise_formula(self):
        """SU(2)'s closed form is `prod_{i<j} x / (2 sin(x/2))` over the
        eigen-angle gaps, with prefactor 1/N -- the SU(3) formula's shape."""
        theta = torch.linspace(0.05, math.pi - 0.05, 41)
        angles = torch.stack([theta, -theta], dim=-1)
        gap = (angles % TWO_PI)[..., 0] - (angles % TWO_PI)[..., 1]

        expected = -math.log(2) - torch.log(torch.sinc(gap / TWO_PI))
        got = compute_sun_group_commutator_log_density(
            torch.diag_embed(torch.exp(1j * angles))
        )
        self.assertTrue(torch.allclose(got, expected, atol=1e-12))

    def test_su3_heat_kernel_series_converges_to_the_closed_form(self):
        """The regularized character sum approaches the closed form as
        `t -> 0` -- the two routes to J(Z) agreeing is what makes the closed
        form more than a curve fit.

        Only the error shrinks monotonically, not the value: the heat kernel
        smooths J, so the series sits below the exact result where J peaks
        and above it where J dips.
        """
        Z = haar_sun(3, 8, seed=3)
        exact = compute_sun_group_commutator_log_density(Z)

        errors = []
        for t, p_max in [(0.04, 24), (0.02, 32), (0.01, 45)]:
            series = su3_log_density_heat_kernel(Z, t=t, p_max=p_max)
            errors.append((exact - series).abs())

        for coarse, fine in zip(errors, errors[1:]):
            self.assertTrue(torch.all(fine < coarse))
        self.assertLess(float(errors[-1].max()), 0.1)

    def test_su3_diverges_at_the_identity(self):
        """`J -> inf` as `Z -> 1`, the one genuine singularity."""
        eps = torch.tensor([1e-2, 1e-3, 1e-4])
        angles = torch.stack([eps, -eps, torch.zeros_like(eps)], dim=-1)
        log_J = compute_sun_group_commutator_log_density(
            torch.diag_embed(torch.exp(1j * angles))
        )
        self.assertTrue(torch.all(log_J[1:] > log_J[:-1]))
        self.assertGreater(log_J[-1], 10)


if __name__ == '__main__':
    unittest.main()
