#!/usr/bin/env python3
# Copyright (c) 2026 Javad Komijani

# pylint does not resolve torch's dynamically populated `linalg`
# submodule, so every call into it is reported as not-callable.
# pylint: disable=not-callable

# `haar_sun` is repeated in each test module on purpose, so that every module
# runs standalone; do not factor it out.
# pylint: disable=duplicate-code

# pylint: disable=invalid-name

"""Tests for the "from the twisted matrix" form of the spectral-twist Jacobian.

`log_jacobian_su{2,3}_spectral_twist` take U's eigenframe, which in the
group-commutator setting is `X_tilde = Q^dag X Q` -- free when encoding, but
only available after X has been rebuilt when decoding.
`log_jacobian_su{2,3}_spectral_twist_from_diag` return the same number from
`diag(I - Z_tilde)` alone -- the same Jacobian in the (lambda, diag) chart.

The checks that matter, in order of how badly a regression would hurt:

1. the W form and the `_from_diag` form agree **for an arbitrary Q**, not
   merely for the Q that `encode` produced -- same value and same sign, so
   the two are interchangeable. Scoring a candidate frame before decoding it
   is the whole reason the second form exists, so a version that agreed only
   on the encode image would pass a careless test and be wrong in use;
2. the normalization, against the group-commutator density, which reaches the
   same number through the character expansion -- an independent route, and
   the only check here that would catch a wrong overall constant;
3. the branch rule survives the `solve_lambda_su*` / `..._from_diag` split,
   for both N;
4. the triangle geometry itself, including the degenerate case.

`_spectral_twist.py` used to carry a `run_self_tests()` of its own; those
checks live here now, as `SpectralTwistIdentitiesTest` and
`SpectralTwistFiniteDifferenceTest`, so the module holds no test code.

Run with:  python3 tests/test_spectral_twist.py -v
      or:  python3 -m unittest tests.test_spectral_twist -v
"""

import math
import unittest

import torch

from lattice_ml.lie_groups import (
    encode_sun_group_commutator,
    decode_sun_group_commutator,
    compute_sun_group_commutator_log_prob,
)
from lattice_ml.lie_groups._spectral_twist import (
    bistochastic_from_W, eigendecompose_sorted, twist_eigenvalues,
    jacobian_sun_spectral_twist, jacobian_su2_spectral_twist,
    jacobian_su3_spectral_twist, jacobian_spectral_twist_from_U,
    _helmert,
    log_jacobian_su2_spectral_twist, log_jacobian_su3_spectral_twist,
    log_jacobian_su2_spectral_twist_from_diag,
    log_jacobian_su3_spectral_twist_from_diag,
    solve_lambda_su2_spectral_twist_from_diag,
    solve_lambda_su3_spectral_twist_from_diag,
    triangle_area_from_sides, inscribed_triangle_area,
)
from lattice_ml.lie_groups._sun_group_commutator import (
    eig_with_options, solve_lambda_su2, solve_lambda_su3,
)
from lattice_ml.random import rand_sun_group_like, rand_diagonal_sun_group_like

torch.set_default_dtype(torch.float64)

EIG_OPTIONS = {'sort_by': 'zero-sum-angle', 'project_to_sun': True}


def haar_sun(n, batch, seed=0):
    """Batch of Haar-random SU(n) matrices."""
    torch.manual_seed(seed)
    return rand_sun_group_like(
        torch.zeros(batch, n, n, dtype=torch.complex128)
    )


def diag_of(Z, Q, n):
    """diag(I - Z_tilde), the argument the `_from_diag` forms take."""
    eye = torch.eye(n, dtype=Z.dtype)
    return torch.diagonal(Q.adjoint() @ (eye - Z) @ Q, dim1=-2, dim2=-1)


def encoded(n, batch=4000, seed=0):
    """A Haar pair and everything `encode` returns for it."""
    X = haar_sun(n, batch, seed=seed)
    Y = haar_sun(n, batch, seed=seed + 1)
    (D, Q, Z), logj = encode_sun_group_commutator(X, Y, return_logj=True)
    return X, Y, D, Q, Z, logj


class SpectralTwistFromFramesTest(unittest.TestCase):
    """The (Z, Q) form of the spectral-twist log-Jacobian."""

    def test_agrees_with_the_eigenframe_form_on_the_encode_image(self):
        for n in (2, 3):
            with self.subTest(n=n):
                X, _, _, Q, Z, logj = encoded(n, seed=10 * n)
                from_W = (log_jacobian_su2_spectral_twist if n == 2
                          else log_jacobian_su3_spectral_twist)
                from_diag = (log_jacobian_su2_spectral_twist_from_diag
                             if n == 2
                             else log_jacobian_su3_spectral_twist_from_diag)
                mine = from_diag(diag_of(Z, Q, n))
                self.assertLess(float((mine - logj).abs().max()), 1e-9)
                self.assertLess(
                    float((mine - from_W(Q.adjoint() @ X @ Q)).abs().max()),
                    1e-9
                )

    def test_agrees_for_frames_outside_the_encode_image(self):
        """The identity must hold for an ARBITRARY Q. This is the test that
        distinguishes a correct implementation from one that happens to agree
        wherever `encode` has already been."""
        for n in (2, 3):
            with self.subTest(n=n):
                _, _, _, _, Z, _ = encoded(n, seed=20 + n)
                from_W = (log_jacobian_su2_spectral_twist if n == 2
                          else log_jacobian_su3_spectral_twist)
                from_diag = (log_jacobian_su2_spectral_twist_from_diag
                             if n == 2
                             else log_jacobian_su3_spectral_twist_from_diag)
                Q = rand_sun_group_like(Z)
                D = rand_diagonal_sun_group_like(Z)
                X_hat, Y_hat = decode_sun_group_commutator(D, Q, Z)
                # the decoded pair must really solve the constraint first,
                # otherwise the comparison below is meaningless
                residual = (
                    X_hat @ Y_hat @ X_hat.adjoint() @ Y_hat.adjoint() - Z
                )
                self.assertLess(float(residual.abs().max()), 1e-9)
                self.assertLess(
                    float((from_diag(diag_of(Z, Q, n))
                           - from_W(Q.adjoint() @ X_hat @ Q)).abs().max()),
                    1e-9
                )

    def test_does_not_depend_on_D(self):
        """D is diagonal, so it drops out of |X_tilde_ij|, not moving J."""
        for n in (2, 3):
            with self.subTest(n=n):
                _, _, _, Q, Z, logj = encoded(n, seed=30 + n)
                X2, Y2 = decode_sun_group_commutator(
                    rand_diagonal_sun_group_like(Z), Q, Z
                )
                _, logj2 = encode_sun_group_commutator(
                    X2, Y2, return_logj=True
                )
                self.assertLess(float((logj2 - logj).abs().max()), 1e-9)

    def test_su3_jacobian_is_bounded_by_three_and_attains_it_at_Q_z(self):
        _, _, _, _, Z, logj = encoded(3, seed=41)
        self.assertLessEqual(float(torch.exp(logj).max()), 3 + 1e-9)

        Q_z = eig_with_options(Z, **EIG_OPTIONS)[1]
        err = (torch.exp(log_jacobian_su3_spectral_twist_from_diag(
            diag_of(Z, Q_z, 3)
        )) - 3.0).abs()

        # Typical accuracy is ~1e-14, but a small fraction of samples land near
        # a degenerate v-triangle, where the SU(3) closure solve takes an
        # arccos of an argument near +-1 and loses about half the mantissa. So
        # assert on a robust quantile and keep only a loose cap on the tail --
        # a tight max() here would be a flaky test, not a stronger one.
        # measured on this batch: q50 3e-14, q90 1e-12, q99 1e-10, max 2e-7.
        self.assertLess(float(err.median()), 1e-12)
        self.assertLess(float(err.quantile(0.99)), 1e-9)
        self.assertLess(float(err.max()), 1e-5)

    def test_su2_form_is_the_harmonic_mean_of_the_diagonal(self):
        _, _, _, Q, Z, logj = encoded(2, seed=51)
        eye = torch.eye(2, dtype=Z.dtype)
        d_1, d_2 = torch.diagonal(
            Q.adjoint() @ (eye - Z) @ Q, dim1=-2, dim2=-1
        ).unbind(-1)
        # d_2 = conj(d_1), and d_1 + d_2 = det(I - Z)
        self.assertLess(float((d_2 - d_1.conj()).abs().max()), 1e-12)
        self.assertLess(
            float((d_1 + d_2 - torch.linalg.det(eye - Z)).abs().max()), 1e-12
        )
        harmonic = (2 * d_1 * d_2 / (d_1 + d_2)).real
        self.assertLess(float((torch.log(harmonic) - logj).abs().max()), 1e-9)


class NormalizationTest(unittest.TestCase):
    """E_Haar[1/J | Z] must be the group-commutator density.

    That density is derived from the character expansion, so this ties two
    modules together that nothing else connects, and it is the only check here
    sensitive to a wrong overall constant.
    """

    def test_haar_average_of_one_over_J_is_the_commutator_density(self):
        for a, b in [(2.4, -2.0), (1.5, -0.4), (2.0, 1.0)]:
            with self.subTest(phi=(a, b)):
                torch.manual_seed(99)
                ang = torch.tensor([a, b, -a - b])
                Z = torch.diag_embed(
                    torch.exp(1j * ang.to(torch.complex128))
                )
                Z = Z.expand(400_000, 3, 3).contiguous()
                diag = diag_of(Z, rand_sun_group_like(Z), 3)
                inv = torch.exp(
                    -log_jacobian_su3_spectral_twist_from_diag(diag)
                )
                exact = float(
                    torch.exp(compute_sun_group_commutator_log_prob(Z[:1]))
                )
                sem = float(inv.std() / len(inv) ** 0.5)
                self.assertLess(abs(float(inv.mean()) - exact), 5 * sem)


class SolveLambdaSplitTest(unittest.TestCase):
    """`solve_lambda_su*` were split into a `(Z, Q)` wrapper here and a
    `_from_diag` core in `_spectral_twist`; the split must be
    behaviour-preserving for both N."""

    def test_from_diag_agrees_with_the_Z_Q_entry_point(self):
        for n, core, wrapper in ((2, solve_lambda_su2_spectral_twist_from_diag,
                                  solve_lambda_su2),
                                 (3, solve_lambda_su3_spectral_twist_from_diag,
                                  solve_lambda_su3)):
            with self.subTest(n=n):
                _, _, _, Q, Z, _ = encoded(n, seed=61 + n)
                diag = diag_of(Z, Q, n)
                self.assertLess(
                    float((core(diag) - wrapper(Z, Q)).abs().max()), 1e-12
                )

    def test_su2_branches_are_the_centre(self):
        """The two SU(2) closure roots differ by the centre, Lambda -> -Lambda
        -- the whole spectrum flips together. This is what makes the N = 2
        Jacobian branch-independent, unlike N = 3."""
        _, _, _, Q, Z, _ = encoded(2, seed=71)
        diag = diag_of(Z, Q, 2)
        lam_a = solve_lambda_su2_spectral_twist_from_diag(
            diag, descending_angle=False
        )
        lam_b = solve_lambda_su2_spectral_twist_from_diag(
            diag, descending_angle=True
        )
        self.assertLess(float((lam_b + lam_a).abs().max()), 1e-14)

    def test_descending_angle_is_forwarded(self):
        _, _, _, Q, Z, _ = encoded(3, seed=62)
        diag = diag_of(Z, Q, 3)
        for flag in (False, True):
            with self.subTest(descending_angle=flag):
                lhs = solve_lambda_su3_spectral_twist_from_diag(
                    diag, descending_angle=flag
                )
                rhs = solve_lambda_su3(Z, Q, descending_angle=flag)
                self.assertLess(float((lhs - rhs).abs().max()), 1e-12)


class TriangleGeometryTest(unittest.TestCase):
    """The two area helpers in `_spectral_twist`."""

    def test_heron_against_known_triangles(self):
        a = torch.tensor([3.0, 5.0, 1.0])
        b = torch.tensor([4.0, 5.0, 1.0])
        c = torch.tensor([5.0, 6.0, 2.0])   # 3-4-5; isoceles; degenerate
        expected = torch.tensor([6.0, 12.0, 0.0])
        self.assertLess(
            float((triangle_area_from_sides(a, b, c) - expected).abs().max()),
            1e-12
        )

    def test_heron_is_zero_not_nan_outside_the_triangle_inequality(self):
        out = triangle_area_from_sides(
            torch.tensor(1.0), torch.tensor(1.0), torch.tensor(5.0)
        )
        self.assertEqual(float(out), 0.0)

    def test_inscribed_area_is_the_vandermonde_over_four(self):
        torch.manual_seed(7)
        ang = (torch.rand(500, 3) * 2 * math.pi).to(torch.complex128)
        lam = torch.exp(1j * ang)
        l1, l2, l3 = lam.unbind(-1)
        vandermonde = ((l1 - l2) * (l1 - l3) * (l2 - l3)).abs()
        area = inscribed_triangle_area(lam)
        self.assertLess(float((area - vandermonde / 4).abs().max()), 1e-12)

    def test_inscribed_area_is_bounded_by_the_equilateral_value(self):
        torch.manual_seed(8)
        lam = torch.exp(
            1j * (torch.rand(20000, 3) * 2 * math.pi).to(torch.complex128)
        )
        self.assertLessEqual(
            float(inscribed_triangle_area(lam).max()),
            3 * math.sqrt(3) / 4 + 1e-12
        )


def sun_basis(n, cdt=torch.complex128, rdt=torch.float64):
    """Orthonormal basis of su(n): off-diagonal pairs plus Helmert diagonal."""
    basis = []
    for i in range(n):
        for j in range(i + 1, n):
            E = torch.zeros(n, n, dtype=cdt)
            E[i, j], E[j, i] = 1, -1
            basis.append(E / 2 ** 0.5)

            E = torch.zeros(n, n, dtype=cdt)
            E[i, j], E[j, i] = 1j, 1j
            basis.append(E / 2 ** 0.5)
    V = _helmert(n, rdt, None)
    for c in range(n - 1):
        basis.append(1j * torch.diag(V[:, c]).to(cdt))
    return torch.stack(basis)


def fd_logjac(U, eps=1e-6):
    """Finite-difference log |det d(Z_tilde^dag dZ_tilde)/d(U^dag dU)|.

    Differentiates the MAP, so it depends on no closed form. This is the only
    check in the suite that would survive every formula here being wrong.
    """
    n = U.shape[-1]
    basis = sun_basis(n)
    Z_tilde0 = twist_eigenvalues(U.unsqueeze(0))[0]
    rows = []
    for a in range(basis.shape[0]):
        Up = U @ torch.matrix_exp(eps * basis[a])
        Um = U @ torch.matrix_exp(-eps * basis[a])
        dZ_tilde = (twist_eigenvalues(Up.unsqueeze(0))[0]
                    - twist_eigenvalues(Um.unsqueeze(0))[0]) / (2 * eps)
        X = Z_tilde0.conj().T @ dZ_tilde
        row = [(basis[b].conj().T @ X).diagonal().sum().real
               for b in range(basis.shape[0])]
        rows.append(torch.stack(row))
    return torch.linalg.slogdet(torch.stack(rows))[1]


class SpectralTwistIdentitiesTest(unittest.TestCase):
    """Ported from `_spectral_twist.run_self_tests`. Unlike the rest of this
    module these run at N = 2..5, since `jacobian_sun_spectral_twist` is
    general-N even though the closed forms are not."""

    def test_eigendecomposition_reconstructs_U(self):
        for n in (2, 3, 4, 5):
            with self.subTest(n=n):
                U = haar_sun(n, 6, seed=n)
                lam, W = eigendecompose_sorted(U)
                rec = W @ torch.diag_embed(lam) @ W.adjoint()
                self.assertLess(float((rec - U).abs().max()), 1e-10)

    def test_twisted_matrix_is_in_sun(self):
        for n in (2, 3, 4, 5):
            with self.subTest(n=n):
                Z_tilde = twist_eigenvalues(haar_sun(n, 6, seed=n))
                eye = torch.eye(n, dtype=torch.complex128)
                unitary = Z_tilde @ Z_tilde.adjoint() - eye
                self.assertLess(float(unitary.abs().max()), 1e-10)
                self.assertLess(
                    float((torch.linalg.det(Z_tilde) - 1).abs().max()), 1e-10
                )

    def test_helmert_form_equals_the_minor_form(self):
        """det(I - B) on the zero-sum plane == N * one principal minor. The
        minors are all equal because B is doubly stochastic."""
        for n in (2, 3, 4, 5):
            with self.subTest(n=n):
                _, W = eigendecompose_sorted(haar_sun(n, 6, seed=n))
                Jg = jacobian_sun_spectral_twist(W)
                eye = torch.eye(n, dtype=torch.float64)
                minor = (eye - bistochastic_from_W(W))[..., 1:, 1:]
                Jm = n * torch.linalg.det(minor)
                self.assertLess(float((Jg - Jm).abs().max()), 1e-10)

    def test_from_U_agrees_and_jacobian_is_positive(self):
        for n in (2, 3, 4, 5):
            with self.subTest(n=n):
                U = haar_sun(n, 6, seed=n)
                _, W = eigendecompose_sorted(U)
                Jg = jacobian_sun_spectral_twist(W)
                self.assertLess(
                    float((Jg - jacobian_spectral_twist_from_U(U))
                          .abs().max()),
                    1e-10
                )
                self.assertTrue(bool((Jg > 0).all()))

    def test_closed_forms_agree_with_the_general_N_form(self):
        for n, closed in ((2, jacobian_su2_spectral_twist),
                          (3, jacobian_su3_spectral_twist)):
            with self.subTest(n=n):
                _, W = eigendecompose_sorted(haar_sun(n, 6, seed=n))
                self.assertLess(
                    float((jacobian_sun_spectral_twist(W)
                           - closed(W)).abs().max()),
                    1e-10
                )

    def test_jacobian_does_not_depend_on_lambda(self):
        """Within a fixed ordering cell -- eigenangles kept sorted ascending,
        so the sort rule returns the same W columns."""
        n = 3
        torch.manual_seed(0)
        W = haar_sun(n, 4, seed=5)
        Jref = jacobian_sun_spectral_twist(W)
        for k in range(3):
            with self.subTest(draw=k):
                th, _ = torch.sort(torch.rand(4, n) * 2 - 1, dim=-1)
                th = th - th.mean(-1, keepdim=True)
                th, _ = torch.sort(th, dim=-1)
                Lam = torch.diag_embed(torch.exp(1j * th.to(torch.complex128)))
                U = W @ Lam @ W.adjoint()
                self.assertLess(
                    float((jacobian_spectral_twist_from_U(U)
                           - Jref).abs().max()),
                    1e-10
                )

    def test_eigenvalue_ordering_changes_the_jacobian(self):
        """The ordering rule is not cosmetic: flipping SU(2)'s two eigenvalues
        gives a different Z_tilde, and the two Jacobians sum to 2."""
        _, W = eigendecompose_sorted(haar_sun(2, 1, seed=3))
        J = float(jacobian_sun_spectral_twist(W))
        J_flip = float(jacobian_sun_spectral_twist(W.flip(-1)))
        self.assertLess(abs(J + J_flip - 2), 1e-10)
        self.assertGreater(abs(J - J_flip), 1e-6)

    def test_expectation_of_the_jacobian_is_one(self):
        """E[J] = 1 because the twist is a.e. one-to-one, so the pushforward
        of Haar is a probability density."""
        for n in (2, 3, 4):
            with self.subTest(n=n):
                W = haar_sun(n, 200_000, seed=100 + n)
                J = jacobian_sun_spectral_twist(W)
                sem = float(J.std() / 200_000 ** 0.5)
                self.assertLess(abs(float(J.mean()) - 1), max(5 * sem, 1e-3))


class SpectralTwistFiniteDifferenceTest(unittest.TestCase):
    """The map differentiated numerically, against the closed forms."""

    def test_general_N_form_matches_finite_differences(self):
        for n in (2, 3, 4, 5):
            with self.subTest(n=n):
                U = haar_sun(n, 3, seed=n)
                fd = torch.stack([fd_logjac(U[b]) for b in range(3)])
                _, W = eigendecompose_sorted(U)
                Jg = jacobian_sun_spectral_twist(W)
                self.assertLess(float((fd - torch.log(Jg)).abs().max()), 1e-6)



if __name__ == '__main__':
    unittest.main()
