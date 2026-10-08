# lattice_ml

`lattice_ml` is a PyTorch-based library for machine-learning-assisted sampling in
**lattice field theory**, with a focus on gauge theories and Lie-group-valued fields.
It provides modular, composable tools for diffusion models, flow matching, Hamiltonian
Monte Carlo, ODE integration, gauge-equivariant neural network layers, prelink and
holonomy parametrizations of gauge fields, and coordinates and random sampling on
Lie groups — all tightly integrated for lattice applications.

The library has been used in the following publications:

- J. Komijani, *Sampling SU(N) gauge theory on a 2D lattice from independent
  plaquettes via holonomies and corner reweighting*,
  [arXiv:2610.09147](https://arxiv.org/abs/2610.09147) (2026) — uses version 0.2.0;
  the notebooks are in
  [holonomy-2d-gauge-notebooks](https://github.com/jkomijani/holonomy-2d-gauge-notebooks)
- J. Komijani, M. K. Marinkovic, L. Turgut, *Diffusion models for SU(N) gauge
  theories*, [arXiv:2605.06134](https://arxiv.org/abs/2605.06134) (2026)
- J. Komijani, *Noise scheduling and linear dynamics in diffusion models on Lie
  groups*, [arXiv:2605.17326](https://arxiv.org/abs/2605.17326) (2026)
- J. Komijani, M. K. Marinkovic, *Normalizing flows for SU(N) gauge theories
  employing singular value decomposition*,
  [arXiv:2501.18288](https://arxiv.org/abs/2501.18288) (2025)


## Overview

`lattice_ml` is a general-purpose toolkit for lattice field theory computations —
not just machine learning.  You can use it for something as straightforward as
running HMC for an SU(3) gauge theory, or as involved as training a diffusion
model to generate decorrelated gauge configurations at scale.

The library is organized around several self-contained layers:

| Layer | Modules | Typical use |
|---|---|---|
| **Generative ML** | `diffusion`, `flow_matching` | Score-based & flow-matching samplers for scalars and SU(N) |
| **Monte Carlo** | `monte_carlo` | HMC / HMD for U(1) and SU(N) gauge theories |
| **Gauge utilities** | `gauge_tools`, `functions` | Gauge-equivariant networks, Wilson loops, prelinks and holonomies, SU(N) matrix functions |
| **Lie groups** | `lie_groups`, `random` | Euler-angle coordinates, group commutators, Haar and other random matrices |
| **ODE/SDE solvers** | `integrate` | Adjoint-based ODE integration, Lie-group and symplectic solvers |
| **Linear algebra** | `linalg` | AD-safe eigensystem and SVD routines for SU(N) |

The `integrate` and `linalg` modules are self-contained and can be imported
independently — for instance in normalizing flow models that need adjoint ODE
integration or differentiable SVD on Lie-group–valued fields (see
[arXiv:2501.18288](https://arxiv.org/abs/2501.18288)).


## Modules

### `diffusion`
Score-based generative models following the SDE framework of Song et al.
The library implements **VP** (variance-preserving) and **SubVP** schedules,
extended to Lie groups.  A key result from [arXiv:2605.17326](https://arxiv.org/abs/2605.17326)
is that a specific noise schedule produces a **linear decay of the Wilson
action** with diffusion time — an emergent property of the Lie-group framework
that has no direct Euclidean analogue.

```python
from lattice_ml.diffusion import DiffusionModel, VPDiffuser

model = DiffusionModel(diffuser=VPDiffuser(), network_fn=my_network)
model.trainer.run_training(training_dataloader=loader, n_epochs=500)
samples = model.reverse(x_0)  # x_0: samples from the prior (noise)
```

For gauge theories (`SU(N)` links), use the Lie-group-aware classes:

```python
from lattice_ml.diffusion import SUnDiffuser, SUnDiffusionProcess
from lattice_ml.diffusion.gauge import SUnDiffusionModel
```

`lattice_ml.diffusion.gauge` also provides the predictor–corrector machinery
(`SUnHMDBasedCorrector`, `SUnLangevinBasedCorrector`) and SDE integrators on the
group.

### `flow_matching`
Flow matching constructs a continuous normalizing flow by regressing a
time-dependent velocity field.  The interpolation

    X_t = (1 − τ(t)) X_0 + τ(t) X_1

is linear in the endpoints, with an optional learned time reparameterization
`τ(t)`.  For SU(N) fields, `SUnFlowMatchingModel` keeps trajectories on the
group manifold throughout.

```python
from lattice_ml.flow_matching.dynamics import FlowMatchingModel

model = FlowMatchingModel(dynamics_fn=my_network)
model.trainer.run_training(training_dataloader=loader, n_epochs=300)
```

The subpackage `lattice_ml.flow_matching.flowmap` provides flow maps
(`FlowMap`, `LieFlowMap`) and their learner (`FlowMapLearner`).

### `monte_carlo`
Classical and machine-learning-enhanced MCMC samplers:

- `HMC` / `SUnHMC` — Hamiltonian Monte Carlo for scalars and SU(N) fields
- `HMD` / `SUnHMD` — Hamiltonian molecular dynamics (without accept/reject);
  `ResampledHMD` with momentum resampling
- `U1HMC` — specialized U(1) HMC

In [arXiv:2605.06134](https://arxiv.org/abs/2605.06134), HMD steps are used as
a **corrector** inside predictor–corrector schemes for the diffusion reverse
process, substantially improving sample quality at large inverse coupling β.

### `gauge_tools`
Gauge-equivariant neural network building blocks and utilities:

- Wilson gauge action (`WilsonGaugeAction`, `WilsonU1GaugeAction`)
- Wilson loops and staples for U(1) and SU(N)
- Gauge-equivariant convolutional layers (`GaugeLinkConv`,
  `TimeConditionedGaugeLinkConv`, and the building blocks in
  `gauge_equivariant_layers`)
- Link smearing (`GaugeLinkSmear`)
- **Prelinks**: conversion between links and prelinks (`link_to_prelink`,
  `prelink_to_link`), the Wilson action in prelink variables
  (`WilsonPrelinkAction`), and sealed prelinks and staples
- **Holonomies**: conversion between links, prelinks and holonomies
  (`link_to_holonomy`, `holonomy_to_link`, `prelink_to_holonomy`,
  `holonomy_to_prelink`), the Wilson action in holonomy variables
  (`WilsonHolonomyAction`), and holonomy clovers; U(1) variants are included

The prelink and holonomy parametrizations are the basis of the plaquette-based
sampler of [arXiv:2610.09147](https://arxiv.org/abs/2610.09147).

### `lie_groups`
Coordinates on compact Lie groups, as invertible maps between a group element
and real coordinates, each carrying the log-Jacobian of the change of
variables:

- Euler angles for SU(2), SU(3) and SU(N) (`su2_to_euler_angles`,
  `euler_angles_to_su2`, and the SU(3) and SU(N) analogs)
- The group commutator `Z = X Y X† Y†`: an encoder and decoder
  (`encode_sun_group_commutator`, `decode_sun_group_commutator`), a solver
  (`solve_sun_group_commutator`), and the density of `Z` for Haar-distributed
  `X, Y` (`compute_sun_group_commutator_log_prob`, closed form for SU(2) and
  SU(3)), as used in [arXiv:2610.09147](https://arxiv.org/abs/2610.09147)

### `random`
Generation of random matrices from named distributions:

- Haar measure on SU(N), U(N) and U(1), and the uniform distribution on
  diagonal SU(N) (`SUnGroup`, `UnGroup`, `U1Group`, `DiagonalSUnGroup`,
  `rand_sun_group_like`, `rand_diagonal_sun_group_like`)
- Complex Ginibre matrices (`GinibreCMatrixDist`)
- Conditional sampling of `(X, Y)` given their group commutator `Z`
  (`sample_sun_group_commutator_xy_given_z`): in closed form for SU(2), and
  with a trained normalizing flow for SU(3)

### `functions`
Matrix functions with exact Jacobians for SU(N) manifold operations:
matrix exponential/logarithm, projection to SU(N), spectral decompositions.
These are essential for the Lie-group diffusion and flow-matching layers.

### `linalg`
Differentiable linear-algebra routines for SU(N)-valued tensors, with custom
autograd rules that are reliable under the symmetry constraints of Lie groups.
Continues and extends [`torch_linalg_ext`](https://github.com/jkomijani/torch_linalg_ext)
(archived; development moved here).

Exported functions: `eigh`, `eigu`, `svd`, `svd_with_simplified_ad`,
`inverse_eigh`, `inverse_eign`, `reciprocal`, `project_grad_sun`,
`project_data_and_grad_sun`.

These are used heavily in the SVD-based normalizing flow construction of
[arXiv:2501.18288](https://arxiv.org/abs/2501.18288), where gauge-invariant
building blocks are built from singular values of SU(N) link products, and
correct Jacobians must propagate through the SVD.

### `integrate`
A full ODE/SDE integration library that can be used **standalone** — e.g.,
inside a normalizing flow that needs continuous-time dynamics or adjoint-based
gradient computation.  Continues and extends
[`torch_solve_ext`](https://github.com/jkomijani/torch_solve_ext)
(archived; development moved here).

Key capabilities:
- **Standard ODE**: `odeint`, `ODEFlow`, `ODEFlow_` (with log-Jacobian)
- **Adjoint backprop**: `AdjODEFlow_`, `AdjLieODEFlow_` — memory-efficient
  gradients through long ODE trajectories
- **Lie-group ODE**: `lie_odeint`, `LieODEFlow`, `LieODEFlow_` — integrators
  that stay on the group manifold
- **Symplectic integrators**: `symplectic_odeint`, `lie_symplectic_odeint`,
  `SymplecticODEFlow`, `SymplecticODEFlow_` — for Hamiltonian dynamics (HMC)

The trailing-underscore convention marks any module that returns an
`(output, log_jacobian)` tuple, consistent with the `normflow` convention.

### `stats`
Statistical utilities for lattice observables, including modal analysis.


## Installation

```bash
git clone https://github.com/jkomijani/lattice_ml.git
cd lattice_ml
pip install -e .
```

To install the exact version used in a given paper, use its tag, for example
version 0.2.0 for [arXiv:2610.09147](https://arxiv.org/abs/2610.09147):

```bash
pip install git+https://github.com/jkomijani/lattice_ml.git@v0.2.0
```

## Examples

Worked examples are in `examples/`:

| Directory | Content |
|---|---|
| `diffusion/` | Diffusion models for SU(3) gauge theories |
| `monte_carlo/` | HMC scripts for prelink and link parametrizations |

The notebooks of [arXiv:2610.09147](https://arxiv.org/abs/2610.09147) are in a
separate repository,
[holonomy-2d-gauge-notebooks](https://github.com/jkomijani/holonomy-2d-gauge-notebooks).


## Upstream Sources

`lattice_ml` integrates functionality from two upstream repositories:

- `lattice_ml.linalg` ← [`torch_linalg_ext`](https://github.com/jkomijani/torch_linalg_ext)
- `lattice_ml.integrate` ← [`torch_solve_ext`](https://github.com/jkomijani/torch_solve_ext)

The repository was started with a clean history intentionally.


| Created by Javad Komijani in 2025 \
| Copyright (C) 2025–2026, Javad Komijani
