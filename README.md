# lattice_ml

`lattice_ml` is a PyTorch-based library for machine-learning-assisted sampling in
**lattice field theory**, with a focus on gauge theories and Lie-group-valued fields.
It provides modular, composable tools for diffusion models, flow matching, Hamiltonian
Monte Carlo, ODE integration, and gauge-equivariant neural network layers — all
tightly integrated for lattice applications.

The library has been used in the following publications:

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
| **Gauge utilities** | `gauge_tools`, `functions` | Gauge-equivariant networks, Wilson loops, SU(N) matrix functions |
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

model = DiffusionModel(score_fn=my_network, diffuser=VPDiffuser())
model.trainer.run_training(n_epochs=500, batch_size=64, data_loader=loader)
samples = model.reverse(n_samples=100)
```

For gauge theories (`SU(N)` links), use the Lie-group-aware classes:

```python
from lattice_ml.diffusion import LieDiffusionProcess
```

### `flow_matching`
Flow matching constructs a continuous normalizing flow by regressing a
time-dependent velocity field.  The interpolation

    X_t = (1 − τ(t)) X_0 + τ(t) X_1

is linear in the endpoints, with an optional learned time reparameterization
`τ(t)`.  For SU(N) fields, `SUnFlowMatchingModel` keeps trajectories on the
group manifold throughout.

```python
from lattice_ml.flow_matching import FlowMatchingModel

model = FlowMatchingModel(dynamics_fn=my_network)
model.trainer.run_training(n_epochs=300, batch_size=128, data_loader=loader)
```

### `monte_carlo`
Classical and machine-learning-enhanced MCMC samplers:

- `HMC` / `LieHMC` — Hamiltonian Monte Carlo for scalars and Lie-group fields
- `HMD` / `LieHMD` — Hamiltonian molecular dynamics (without accept/reject)
- `U1HMC` — specialized U(1) HMC
- `metropolis_hastings` — generic Metropolis–Hastings wrapper

In [arXiv:2605.06134](https://arxiv.org/abs/2605.06134), HMD steps are used as
a **corrector** inside predictor–corrector schemes for the diffusion reverse
process, substantially improving sample quality at large inverse coupling β.

### `gauge_tools`
Gauge-equivariant neural network building blocks and utilities:

- Wilson loops and staples for U(1) and SU(N)
- Gauge-equivariant convolutional layers (`GaugeEquivariantLayer`,
  `GaugeLinkConv`, `GaugePlaqConv`)
- Link smearing utilities
- Wilson gauge action (`GaugeAction`, `GaugeActionU1`)

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

## Examples

Worked examples are in `examples/`:

| Directory | Content |
|---|---|
| `diffusion/` | Diffusion models for SU(3) gauge theories |
| `monte_carlo/` | HMC scripts for prelink and link parametrizations |


## Upstream Sources

`lattice_ml` integrates functionality from two upstream repositories:

- `lattice_ml.linalg` ← [`torch_linalg_ext`](https://github.com/jkomijani/torch_linalg_ext)
- `lattice_ml.integrate` ← [`torch_solve_ext`](https://github.com/jkomijani/torch_solve_ext)

The repository was started with a clean history intentionally.


| Created by Javad Komijani in 2025 \
| Copyright (C) 2025–2026, Javad Komijani
