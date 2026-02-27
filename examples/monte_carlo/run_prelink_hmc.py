# Created by Javad Komijani, 2026

"""Hybrid Monte Carlo (HMC) driver for SU(3) lattice gauge theory.

This script generates gauge-field configurations using Hamiltonian Monte Carlo
with the Wilson prelink action. It supports running many independent Markov
chains in parallel on CPU or GPU via PyTorch.

Main features
-------------
- SU(3) prelink gauge-field sampling using HMC
- Wilson plaquette prelink action
- Parallel independent chains
- Projection back to SU(3) after each trajectory
- Basic plaquette monitoring during thermalization

Notes
-----
- Only a single configuration per chain is currently returned.
- Multi-sample collection per chain is not yet implemented.
- Designed for research / experimentation rather than production runs.
"""


from typing import Tuple
import time
import torch

from lattice_ml.monte_carlo import SUnHMC
from lattice_ml.gauge_tools import (
    WilsonPrelinkAction,
    prelink_to_link,
    compute_mean_normalized_trace_wilson_mxn_loop
)

from lattice_ml.functions import naive_project_onto_su3

from normflow.prior import UniformSUnPrior


if torch.cuda.is_available():
    torch.set_default_device('cuda')
    torch.set_default_dtype(torch.float32)


# =============================================================================
def main(
    beta: int = 3,
    lat_shape: Tuple[int, ...] = (5, 5, 5, 5),
    num_parallel_chains: int = 1024,
    num_samples_per_chain: int = 1,
    num_thermal_traj: int = 100,
    num_leapfrog_steps: int = 15,
    save_fname: str = None
):
    """Run Hybrid Monte Carlo and generate SU(3) gauge configurations.

    Parameters
    ----------
    beta : int, default=3
        Inverse gauge coupling β appearing in the Wilson gauge action.

    lat_shape : Tuple[int, ...], default=(5,5,5,5)
        Lattice dimensions. Length determines number of spacetime dimensions.
        Example: (Nt, Nx, Ny, Nz) for a 4D lattice.

    num_parallel_chains : int, default=1024
        Number of independent Markov chains evolved in parallel. Each chain
        produces one gauge configuration.

    num_samples_per_chain : int, default=1
        Number of configurations to collect per chain after thermalization.
        Currently only `1` is supported (multi-sample not implemented).

    num_thermal_traj : int, default=100
        Number of HMC trajectories used for thermalization before returning
        configurations.

    num_leapfrog_steps : int, default=15
        Number of leapfrog integration steps per HMC trajectory.

    save_fname : str or None, default=None
        If provided, the resulting tensor of gauge fields is saved to this path
        using `torch.save`.

    Returns
    -------
    x : torch.Tensor
        Final gauge configurations after thermalization.

        Shape:
            (num_parallel_chains, *lat_shape, ndim, 3, 3)

        where `ndim = len(lat_shape)` is the number of spacetime directions and
        matrices are SU(3) link variables.

    Side Effects
    ------------
    - Prints acceptance rate and plaquette estimate during thermalization.
    - May move computation to GPU if CUDA is available.
    - Optionally writes configurations to disk.

    Notes
    -----
    - Gauge fields are projected back onto SU(3) after every trajectory using
      a naive projection method to control numerical drift.
    - The returned tensor is cloned and made contiguous for safe use with
      multi-worker PyTorch DataLoaders.

    Initialization
    --------------
    Initial gauge configurations are sampled from `UniformSUnPrior`, which
    draws SU(n) matrices uniformly with respect to the Haar measure on the
    group. This provides a "warm" start where link variables are already valid
    SU(3) elements and broadly distributed over configuration space.
    """
    action = WilsonPrelinkAction(beta=beta)

    gauge_field_shape = (*lat_shape, len(lat_shape))
    prior = UniformSUnPrior(n=3, shape=gauge_field_shape)

    hmc = SUnHMC(
        lambda t, q: action.algebra_force(q),
        t_span=(0, 1),
        num_steps=num_leapfrog_steps,
        action=action
    )

    x = prior.sample(num_parallel_chains)

    t_0 = time.time()

    for k in range(num_thermal_traj):
        x, is_accepted = hmc.step(x)
        acc_rate = torch.sum(is_accepted).item() / len(is_accepted)
        x = naive_project_onto_su3(x)
        print(f"{k}\t{acc_rate:.2f}\t", analyize(x))

    print(f"Total thermalization time: {time.time() - t_0}")

    if num_samples_per_chain > 1:
        pass  # not ready yet

    x = x.clone().contiguous()  # safe for multi-worker DataLoader

    if save_fname is not None:
        torch.save(x, save_fname)

    return x


def analyize(x):
    """Estimate the average plaquette and its statistical error."""
    x = prelink_to_link(x)
    w_1x1 = compute_mean_normalized_trace_wilson_mxn_loop(x, 1, 1)
    w_1x1_mean = w_1x1.mean()
    w_1x1_error = w_1x1.std() / x.shape[0] ** 0.5
    w_1x1_str = f"{w_1x1_mean.item():.4f}({10_000 * w_1x1_error.item():0.0f})"
    return f"{w_1x1_str}"


# =============================================================================
if __name__ == '__main__':
    from argparse import ArgumentParser
    import yaml

    parser = ArgumentParser()
    add = parser.add_argument

    # YAML config file
    add("--config", type=str, help="Path to config YAML file")

    # CLI arguments
    add("--lat_shape", type=int, nargs='+')
    add("--beta", type=float)
    add("--num_parallel_chains", type=int)
    add("--num_samples_per_chain", type=int)
    add("--num_thermal_traj", type=int)
    add("--num_leapfrog_steps", type=int)
    add("--save_fname", type=str)

    args = vars(parser.parse_args())

    # Start with YAML config if provided
    config = {}
    if args.get("config"):
        with open(args["config"], "r") as f:
            config = yaml.safe_load(f)

    # Override config with CLI args if provided
    config.update(
        {k: v for k, v in args.items() if v is not None and k != "config"}
    )

    main(**config)
