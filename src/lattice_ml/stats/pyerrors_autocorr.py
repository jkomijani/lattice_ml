# Created by Javad Komijani, 2025

"""
pyerrors_autocorr.py

Utilities for plotting observables analyzed with the `pyerrors` package.

It includes:

- `plot_rho`: Plot the normalized autocorrelation function (rho).
- `plot_tauint`: Plot the integrated autocorrelation time.
- `compute_and_plot_autocorr_and_tauint`: Compute the integrated
   autocorrelation time for a single observable and generate side-by-side plots
   of rho and tau_int.
"""


import pyerrors
import numpy as np
from matplotlib import pyplot as plt


__all__ = [
    "compute_and_plot_autocorr_and_tauint",
    "plot_tauint",
    "plot_rho"
]


# =============================================================================
def compute_and_plot_autocorr_and_tauint(obs, tag='', axs=None):
    """
    Compute integrated autocorrelation time using pyerrors and plot both
    the normalized autocorrelation function (rho) and integrated
    autocorrelation time.

    Parameters
    ----------
    obs : array-like
        Observable data to analyze.
    tag : str, optional
        Name assigned to this observable in pyerrors. Default is '|M|'.
    axs : matplotlib.axes.Axes, optional
        Axis to plot on. If None, a new figure is created.

    Returns
    -------
    tau_int : float
        Integrated autocorrelation time for the observable.
    dtau_int : float
        Error of the integrated autocorrelation time.
    """
    if axs is None:
        _, axs = plt.subplots(1, 2, figsize=(10, 4))

    # Wrap the observable in a pyerrors.Obs object with the given tag
    pye_obs = pyerrors.Obs([obs], names=[tag])
    pye_obs.gamma_method()  # Run the Gamma method for autocorrelation analysis

    # Plot normalized autocorrelation function (rho) on the first axis
    plot_rho(pye_obs, tag=tag, ax=axs[0])

    # Plot integrated autocorrelation time on the second axis
    plot_tauint(pye_obs, tag=tag, ax=axs[1])

    # Return the tau_int and its error for this observable
    return pye_obs.e_tauint[tag], pye_obs.e_dtauint[tag]


def plot_tauint(pye_obs, tag='', ax=None):
    # This is similar to pye_obs.plot_tauint()
    """
    Plot integrated autocorrelation time for each ensemble.

    Parameters
    ----------
    pye_obs : object
        A `pyerrors.Obs`-like object with attributes:
        - mc_names
        - e_n_tauint, e_n_dtauint
        - e_tauint, e_dtauint
        - e_rho, e_windowsize
        - e_dvalue (must exist)
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, a new figure and axis are created.
    Notes
    -----
    Adapted from the `pyerrors` project:
    https://fjosw.github.io/pyerrors/pyerrors/obs.html#Obs.plot_tauint
    """

    # Create axis if none provided
    if ax is None:
        _, ax = plt.subplots()

    if not hasattr(pye_obs, 'e_dvalue'):
        raise Exception("Run the Gamma method first.")

    # Loop over ensembles & plot their tauint
    # e_name: ensemble_name, which is tag in pyerrors.Obs([obs], names=[tag])
    for e_name in pye_obs.mc_names:

        ax.set_xlabel(r"$W$")  # Lag (window size)
        ax.set_ylabel(r"$\tau_\mathrm{int}$")

        # Determine number of available autocorrelation values
        length = int(len(pye_obs.e_n_tauint[e_name]))
        w = pye_obs.e_windowsize[e_name]
        xmax = max(10.5, 2 * w - 0.5)

        # Plot tauint against different window size
        tau, dtau = pye_obs.e_tauint[e_name], pye_obs.e_dtauint[e_name]
        tau_str = format_val_err(tau, dtau)
        label = f"{tag} " + r"$\tau_{\rm int} = $" + tau_str

        ax.errorbar(
            np.arange(length)[: int(xmax) + 1],
            pye_obs.e_n_tauint[e_name][: int(xmax) + 1],
            yerr=pye_obs.e_n_dtauint[e_name][: int(xmax) + 1],
            linewidth=1, capsize=2, label=label,
        )

        ax.axvline(x=w, color="k", alpha=0.5, ls=":", marker=",")

        ax.set_xlim(-0.5, xmax)
        ylim = ax.get_ylim()
        ax.set_ylim(bottom=0.0, top=max(1.0, ylim[1]))
        ax.legend()


def plot_rho(pye_obs, tag='', ax=None):
    # This is similar to pye_obs.plot_rho()
    """
    Plot the normalized autocorrelation function (rho) for each ensemble.

    This function visualizes the normalized autocorrelation function of an
    observable estimated using the Gamma method from `pyerrors`.

    Parameters
    ----------
    pye_obs : object
        A `pyerrors.Obs`-like object that has undergone the Gamma method
        analysis. Must contain attributes:
        - `mc_names`: list of ensemble names.
        - `e_rho`: dict of autocorrelation values for each ensemble.
        - `e_drho`: dict of autocorrelation uncertainties.
        - `e_windowsize`: dict of window sizes.
        - `e_dvalue`: exists if Gamma method has been applied.

    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, a new figure and axis are created.

    Notes
    -----
    Adapted from the `pyerrors` project:
    https://fjosw.github.io/pyerrors/pyerrors/obs.html#Obs.plot_rho
    """

    # Create axis if none provided
    if ax is None:
        _, ax = plt.subplots()

    # Ensure Gamma method results are available
    if not hasattr(pye_obs, 'e_dvalue'):
        raise Exception("Run the Gamma method first before plotting rho.")

    # Loop over ensembles & plot their autocorrelation
    # e_name: ensemble_name, which is tag in pyerrors.Obs([obs], names=[tag])
    for e_name in pye_obs.mc_names:

        ax.set_xlabel(r"$t$")  # auto-correlation time
        ax.set_ylabel(r"$\rho$")  # Normalized autocorrelation

        # Determine number of available autocorrelation values
        length = int(len(pye_obs.e_drho[e_name]))
        w = pye_obs.e_windowsize[e_name]

        # Plot rho with error bars and label with tau_int estimate
        # Note: pye_obs.e_drho is mostly zero except at one point
        tau, dtau = pye_obs.e_tauint[e_name], pye_obs.e_dtauint[e_name]
        tau_str = format_val_err(tau, dtau)
        label = f"{tag} " + r"$\tau_{\rm int} = $" + tau_str

        ax.errorbar(
            np.arange(length), pye_obs.e_rho[e_name][:length],
            yerr=pye_obs.e_drho[e_name][:], linewidth=1, capsize=2, label=label
        )

        # Draw a vertical line at the Gamma method window size
        ax.axvline(x=w, color="k", alpha=0.5, ls=":", marker=",")

        # Draw a horizontal line at rho = 0
        xmax = max(10.5, 2 * w - 0.5)
        ax.plot([-0.5, xmax], [0, 0], "--k", lw=1)

        ax.set_xlim(-0.5, xmax)
        ax.legend()


def format_val_err(value, error, err_digits=1):
    """
    Format a numerical value with its uncertainty in concise scientific style.

    Parameters
    ----------
    value : float
        The central value of the measurement or computed quantity.
    error : float
        The associated standard uncertainty or error estimate.
    err_digits : int, optional
        Number of significant digits to retain in the uncertainty (default: 1).
        For example, err_digits=2 gives '1.234(56)'.

    Returns
    -------
    str
        A formatted string representing the value with its uncertainty, in the
        form "value(error)" where the error is shown in parentheses using the
        same scale as the last digits of the value. If formatting fails, falls
        back to the simpler "value ± error" representation.
    """
    try:
        # Ensure error is positive and finite
        if not np.isfinite(error) or error <= 0:
            raise ValueError("Error must be positive and finite.")

        # Determine number of decimal places to show
        digits = -int(np.floor(np.log10(error))) + err_digits - 1
        digits = max(digits, 0)

        # Scale the uncertainty accordingly and format
        scaled_err = round(error * 10**digits)
        return f"{value:.{digits}f}({int(scaled_err)})"

    except Exception:
        # Fallback in case of bad inputs
        return f"{value} ± {error}"
