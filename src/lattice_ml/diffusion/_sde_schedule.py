# Created by Javad Komijani, June 2026

"""Builds varios schedules for diffusion processes."""

from abc import ABC, abstractmethod
import torch
from numpy import pi


__all__ = [
    "OrnsteinUhlenbeckSchedule",
    "VPScheduleWithInverseTimeGamma",
    "SubVPScheduleWithInverseTimeGamma",
    "VEScheduleWithInverseTimeSigmaSquared",
]


# =============================================================================
class LinearDriftSDESchedule(torch.nn.Module, ABC):
    r"""
    Abstract base class for diffusion processes with linear drift.

    The forward process is defined by the stochastic differential equation

    .. math::
        dx(t) = -\gamma(t) x(t) dt + \sigma(t) dW_t,

    where :math:`W_t` is a standard Wiener process.

    Under standard assumptions, the process admits a Gaussian transition
    distribution of the form

    .. math::
        x(t_1) = a(t_0, t_1) x(t_0) + b(t_0, t_1) \epsilon,

    where :math:`\epsilon \sim \mathcal{N}(0, I)` and

    * :math:`a(t_0, t_1)` determines the conditional mean,
    * :math:`b(t_0, t_1)` determines the conditional standard deviation.

    Equivalently,

    .. math::
        p(x(t_1) | x(t_0))
        =
        \mathcal{N} (a(t_0, t_1) x(t_0), b^2(t_0, t_1) I).

    Methods
    -------
    gamma(t)
        Drift coefficient controlling deterministic contraction of the signal.

    sigma(t)
        Diffusion coefficient controlling instantaneous noise injection.

    transition_mean_scale(t0, t1)
        Coefficient :math:`a(t_0, t_1)` multiplying :math:`x(t_0)` in the
        conditional mean.

    transition_noise_std(t0, t1)
        Coefficient :math:`b(t_0, t_1)` multiplying a standard Gaussian random
        variable in the transition distribution.
    """

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Alias for `self.sigma(t)`."""
        return self.sigma(t)

    @abstractmethod
    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        """
        Return the drift coefficient at time `t`.

        Args:
            t (torch.Tensor): Time tensor with values in (0, 1).
        """

    @abstractmethod
    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """
        Return the instantaneous noise coefficient at time `t`.

        Args:
            t (torch.Tensor): Time tensor with values in (0, 1).
        """

    @abstractmethod
    def half_sigma_square(self, t: torch.Tensor) -> torch.Tensor:
        """
        Return the half square of instantaneous noise coefficient at time `t`.

        Args:
            t (torch.Tensor): Time tensor with values in (0, 1).
        """

    @abstractmethod
    def transition_mean_scale(self, t_0: torch.Tensor, t_1: torch.Tensor):
        """
        Return the mean scale factor for the transition from `t_0` to `t_1`,
        i.e. the signal attenuation factor over the transition.

        Args:
            t_0 (torch.Tensor): Start time tensor.
            t_1 (torch.Tensor): End time tensor. Note: `t_1 >= t_0`.
        """

    @abstractmethod
    def transition_noise_std(self, t_0: torch.Tensor, t_1: torch.Tensor):
        """
        Return the accumulated noise standard deviation for the transition
        from `t_0` to `t_1`.

        Args:
            t_0 (torch.Tensor): Start time tensor.
            t_1 (torch.Tensor): End time tensor. Note: `t_1 >= t_0`.
        """


# =============================================================================
class OrnsteinUhlenbeckSchedule(LinearDriftSDESchedule):
    r"""
    Ornstein–Uhlenbeck diffusion schedule.

    The forward process is defined by the stochastic differential equation

    .. math::
        dx(t) = -\gamma_0 x(t) dt + \sigma_0 dW_t,

    where both drift and diffusion coefficients are time independent.

    Args:
        gamma_0 (float): Constant scaling factor of the drift term.
        sigma_0 (float): Constant scaling factor of the noise term.
    """

    def __init__(self, gamma_0: float = pi, sigma_0: float = (2 * pi)**0.5):
        super().__init__()
        self.train(False)  # indicating it is not trainable
        self.register_buffer(
            "gamma_0", torch.tensor(gamma_0, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "sigma_0", torch.tensor(sigma_0, dtype=torch.get_default_dtype())
        )

    def gamma(self, t: torch.Tensor):
        return self.gamma_0

    def sigma(self, t: torch.Tensor):
        return self.sigma_0

    def half_sigma_square(self, t: torch.Tensor):
        return 0.5 * self.sigma_0 ** 2

    def transition_mean_scale(self, t_0: torch.Tensor, t_1: torch.Tensor):
        return torch.exp(-self.gamma_0 * (t_1 - t_0))

    def transition_noise_std(self, t_0: torch.Tensor, t_1: torch.Tensor):
        c = self.sigma_0 / (2 * self.gamma_0)**0.5
        return c * torch.sqrt(1 - torch.exp(-2 * self.gamma_0 * (t_1 - t_0)))


# =============================================================================
class VPScheduleWithInverseTimeGamma(LinearDriftSDESchedule):
    r"""
    Variance Preserving (VP) diffusion schedule with inverse-time `gamma`.

    The forward process is defined by the stochastic differential equation

    .. math::
        dx(t) = -\gamma(t) x(t) dt + \sigma(t) dW_t,

    where:

    .. math::
        \gamma(t) = \frac{\gamma_0}{1 - t + \epsilon}, \quad
        \sigma^2(t) = 2 \gamma(t),

    and :math:`\epsilon > 0` prevents divergence as :math:`t \to 1`.
    """

    EPS = 1e-8  # Small constant to regulate the divergence at t = 1

    gamma_0 = 1

    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        return self.gamma_0 / (1 - t + self.EPS)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return (2 * self.gamma(t)) ** 0.5

    def half_sigma_square(self, t: torch.Tensor) -> torch.Tensor:
        return self.gamma(t)

    def transition_mean_scale(self, t_0: torch.Tensor, t_1: torch.Tensor):
        ratio = (1 - t_1 + self.EPS) / (1 - t_0 + self.EPS)
        return ratio ** self.gamma_0

    def transition_noise_std(self, t_0: torch.Tensor, t_1: torch.Tensor):
        return torch.sqrt(1 - self.transition_mean_scale(t_0, t_1)**2)


# =============================================================================
class SubVPScheduleWithInverseTimeGamma(LinearDriftSDESchedule):
    r"""
    Sub Variance Preserving (VP) diffusion schedule with inverse-time `gamma`.

    The forward process is defined by the stochastic differential equation

    .. math::
        dx(t) = -\gamma(t) x(t) dt + \sigma(t) dW_t,

    where:

    .. math::
        \gamma(t) = \frac{1}{1 - t + \epsilon}, \quad
        \sigma^2(t) = 2 \gamma(t) (1 - e^{-\int_0^t \gamma(s) ds}),

    and :math:`\epsilon > 0` prevents divergence as :math:`t \to 1`.
    """

    EPS = 1e-8  # Small constant to regulate the divergence at t = 1

    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        return 1 / (1 - t + self.EPS)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return (2 * self.gamma(t) * self.transition_noise_std(0, t)) ** 0.5

    def half_sigma_square(self, t: torch.Tensor) -> torch.Tensor:
        return self.gamma(t) * self.transition_noise_std(0, t)

    def transition_mean_scale(self, t_0: torch.Tensor, t_1: torch.Tensor):
        ratio = (1 - t_1 + self.EPS) / (1 - t_0 + self.EPS)
        return ratio

    def transition_noise_std(self, t_0: torch.Tensor, t_1: torch.Tensor):
        ratio = (1 - t_1 + self.EPS) / (1 - t_0 + self.EPS)
        return torch.sqrt(t_1**2 - (ratio * t_0)**2) / (1 + self.EPS)


# =============================================================================
class VEScheduleWithInverseTimeSigmaSquared(LinearDriftSDESchedule):
    r"""
    Variance Exploding (VE) diffusion schedule with inverse-time variance.

    The forward process is defined by the stochastic differential equation

    .. math::
        dx(t) = \sigma(t)\, dW_t, \quad
        \sigma^2(t) = \frac{\sigma_0^2}{1 - t + \epsilon},

    i.e. pure noise injection with no drift (:math:`\gamma \equiv 0`).
    This schedule applies equally to Euclidean and to Lie-group diffusion,
    which only needs `sigma(t)` -- see Komijani, "Noise scheduling and linear
    dynamics in diffusion models on Lie groups," arXiv:2605.17326 (2026).

    :math:`\epsilon > 0` prevents divergence as :math:`t \to 1`.
    """

    EPS = 1e-8  # Small constant to regulate the divergence at t = 1

    def __init__(self, sigma_0: float = 1.0):
        """Initializes the schedule.

        Args:
            sigma_0 (float): Scaling factor (default is 1).
        """
        super().__init__()
        self.train(False)  # indicating it is not trainable
        self.register_buffer(
            "sigma_0", torch.tensor(sigma_0, dtype=torch.get_default_dtype())
        )

    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(t)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma_0 / (1 - t + self.EPS) ** 0.5

    def half_sigma_square(self, t: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.sigma(t) ** 2

    def transition_mean_scale(self, t_0: torch.Tensor, t_1: torch.Tensor):
        return 1.0

    def transition_noise_std(self, t_0: torch.Tensor, t_1: torch.Tensor):
        ratio = (1 - t_0 + self.EPS) / (1 - t_1 + self.EPS)
        return self.sigma_0 * torch.sqrt(torch.log(ratio).abs())
