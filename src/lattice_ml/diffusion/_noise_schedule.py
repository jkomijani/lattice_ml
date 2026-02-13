# Created by Javad Komijani, 2025

"""Schedulers for diffusion training."""

import torch


__all__ = [
    "InverseTimeNoiseSchedule",
    "ConstantNoiseSchedule",
    "ExponentialNoiseSchedule",
    "CosineNoiseSchedule"
]


# =============================================================================
class InverseTimeNoiseSchedule(torch.nn.Module):
    """
    Noise standard deviation scheduler derived from an inverse-time variance
    law: Var(t) ∝ 1 / (1 - t).

    This scheduler provides both the instantaneous noise std as a function of
    time, and its cumulative value between two time points.
    """

    EPS = 1e-8  # Small constant to regulate the divergence at t = 1

    def __init__(self, sigma_0: float = 1.0):
        """Initialize the noise standard deviation scheduler.

        Args:
            sigma_0 (float): Scaling factor (default is 1).
        """
        super().__init__()
        self.train(False)  # indicating it is not trainable
        self.register_buffer("sigma_0", torch.tensor(sigma_0))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the instantaneous noise standard deviation at time `t`.

        Args:
            t (torch.Tensor): Time tensor with values in (0, 1).

        Returns:
            torch.Tensor: Standard deviation of noise at time `t`.
        """
        return self.sigma_0 / (1 + self.EPS - t) ** 0.5

    def cumulative(self, t_0: torch.Tensor, t_1: torch.Tensor) -> torch.Tensor:
        """Compute the cumulative noise std between two times `t_0` and `t_1`.

        Args:
            t_0 (torch.Tensor): Start time tensor.
            t_1 (torch.Tensor): End time tensor.

        Returns:
            torch.Tensor: Cumulative noise standard deviation.
        """
        t_max = 1 + self.EPS
        return self.sigma_0 * torch.sqrt(
            torch.log((t_max - t_0) / (t_max - t_1)).abs()
        )


# =============================================================================
class ConstantNoiseSchedule(torch.nn.Module):
    """
    Noise standard deviation scheduler derived from a constant variance.

    This scheduler provides both the instantaneous noise std as a function of
    time, and its cumulative value between two time points.
    """

    def __init__(self, sigma_0: float = 3.14):
        """Initialize the noise standard deviation scheduler.

        Args:
            sigma_0 (float): Scaling factor (default is 1).
        """
        super().__init__()
        self.train(False)  # indicating it is not trainable
        self.register_buffer("sigma_0", torch.tensor(sigma_0))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the instantaneous noise standard deviation at time `t`.

        Args:
            t (torch.Tensor): Time tensor with values in (0, 1).

        Returns:
            torch.Tensor: Standard deviation of noise at time `t`.
        """
        return self.sigma_0 * torch.ones_like(t)

    def cumulative(self, t_0: torch.Tensor, t_1: torch.Tensor) -> torch.Tensor:
        """Compute the cumulative noise std between two times `t_0` and `t_1`.

        Args:
            t_0 (torch.Tensor): Start time tensor.
            t_1 (torch.Tensor): End time tensor.

        Returns:
            torch.Tensor: Cumulative noise standard deviation.
        """
        return self.sigma_0 * torch.sqrt(t_1 - t_0)


# =============================================================================
class ExponentialNoiseSchedule(torch.nn.Module):
    """
    Noise standard deviation scheduler derived from an exponential variance
    law: Var(t) ∝ exp(2 gamma t).

    This scheduler provides both the instantaneous noise std as a function of
    time, and its cumulative value between two time points.
    """

    def __init__(self, sigma_0: float = 1.0, gamma: float = 1.0):
        """Initialize the noise standard deviation scheduler.

        Args:
            sigma_0 (float): Scaling factor (default is 1).
            gamma (float): Scaling factor of the exponent (default is 1).
        """
        super().__init__()
        self.train(False)  # indicating it is not trainable
        self.register_buffer("sigma_0", torch.tensor(sigma_0))
        self.register_buffer("gamma", torch.tensor(gamma))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the instantaneous noise standard deviation at time `t`.

        Args:
            t (torch.Tensor): Time tensor with values in (0, 1).

        Returns:
            torch.Tensor: Standard deviation of noise at time `t`.
        """
        return self.sigma_0 * torch.exp(self.gamma * t)

    def cumulative(self, t_0: torch.Tensor, t_1: torch.Tensor) -> torch.Tensor:
        """Compute the cumulative noise std between two times `t_0` and `t_1`.

        Args:
            t_0 (torch.Tensor): Start time tensor.
            t_1 (torch.Tensor): End time tensor.

        Returns:
            torch.Tensor: Cumulative noise standard deviation.
        """
        const = self.sigma_0 / (2 * self.gamma)**0.5
        exp_int = torch.exp(2*self.gamma * t_1) - torch.exp(2*self.gamma * t_0)
        return const * torch.sqrt(exp_int)


# =============================================================================
class CosineNoiseSchedule(torch.nn.Module):
    """
    Noise standard deviation scheduler derived from a cosine variance
    law: Var(t) ∝ cos^2(gamma t).

    This scheduler provides both the instantaneous noise std as a function of
    time, and its cumulative value between two time points.
    """

    def __init__(self, sigma_0: float = 1.0, gamma: float = 3.14):
        """Initialize the noise standard deviation scheduler.

        Args:
            sigma_0 (float): Scaling factor (default is 1).
            gamma (float): Scaling factor of the exponent (default is 3.14).
        """
        super().__init__()
        self.train(False)  # indicating it is not trainable
        self.register_buffer("sigma_0", torch.tensor(sigma_0))
        self.register_buffer("gamma", torch.tensor(gamma))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the instantaneous noise standard deviation at time `t`.

        Args:
            t (torch.Tensor): Time tensor with values in (0, 1).

        Returns:
            torch.Tensor: Standard deviation of noise at time `t`.
        """
        return self.sigma_0 * torch.cos(self.gamma * t)

    def cumulative(self, t_0: torch.Tensor, t_1: torch.Tensor) -> torch.Tensor:
        """Compute the cumulative noise std between two times `t_0` and `t_1`.

        Args:
            t_0 (torch.Tensor): Start time tensor.
            t_1 (torch.Tensor): End time tensor.

        Returns:
            torch.Tensor: Cumulative noise standard deviation.
        """
        linear_part = (t_1 - t_0) / 2
        sinusoidal_part = (
            torch.sin(2*self.gamma * t_1) - torch.sin(2*self.gamma * t_0)
        ) / (4 * self.gamma)
        return self.sigma_0 * torch.sqrt(linear_part + sinusoidal_part)
