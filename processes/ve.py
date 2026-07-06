import torch
import math
from .base import DiffusionProcess


class VEProcess(DiffusionProcess):
    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 50.0, T: float = 1.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self._T = T

    def _sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        """sigma(t) = sigma_min * (sigma_max / sigma_min)^t  (geometric schedule)"""
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def m(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """sigma_bar(t) = sqrt(sigma(t)^2 - sigma(0)^2), so sigma_bar(0)=0."""
        s = self._sigma_t(t)
        s0 = self.sigma_min
        var = torch.clamp(s ** 2 - s0 ** 2, min=0.0)
        return torch.sqrt(var)

    def f(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    def g(self, t: torch.Tensor) -> torch.Tensor:
        log_ratio = math.log(self.sigma_max / self.sigma_min)
        return self._sigma_t(t) * math.sqrt(2.0 * log_ratio)

    def prior_sample(self, n: int) -> torch.Tensor:
        sigma_T = self._sigma_t(torch.tensor(self._T)).item()
        return torch.randn(n, 2) * sigma_T

    @property
    def T(self) -> float:
        return self._T
