import torch
from .base import DiffusionProcess


class VPProcess(DiffusionProcess):
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0, T: float = 1.0):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self._T = T

    def _B(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min * t + (self.beta_max - self.beta_min) / 2.0 * t ** 2

    def _beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def m(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self._B(t) / 2.0)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(1.0 - torch.exp(-self._B(t)))

    def f(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        beta = self._beta(t).unsqueeze(-1)
        return -0.5 * beta * x

    def g(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self._beta(t))

    def prior_sample(self, n: int) -> torch.Tensor:
        return torch.randn(n, 2)

    @property
    def T(self) -> float:
        return self._T
