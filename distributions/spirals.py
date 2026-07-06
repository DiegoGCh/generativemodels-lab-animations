import torch
import math
from .base import Distribution


class Spirals(Distribution):
    def sample(self, n: int) -> torch.Tensor:
        half = n // 2
        t = torch.rand(half) * 3 * math.pi + 0.5

        x1 = torch.stack([t * torch.cos(t) / (3 * math.pi), t * torch.sin(t) / (3 * math.pi)], dim=1)
        x2 = -x1[:n - half]

        x1 += torch.randn_like(x1) * 0.05
        x2 += torch.randn_like(x2) * 0.05

        return torch.cat([x1, x2], dim=0)

    @property
    def name(self) -> str:
        return "spirals"
