import torch
import math
from .base import Distribution


class TwoMoons(Distribution):
    def sample(self, n: int) -> torch.Tensor:
        half = n // 2
        r = 1.0
        w = 0.3
        theta1 = torch.rand(half) * math.pi
        theta2 = torch.rand(n - half) * math.pi + math.pi

        x1 = torch.stack([r * torch.cos(theta1) - 0.5, r * torch.sin(theta1) - 0.25], dim=1)
        x2 = torch.stack([r * torch.cos(theta2) + 0.5, r * torch.sin(theta2) + 0.25], dim=1)

        x1 += torch.randn_like(x1) * w
        x2 += torch.randn_like(x2) * w

        return torch.cat([x1, x2], dim=0)

    @property
    def name(self) -> str:
        return "two_moons"
