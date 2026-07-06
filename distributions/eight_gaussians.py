import torch
import math
from .base import Distribution


class EightGaussians(Distribution):
    def sample(self, n: int) -> torch.Tensor:
        centers = torch.tensor([
            [math.cos(2 * math.pi * i / 8), math.sin(2 * math.pi * i / 8)]
            for i in range(8)
        ]) * 2.0

        idx = torch.randint(8, (n,))
        samples = centers[idx] + torch.randn(n, 2) * 0.3
        return samples

    @property
    def name(self) -> str:
        return "eight_gaussians"
