import torch
import math
from .base import Distribution


class Pinwheel(Distribution):
    def __init__(self, n_blades=5):
        self.n_blades = n_blades

    def sample(self, n: int) -> torch.Tensor:
        per_blade = n // self.n_blades
        counts = [per_blade] * (self.n_blades - 1) + [n - per_blade * (self.n_blades - 1)]
        parts = []
        for i, c in enumerate(counts):
            angle_offset = 2 * math.pi * i / self.n_blades
            t = torch.rand(c)
            r = 0.3 + t * 1.5
            theta = angle_offset + t * math.pi / self.n_blades
            noise_r = torch.randn(c) * 0.08
            noise_theta = torch.randn(c) * 0.08
            pts = torch.stack([
                (r + noise_r) * torch.cos(theta + noise_theta),
                (r + noise_r) * torch.sin(theta + noise_theta)
            ], dim=1)
            parts.append(pts)
        return torch.cat(parts, dim=0)

    @property
    def name(self) -> str:
        return "pinwheel"
