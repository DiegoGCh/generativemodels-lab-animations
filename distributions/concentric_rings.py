import torch
import math
from .base import Distribution


class ConcentricRings(Distribution):
    def __init__(self, radii=(1.0, 2.0, 3.0), width=0.15):
        self.radii = radii
        self.width = width

    def sample(self, n: int) -> torch.Tensor:
        per_ring = n // len(self.radii)
        counts = [per_ring] * (len(self.radii) - 1) + [n - per_ring * (len(self.radii) - 1)]
        parts = []
        for r, c in zip(self.radii, counts):
            theta = torch.rand(c) * 2 * math.pi
            radius = r + torch.randn(c) * self.width
            pts = torch.stack([radius * torch.cos(theta), radius * torch.sin(theta)], dim=1)
            parts.append(pts)
        return torch.cat(parts, dim=0)

    @property
    def name(self) -> str:
        return "concentric_rings"
