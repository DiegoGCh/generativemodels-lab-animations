import torch
from .base import Distribution


class Checkerboard(Distribution):
    def sample(self, n: int) -> torch.Tensor:
        samples = []
        while len(samples) < n:
            batch = torch.rand(n * 3, 2) * 4 - 2
            x, y = batch[:, 0], batch[:, 1]
            xi = torch.floor(x + 2).long()
            yi = torch.floor(y + 2).long()
            mask = (xi + yi) % 2 == 0
            accepted = batch[mask]
            samples.append(accepted)
        return torch.cat(samples, dim=0)[:n]

    @property
    def name(self) -> str:
        return "checkerboard"
