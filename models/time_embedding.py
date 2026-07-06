import torch
import torch.nn as nn
import math


class FourierFeatures(nn.Module):
    """Random Fourier Features for spatial input x.

    Projects x through fixed random frequencies so the MLP can represent
    high-frequency spatial structure (thin arcs, sharp boundaries) that
    plain MLPs miss due to spectral bias.
    """

    def __init__(self, in_dim: int = 2, n_freqs: int = 64, sigma: float = 1.0):
        super().__init__()
        B = torch.randn(in_dim, n_freqs) * sigma
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.B  # (batch, n_freqs)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (batch, 2*n_freqs)


class SinusoidalEmbedding(nn.Module):
    """Maps t in [0,1] to vector of dim `dim` via log-spaced Fourier features."""

    def __init__(self, dim: int = 128):
        super().__init__()
        assert dim % 2 == 0, "dim must be even"
        half = dim // 2
        # log-spaced frequencies between 1 and 1000
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), half))
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) -> (B, dim)
        t = t.unsqueeze(-1)  # (B, 1)
        args = 2 * math.pi * t * self.freqs  # (B, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
