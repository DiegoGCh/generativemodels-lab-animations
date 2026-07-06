import torch
import torch.nn as nn
import math


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
