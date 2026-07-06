import torch
import torch.nn as nn
from .time_embedding import SinusoidalEmbedding, FourierFeatures


class VelocityNet(nn.Module):
    """MLP predicting velocity field v_theta(x_t, t) for Flow Matching."""

    def __init__(self, hidden_dim: int = 512, time_dim: int = 128,
                 n_freqs: int = 64, fourier_sigma: float = 1.0):
        super().__init__()
        self.time_emb = SinusoidalEmbedding(dim=time_dim)
        self.fourier = FourierFeatures(in_dim=2, n_freqs=n_freqs, sigma=fourier_sigma)

        in_dim = 2 * n_freqs + time_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)
        x_emb = self.fourier(x)
        inp = torch.cat([x_emb, t_emb], dim=-1)
        return self.net(inp)
