import torch
import torch.nn as nn
from .time_embedding import SinusoidalEmbedding, FourierFeatures


class ScoreNet(nn.Module):
    """
    MLP predicting score s_theta(x_t, t).
    Internal parametrization: 'epsilon' or 'v'.
    Exposes predict_score() for unified score access.
    """

    def __init__(self, process, parametrization: str = "epsilon",
                 hidden_dim: int = 512, time_dim: int = 128,
                 n_freqs: int = 64, fourier_sigma: float = 1.0):
        super().__init__()
        self.process = process
        self.parametrization = parametrization
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
        """Raw network output (epsilon or v depending on parametrization)."""
        t_emb = self.time_emb(t)
        x_emb = self.fourier(x)
        inp = torch.cat([x_emb, t_emb], dim=-1)
        return self.net(inp)

    def predict_score(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Score s_theta(x_t, t), independent of internal parametrization."""
        out = self.forward(x, t)
        sigma = self.process.sigma(t).unsqueeze(-1)  # (B, 1)

        if self.parametrization == "epsilon":
            return -out / sigma
        elif self.parametrization == "v":
            # a_t = m(t), b_t = sigma(t)
            a_t = self.process.m(t).unsqueeze(-1)
            b_t = sigma
            # eps_hat = b_t * x_t + a_t * v_theta  (from rotation inverse)
            eps_hat = b_t * x + a_t * out
            return -eps_hat / sigma
        else:
            raise ValueError(f"Unknown parametrization: {self.parametrization}")
