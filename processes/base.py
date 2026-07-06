from typing import Tuple
import torch


class DiffusionProcess:
    def m(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sample_forward(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (x_t, epsilon) where x_t = m(t) x0 + sigma(t) epsilon."""
        mt = self.m(t)
        st = self.sigma(t)
        # broadcast: t is (B,), x0 is (B, 2)
        mt = mt.unsqueeze(-1)
        st = st.unsqueeze(-1)
        epsilon = torch.randn_like(x0)
        xt = mt * x0 + st * epsilon
        return xt, epsilon

    def f(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def g(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def prior_sample(self, n: int) -> torch.Tensor:
        raise NotImplementedError

    @property
    def T(self) -> float:
        raise NotImplementedError
