import torch


class Distribution:
    def sample(self, n: int) -> torch.Tensor:
        """Returns tensor shape (n, 2)"""
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError
