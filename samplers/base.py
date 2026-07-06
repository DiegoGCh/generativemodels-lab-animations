import torch


class Sampler:
    def sample(self, model, process, n_particles: int, n_steps: int,
               seed: int = 42) -> torch.Tensor:
        """
        Returns tensor shape (n_steps+1, n_particles, 2) — full trajectory.
        """
        raise NotImplementedError
