import torch
from .base import Sampler


class HeunSampler(Sampler):
    """
    Heun predictor-corrector (order 2) for ODEs.
    Two drift evaluations per step.
    """

    def __init__(self, mode: str = "diffusion"):
        assert mode in ("diffusion", "flow_matching")
        self.mode = mode

    def _drift(self, model, process, x, t_vec):
        if self.mode == "diffusion":
            score = model.predict_score(x, t_vec)
            return process.f(x, t_vec) - 0.5 * process.g(t_vec).unsqueeze(-1) ** 2 * score
        else:
            return model(x, t_vec)

    def sample(self, model, process, n_particles: int, n_steps: int,
               seed: int = 42, device: str = "cpu") -> torch.Tensor:
        torch.manual_seed(seed)
        traj = []

        if self.mode == "diffusion":
            t_start, t_end = process.T, 1e-5
            x = process.prior_sample(n_particles).to(device)
            ts = torch.linspace(t_start, t_end, n_steps + 1, device=device)
        else:
            t_start, t_end = 0.0, 1.0
            x = torch.randn(n_particles, 2, device=device)
            ts = torch.linspace(t_start, t_end, n_steps + 1, device=device)

        traj.append(x.clone())

        for i in range(n_steps):
            t_cur = ts[i]
            t_next = ts[i + 1]
            dt = t_next - t_cur
            t_vec_cur = t_cur.expand(n_particles)
            t_vec_next = t_next.expand(n_particles)

            d1 = self._drift(model, process, x, t_vec_cur)
            x_pred = x + dt * d1
            d2 = self._drift(model, process, x_pred, t_vec_next)
            x = x + dt / 2 * (d1 + d2)
            traj.append(x.clone())

        return torch.stack(traj, dim=0)
