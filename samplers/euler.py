import torch
from .base import Sampler


class EulerSampler(Sampler):
    """
    Euler integrator for ODEs.
    - Probability Flow ODE: integrates T->0 (dt < 0)
    - Flow Matching ODE: integrates 0->1 (dt > 0)
    """

    def __init__(self, mode: str = "diffusion"):
        assert mode in ("diffusion", "flow_matching")
        self.mode = mode

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
            t_vec = t_cur.expand(n_particles)

            if self.mode == "diffusion":
                score = model.predict_score(x, t_vec)
                drift = process.f(x, t_vec) - 0.5 * process.g(t_vec).unsqueeze(-1) ** 2 * score
            else:
                drift = model(x, t_vec)

            x = x + dt * drift
            traj.append(x.clone())

        return torch.stack(traj, dim=0)  # (n_steps+1, n_particles, 2)
