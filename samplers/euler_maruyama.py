import torch
from .base import Sampler


class EulerMaruyamaSampler(Sampler):
    """
    Euler-Maruyama for reverse-time SDE. Integrates T->0 (dt < 0).
    x_{k+1} = x_k + drift_rev(x_k, t_k)*dt + g(t_k)*sqrt(|dt|)*z
    drift_rev(x, t) = f(x,t) - g²(t)*s_theta(x,t)
    """

    def sample(self, model, process, n_particles: int, n_steps: int,
               seed: int = 42, device: str = "cpu") -> torch.Tensor:
        torch.manual_seed(seed)
        traj = []

        x = process.prior_sample(n_particles).to(device)
        ts = torch.linspace(process.T, 1e-5, n_steps + 1, device=device)
        traj.append(x.clone())

        for i in range(n_steps):
            t_cur = ts[i]
            dt = ts[i + 1] - t_cur  # negative
            t_vec = t_cur.expand(n_particles)

            score = model.predict_score(x, t_vec)
            g = process.g(t_vec).unsqueeze(-1)
            drift = process.f(x, t_vec) - g ** 2 * score

            z = torch.randn_like(x)
            x = x + drift * dt + g * torch.sqrt(torch.abs(dt)) * z
            traj.append(x.clone())

        return torch.stack(traj, dim=0)
