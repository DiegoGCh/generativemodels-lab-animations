"""Animación 6: Probability Flow ODE (Euler/Heun), determinista."""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from samplers import EulerSampler, HeunSampler


def animate_probability_flow(process, model, n_particles=200, n_steps=100,
                               seed=42, fps=20, sampler_name="euler",
                               out_path="outputs/6_probability_flow.mp4", device="cpu"):
    sampler = HeunSampler(mode="diffusion") if sampler_name == "heun" else EulerSampler(mode="diffusion")
    model.eval()
    with torch.no_grad():
        traj = sampler.sample(model, process, n_particles, n_steps, seed=seed, device=device)
    traj_np = traj.cpu().numpy()

    lim = max(abs(traj_np[0]).max() * 1.1, 4.0)
    n_frames = traj_np.shape[0]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")

    lines = [ax.plot([], [], lw=0.5, alpha=0.6, c="green")[0] for _ in range(n_particles)]
    sc = ax.scatter([], [], s=5, c="steelblue", zorder=3)

    def update(frame):
        for i, line in enumerate(lines):
            line.set_data(traj_np[:frame + 1, i, 0], traj_np[:frame + 1, i, 1])
        sc.set_offsets(traj_np[frame])
        frac = frame / (n_frames - 1)
        t_val = process.T * (1 - frac)
        ax.set_title(f"Prob. Flow ODE ({sampler_name}) — t={t_val:.2f}")
        return lines + [sc]

    ani = animation.FuncAnimation(fig, update, frames=n_frames, blit=False, interval=1000 // fps)
    ani.save(out_path, writer="ffmpeg", fps=fps)
    plt.close(fig)
    print(f"  Saved: {out_path}")
