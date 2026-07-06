"""Animación 7: Flow Matching (Euler), partículas + trayectorias + campo de velocidades."""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from samplers import EulerSampler


def animate_flow_matching(fm_model, n_particles=200, n_steps=100,
                           seed=42, fps=20, grid_size=15,
                           out_path="outputs/7_flow_matching.mp4", device="cpu"):
    sampler = EulerSampler(mode="flow_matching")
    fm_model.eval()
    with torch.no_grad():
        traj = sampler.sample(fm_model, None, n_particles, n_steps, seed=seed, device=device)
    traj_np = traj.cpu().numpy()

    lim = max(abs(traj_np).max() * 1.1, 3.0)
    n_frames = traj_np.shape[0]

    # Grid for velocity field
    gx = torch.linspace(-lim, lim, grid_size)
    gy = torch.linspace(-lim, lim, grid_size)
    GX, GY = torch.meshgrid(gx, gy, indexing="ij")
    grid = torch.stack([GX.flatten(), GY.flatten()], dim=1).to(device)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")

    lines = [ax.plot([], [], lw=0.4, alpha=0.5, c="orange")[0] for _ in range(n_particles)]
    sc = ax.scatter([], [], s=6, c="steelblue", zorder=4)
    q = ax.quiver(GX.numpy(), GY.numpy(),
                  np.zeros((grid_size, grid_size)),
                  np.zeros((grid_size, grid_size)),
                  alpha=0.5, color="red", scale=None, zorder=2)

    ts_anim = torch.linspace(0.0, 1.0, n_frames)

    def update(frame):
        for i, line in enumerate(lines):
            line.set_data(traj_np[:frame + 1, i, 0], traj_np[:frame + 1, i, 1])
        sc.set_offsets(traj_np[frame])

        t_val = ts_anim[frame].item()
        t_grid = torch.full((grid_size * grid_size,), t_val, device=device)
        with torch.no_grad():
            vel = fm_model(grid, t_grid).cpu()
        vx = vel[:, 0].reshape(grid_size, grid_size).numpy()
        vy = vel[:, 1].reshape(grid_size, grid_size).numpy()
        q.set_UVC(vx, vy)
        ax.set_title(f"Flow Matching — t={t_val:.2f}")
        return lines + [sc, q]

    ani = animation.FuncAnimation(fig, update, frames=n_frames, blit=False, interval=1000 // fps)
    ani.save(out_path, writer="ffmpeg", fps=fps)
    plt.close(fig)
    print(f"  Saved: {out_path}")
