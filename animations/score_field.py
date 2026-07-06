"""Animación 4: Campo de score animado variando t."""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def animate_score_field(distribution, process, model,
                         n_samples=200, grid_size=20, n_frames=40, fps=15,
                         out_path="outputs/4_score_field.mp4", device="cpu"):
    x0 = distribution.sample(n_samples)
    ts = torch.linspace(0.05, 1.00, n_frames)

    lim = 3.0
    gx = torch.linspace(-lim, lim, grid_size)
    gy = torch.linspace(-lim, lim, grid_size)
    GX, GY = torch.meshgrid(gx, gy, indexing="ij")
    grid = torch.stack([GX.flatten(), GY.flatten()], dim=1).to(device)

    model.eval()

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")

    sc = ax.scatter([], [], s=5, alpha=0.5, c="steelblue", zorder=3)
    q = ax.quiver(GX.numpy(), GY.numpy(),
                  np.zeros((grid_size, grid_size)),
                  np.zeros((grid_size, grid_size)),
                  alpha=0.7, color="red", scale=None)

    def update(frame):
        t_val = ts[frame].item()
        t_tensor = torch.full((n_samples,), t_val)
        xt, _ = process.sample_forward(x0, t_tensor)

        t_grid = torch.full((grid_size * grid_size,), t_val, device=device)
        with torch.no_grad():
            score = model.predict_score(grid, t_grid).cpu()

        sc.set_offsets(xt.numpy())
        sx = score[:, 0].reshape(grid_size, grid_size).numpy()
        sy = score[:, 1].reshape(grid_size, grid_size).numpy()
        max_mag = np.sqrt(sx ** 2 + sy ** 2).max() + 1e-8
        q.set_UVC(sx / max_mag, sy / max_mag)
        ax.set_title(f"Score field — t={t_val:.2f}")
        return sc, q

    ani = animation.FuncAnimation(fig, update, frames=n_frames, blit=False, interval=1000 // fps)
    ani.save(out_path, writer="ffmpeg", fps=fps)
    plt.close(fig)
    print(f"  Saved: {out_path}")
