"""Animación 2: Evolución de la densidad (KDE 2D)."""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import gaussian_kde


def animate_density_evolution(distributions, processes, proc_names,
                               n_samples=1500, n_frames=40, fps=15,
                               out_path="outputs/2_density_evolution.mp4"):
    ts = torch.linspace(0.0, 1.0, n_frames)
    grid_size = 60
    lim = 4.0
    xs = np.linspace(-lim, lim, grid_size)
    ys = np.linspace(-lim, lim, grid_size)
    XX, YY = np.meshgrid(xs, ys)
    pos = np.vstack([XX.ravel(), YY.ravel()])

    n_dist = min(len(distributions), 2)
    fig, axes = plt.subplots(n_dist, 1, figsize=(5, 4 * n_dist))
    if n_dist == 1:
        axes = [axes]
    imgs = []
    for ax in axes:
        im = ax.imshow(np.zeros((grid_size, grid_size)), origin="lower",
                       extent=[-lim, lim, -lim, lim], aspect="equal",
                       cmap="viridis", vmin=0, vmax=1)
        imgs.append(im)
        ax.set_xlabel("x"); ax.set_ylabel("y")

    time_text = fig.suptitle("")
    proc = processes[0]

    x0s = [dist.sample(n_samples) for dist in distributions[:n_dist]]

    def kde_frame(x0, t_val):
        t = torch.full((n_samples,), t_val)
        xt, _ = proc.sample_forward(x0, t)
        xy = xt.numpy()
        try:
            kde = gaussian_kde(xy.T, bw_method=0.10)
            Z = kde(pos).reshape(grid_size, grid_size)
        except Exception:
            Z = np.zeros((grid_size, grid_size))
        return Z / (Z.max() + 1e-8)

    def update(frame):
        t_val = ts[frame].item()
        for i, (x0, im, ax, dist) in enumerate(zip(x0s, imgs, axes, distributions[:n_dist])):
            Z = kde_frame(x0, t_val)
            im.set_data(Z)
            ax.set_title(f"{dist.name}")
        time_text.set_text(f"t = {t_val:.2f}")
        return imgs + [time_text]

    ani = animation.FuncAnimation(fig, update, frames=n_frames, blit=False, interval=1000 // fps)
    ani.save(out_path, writer="ffmpeg", fps=fps)
    plt.close(fig)
    print(f"  Saved: {out_path}")
