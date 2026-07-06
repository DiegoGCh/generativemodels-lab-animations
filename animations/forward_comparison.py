"""Animación 1: Comparación de procesos forward VP/VE/Sub-VP."""
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def animate_forward_comparison(distribution, processes, proc_names,
                                n_samples=500, n_frames=50, fps=20,
                                out_path="outputs/1_forward_comparison.mp4"):
    x0 = distribution.sample(n_samples)
    ts = torch.linspace(0.0, 1.0, n_frames)

    # Pre-compute all forward samples
    all_xt = []
    for proc in processes:
        frames = []
        for t_val in ts:
            t = torch.full((n_samples,), t_val.item())
            xt, _ = proc.sample_forward(x0, t)
            frames.append(xt.numpy())
        all_xt.append(frames)

    # Fixed axis at ±5: VP/Sub-VP end at σ≈1, VE particles go off-screen at late t
    # (which correctly visualises "variance exploding"). Auto-scaling to VE crushes VP/Sub-VP.
    lim = 5.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    scatters = []
    for ax, name in zip(axes, proc_names):
        sc = ax.scatter([], [], s=3, alpha=0.5, c="steelblue")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal"); ax.set_title(name)
        scatters.append(sc)

    time_text = fig.suptitle("")

    def update(frame):
        for i, sc in enumerate(scatters):
            sc.set_offsets(all_xt[i][frame])
        t_val = ts[frame].item()
        time_text.set_text(f"t = {t_val:.2f}")
        return scatters + [time_text]

    ani = animation.FuncAnimation(fig, update, frames=n_frames, blit=False, interval=1000 // fps)
    ani.save(out_path, writer="ffmpeg", fps=fps)
    plt.close(fig)
    print(f"  Saved: {out_path}")
