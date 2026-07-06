"""Animación 8: Comparación de N pasos [10, 25, 50, 100, 250]."""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from samplers import EulerSampler


def animate_step_comparison(process_or_none, model, mode="diffusion",
                              n_particles=300, seed=42, fps=15,
                              out_path="outputs/8_step_comparison.mp4", device="cpu"):
    step_counts = [10, 25, 50, 100, 250]
    sampler = EulerSampler(mode=mode)
    model.eval()

    # Run all samplers with same seed -> same initial noise
    trajs = []
    for n_steps in step_counts:
        with torch.no_grad():
            traj = sampler.sample(model, process_or_none, n_particles, n_steps,
                                  seed=seed, device=device)
        trajs.append(traj.cpu().numpy())

    # All final frames for display
    finals = [t[-1] for t in trajs]
    all_pts = np.concatenate(finals)
    lim = max(abs(all_pts).max() * 1.1, 3.0)

    # Static figure: 5 panels showing final result
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    scs = []
    for ax, final, n_steps in zip(axes, finals, step_counts):
        sc = ax.scatter(final[:, 0], final[:, 1], s=3, alpha=0.6, c="steelblue")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(f"N={n_steps}")
        scs.append(sc)

    method_name = "Probability Flow ODE" if mode == "diffusion" else "Flow Matching"
    fig.suptitle(f"{method_name} — step count comparison")
    plt.tight_layout()

    # Animate by revealing trajectories side by side
    # Use n_frames = max_steps for smooth animation (subsample smaller trajs)
    n_frames = 60

    def get_frame(traj, frame_idx):
        total = traj.shape[0]
        t_idx = int(frame_idx / n_frames * (total - 1))
        return traj[t_idx]

    def update(frame):
        for i, (sc, traj) in enumerate(zip(scs, trajs)):
            pts = get_frame(traj, frame)
            sc.set_offsets(pts)
        return scs

    ani = animation.FuncAnimation(fig, update, frames=n_frames, blit=False, interval=1000 // fps)
    ani.save(out_path, writer="ffmpeg", fps=fps)
    plt.close(fig)
    print(f"  Saved: {out_path}")
