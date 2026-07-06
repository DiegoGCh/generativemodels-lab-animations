"""Animación 3: Trayectorias forward (50 partículas acumuladas)."""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def animate_forward_trajectories(distribution, process,
                                  n_particles=75, n_frames=60, fps=20,
                                  out_path="outputs/3_forward_trajectories.mp4"):
    x0 = distribution.sample(n_particles)
    ts = torch.linspace(0.0, 1.0, n_frames)

    # True forward SDE simulation (Euler-Maruyama) for physically correct paths
    x = x0.clone()
    traj = [x.numpy().copy()]
    for i in range(1, n_frames):
        dt = ts[i] - ts[i - 1]
        t_vec = ts[i - 1].expand(n_particles)
        drift = process.f(x, t_vec)
        g = process.g(t_vec).unsqueeze(-1)
        z = torch.randn_like(x)
        x = x + drift * dt + g * torch.sqrt(dt) * z
        traj.append(x.numpy().copy())
    traj = np.array(traj)  # (n_frames, n_particles, 2)

    lim = max(abs(traj).max() * 1.05, 3.0)
    colors = plt.cm.tab20(np.linspace(0, 1, n_particles))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_title(f"Forward trajectories — {distribution.name}")
    lines = [ax.plot([], [], lw=0.5, alpha=0.6, color=colors[i])[0] for i in range(n_particles)]
    dots = ax.scatter([], [], s=10, zorder=5)
    time_text = ax.set_title("")

    def update(frame):
        for i, line in enumerate(lines):
            line.set_data(traj[:frame + 1, i, 0], traj[:frame + 1, i, 1])
        dots.set_offsets(traj[frame])
        ax.set_title(f"Forward trajectories — t={ts[frame].item():.2f}")
        return lines + [dots]

    ani = animation.FuncAnimation(fig, update, frames=n_frames, blit=False, interval=1000 // fps)
    ani.save(out_path, writer="ffmpeg", fps=fps)
    plt.close(fig)
    print(f"  Saved: {out_path}")
