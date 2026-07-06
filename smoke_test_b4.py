"""Bloque 4 smoke test: all 3 samplers produce correct trajectory shapes."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from distributions import TwoMoons
from processes import VPProcess
from models import ScoreNet, VelocityNet
from training import DiffusionTrainer, FlowMatchingTrainer
from samplers import EulerSampler, HeunSampler, EulerMaruyamaSampler

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

dist = TwoMoons()
process = VPProcess()
n_particles = 500
n_steps = 100

# Load or train diffusion model (VP epsilon)
ckpt_diff = os.path.join("checkpoints", "two_moons_diffusion_vp_epsilon")
model_diff = ScoreNet(process, parametrization="epsilon")
if os.path.exists(os.path.join(ckpt_diff, "model.pt")):
    DiffusionTrainer.load_model(model_diff, ckpt_diff, device=device)
    print("  Loaded diffusion checkpoint")
else:
    print("  Training diffusion (500 epochs)...")
    trainer = DiffusionTrainer(model_diff, process, dist, n_epochs=500, batch_size=4096,
                                lr=3e-4, t_eps=1e-5, seed=42, device=device)
    trainer.train(log_every=250)
    trainer.save(ckpt_diff)
model_diff = model_diff.to(device)
model_diff.eval()

# Load or train FM model
ckpt_fm = os.path.join("checkpoints", "two_moons_flow_matching")
fm_model = VelocityNet()
if os.path.exists(os.path.join(ckpt_fm, "model.pt")):
    FlowMatchingTrainer.load_model(fm_model, ckpt_fm, device=device)
    print("  Loaded FM checkpoint")
else:
    print("  Training FM (500 epochs)...")
    trainer_fm = FlowMatchingTrainer(fm_model, dist, n_epochs=500, batch_size=4096,
                                      lr=3e-4, seed=42, device=device)
    trainer_fm.train(log_every=250)
    trainer_fm.save(ckpt_fm)
fm_model = fm_model.to(device)
fm_model.eval()

# --- Euler (Probability Flow ODE) ---
euler = EulerSampler(mode="diffusion")
with torch.no_grad():
    traj_euler = euler.sample(model_diff, process, n_particles, n_steps, seed=42, device=device)
assert traj_euler.shape == (n_steps + 1, n_particles, 2), f"Euler shape {traj_euler.shape}"
assert not torch.isnan(traj_euler).any(), "Euler: NaN in trajectory"
print(f"  Euler PF-ODE: shape={tuple(traj_euler.shape)} PASS")

# --- Heun (Probability Flow ODE) ---
heun = HeunSampler(mode="diffusion")
with torch.no_grad():
    traj_heun = heun.sample(model_diff, process, n_particles, n_steps, seed=42, device=device)
assert traj_heun.shape == (n_steps + 1, n_particles, 2), f"Heun shape {traj_heun.shape}"
assert not torch.isnan(traj_heun).any(), "Heun: NaN in trajectory"
print(f"  Heun PF-ODE: shape={tuple(traj_heun.shape)} PASS")

# --- Euler-Maruyama (reverse SDE) ---
em = EulerMaruyamaSampler()
with torch.no_grad():
    traj_em = em.sample(model_diff, process, n_particles, n_steps, seed=42, device=device)
assert traj_em.shape == (n_steps + 1, n_particles, 2), f"EM shape {traj_em.shape}"
assert not torch.isnan(traj_em).any(), "EM: NaN in trajectory"
print(f"  Euler-Maruyama SDE: shape={tuple(traj_em.shape)} PASS")

# dt sign check: diffusion starts at T goes to 0
assert traj_euler[0].std() > traj_euler[-1].std() * 0.5, "Diffusion should denoise"
print(f"  dt sign (diffusion T->0): init_std={traj_euler[0].std():.3f}, final_std={traj_euler[-1].std():.3f} PASS")

# --- Flow Matching Euler ---
euler_fm = EulerSampler(mode="flow_matching")
with torch.no_grad():
    traj_fm = euler_fm.sample(fm_model, None, n_particles, n_steps, seed=42, device=device)
assert traj_fm.shape == (n_steps + 1, n_particles, 2), f"FM Euler shape {traj_fm.shape}"
assert not torch.isnan(traj_fm).any()
print(f"  Euler FM-ODE: shape={tuple(traj_fm.shape)} PASS")

# dt sign: FM starts at noise, ends at data-like
assert traj_fm[-1].abs().mean() < traj_fm[0].abs().mean() * 3, "FM should transport toward data"
print(f"  dt sign (FM 0->1): init_std={traj_fm[0].std():.3f}, final_std={traj_fm[-1].std():.3f} PASS")

# Visualize final samples
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
data_ref = dist.sample(500)
for ax, traj, title in zip(axes,
    [traj_euler, traj_heun, traj_em, traj_fm],
    ["Euler PF-ODE", "Heun PF-ODE", "EM reverse SDE", "Euler FM"]):
    final = traj[-1].cpu()
    ax.scatter(data_ref[:, 0], data_ref[:, 1], s=3, alpha=0.3, c="blue", label="ref")
    ax.scatter(final[:, 0], final[:, 1], s=3, alpha=0.5, c="red", label="sampled")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig(os.path.join("outputs", "smoke_b4_samplers.png"), dpi=80)

print("\nSmoke test B4: ALL PASS")
