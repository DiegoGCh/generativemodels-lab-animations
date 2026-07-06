"""Bloque 5 smoke test: generate all 8 animations."""
import sys
sys.path.insert(0, r"C:\Users\dguer\Desktop\Ciclo2026_1\GenerativeModels\Lab3")

import os
import torch
import matplotlib
matplotlib.use("Agg")

from distributions import TwoMoons, EightGaussians
from processes import VPProcess, VEProcess, SubVPProcess
from models import ScoreNet, VelocityNet
from training import DiffusionTrainer, FlowMatchingTrainer
from animations import (
    animate_forward_comparison, animate_density_evolution,
    animate_forward_trajectories, animate_score_field,
    animate_reverse_sde, animate_probability_flow,
    animate_flow_matching, animate_step_comparison,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
OUT = r"C:\Users\dguer\Desktop\Ciclo2026_1\GenerativeModels\Lab3\outputs"
os.makedirs(OUT, exist_ok=True)

dist = TwoMoons()
dist2 = EightGaussians()
vp = VPProcess()
ve = VEProcess()
sub_vp = SubVPProcess()

# Load or train models
ckpt_diff = r"C:\Users\dguer\Desktop\Ciclo2026_1\GenerativeModels\Lab3\checkpoints\two_moons_diffusion_vp_epsilon"
model_diff = ScoreNet(vp, parametrization="epsilon")
if os.path.exists(os.path.join(ckpt_diff, "model.pt")):
    DiffusionTrainer.load_model(model_diff, ckpt_diff, device=device)
    print("  Loaded diffusion checkpoint")
else:
    print("  Training diffusion 500 epochs...")
    t = DiffusionTrainer(model_diff, vp, dist, n_epochs=500, batch_size=4096, lr=3e-4, t_eps=1e-5, seed=42, device=device)
    t.train(log_every=250); t.save(ckpt_diff)
model_diff = model_diff.to(device).eval()

ckpt_fm = r"C:\Users\dguer\Desktop\Ciclo2026_1\GenerativeModels\Lab3\checkpoints\two_moons_flow_matching"
fm_model = VelocityNet()
if os.path.exists(os.path.join(ckpt_fm, "model.pt")):
    FlowMatchingTrainer.load_model(fm_model, ckpt_fm, device=device)
    print("  Loaded FM checkpoint")
else:
    print("  Training FM 500 epochs...")
    t = FlowMatchingTrainer(fm_model, dist, n_epochs=500, batch_size=4096, lr=3e-4, seed=42, device=device)
    t.train(log_every=250); t.save(ckpt_fm)
fm_model = fm_model.to(device).eval()

# --- Animación 1 ---
print("Generating animation 1...")
animate_forward_comparison(
    dist, [vp, ve, sub_vp], ["VP", "VE", "Sub-VP"],
    n_samples=400, n_frames=50, fps=20,
    out_path=f"{OUT}/1_forward_comparison_two_moons.mp4"
)

# --- Animación 2 ---
print("Generating animation 2...")
animate_density_evolution(
    [dist, dist2], [vp], ["VP"],
    n_samples=800, n_frames=40, fps=15,
    out_path=f"{OUT}/2_density_evolution_two_moons.mp4"
)

# --- Animación 3 ---
print("Generating animation 3...")
animate_forward_trajectories(
    dist, vp, n_particles=50, n_frames=60, fps=20,
    out_path=f"{OUT}/3_forward_trajectories_two_moons_vp.mp4"
)

# --- Animación 4 ---
print("Generating animation 4...")
animate_score_field(
    dist, vp, model_diff, n_samples=200, grid_size=20, n_frames=40, fps=15,
    out_path=f"{OUT}/4_score_field_two_moons_vp_epsilon.mp4", device=device
)

# --- Animación 5 ---
print("Generating animation 5...")
animate_reverse_sde(
    vp, model_diff, n_particles=200, n_steps=100, seed=42, fps=20,
    out_path=f"{OUT}/5_reverse_sde_two_moons_vp_epsilon_N100.mp4", device=device
)

# --- Animación 6 ---
print("Generating animation 6...")
animate_probability_flow(
    vp, model_diff, n_particles=200, n_steps=100, seed=42, fps=20,
    out_path=f"{OUT}/6_probability_flow_two_moons_vp_epsilon_N100.mp4", device=device
)

# --- Animación 7 ---
print("Generating animation 7...")
animate_flow_matching(
    fm_model, n_particles=200, n_steps=100, seed=42, fps=20, grid_size=15,
    out_path=f"{OUT}/7_flow_matching_two_moons_N100.mp4", device=device
)

# --- Animación 8 ---
print("Generating animation 8...")
animate_step_comparison(
    vp, model_diff, mode="diffusion", n_particles=300, seed=42, fps=15,
    out_path=f"{OUT}/8_step_comparison_two_moons_diffusion.mp4", device=device
)

# Verify all outputs
expected = [
    "1_forward_comparison_two_moons.mp4",
    "2_density_evolution_two_moons.mp4",
    "3_forward_trajectories_two_moons_vp.mp4",
    "4_score_field_two_moons_vp_epsilon.mp4",
    "5_reverse_sde_two_moons_vp_epsilon_N100.mp4",
    "6_probability_flow_two_moons_vp_epsilon_N100.mp4",
    "7_flow_matching_two_moons_N100.mp4",
    "8_step_comparison_two_moons_diffusion.mp4",
]
all_ok = True
for fname in expected:
    path = os.path.join(OUT, fname)
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    ok = exists and size > 1000
    print(f"  {'✓' if ok else '✗'} {fname} ({size} bytes)")
    if not ok:
        all_ok = False

if all_ok:
    print("\nSmoke test B5: ALL PASS")
else:
    print("\nSmoke test B5: SOME FAILED")
    sys.exit(1)
