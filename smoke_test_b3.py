"""Bloque 3 smoke test: 500 epochs training, loss drops, checkpoint round-trip, score grid."""
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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

dist = TwoMoons()
process = VPProcess()

# ---- 1. Diffusion (VP + epsilon) ----
print("\n--- Diffusion VP epsilon ---")
model_diff = ScoreNet(process, parametrization="epsilon")
trainer_diff = DiffusionTrainer(model_diff, process, dist,
                                 n_epochs=500, batch_size=4096, lr=3e-4,
                                 t_eps=1e-5, seed=42, device=device)
losses_diff = trainer_diff.train(log_every=100)
assert losses_diff[0] > losses_diff[-1], f"Loss did not drop! {losses_diff[0]:.4f} -> {losses_diff[-1]:.4f}"
print(f"  Loss drop: {losses_diff[0]:.4f} -> {losses_diff[-1]:.4f} PASS")

# Checkpoint save + reload
ckpt = os.path.join("checkpoints", "two_moons_diffusion_vp_epsilon")
trainer_diff.save(ckpt)

# Reload and verify identical weights
model_reload = ScoreNet(process, parametrization="epsilon")
DiffusionTrainer.load_model(model_reload, ckpt, device=device)
x_test = torch.randn(10, 2).to(device)
t_test = torch.full((10,), 0.5).to(device)
model_diff.eval(); model_reload.eval()
with torch.no_grad():
    out_orig = model_diff(x_test, t_test)
    out_reload = model_reload.to(device)(x_test, t_test)
assert torch.allclose(out_orig, out_reload), "Reloaded weights differ!"
print("  Checkpoint reload: identical weights PASS")

# Score grid post-training
lim = 3.0
gx = torch.linspace(-lim, lim, 30)
gy = torch.linspace(-lim, lim, 30)
xx, yy = torch.meshgrid(gx, gy, indexing="ij")
grid = torch.stack([xx.flatten(), yy.flatten()], dim=1).to(device)
t_grid = torch.full((900,), 0.1).to(device)
model_diff.eval()
with torch.no_grad():
    score_grid = model_diff.predict_score(grid, t_grid)
assert not torch.isnan(score_grid).any()
print("  Score grid: no NaN PASS")

# Save score quiver plot
fig, ax = plt.subplots(figsize=(5, 5))
sx = score_grid[:, 0].cpu().reshape(30, 30).numpy()
sy = score_grid[:, 1].cpu().reshape(30, 30).numpy()
data_pts = dist.sample(500)
ax.scatter(data_pts[:, 0].numpy(), data_pts[:, 1].numpy(), s=3, alpha=0.3, c="blue", label="data")
ax.quiver(xx.numpy(), yy.numpy(), sx, sy, alpha=0.7, color="red")
ax.set_title("Score field at t=0.1 (500 epoch VP)")
plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig(os.path.join("outputs", "smoke_b3_score_field.png"), dpi=80)

# ---- 2. Flow Matching ----
print("\n--- Flow Matching ---")
fm_model = VelocityNet()
trainer_fm = FlowMatchingTrainer(fm_model, dist,
                                  n_epochs=500, batch_size=4096, lr=3e-4,
                                  seed=42, device=device)
losses_fm = trainer_fm.train(log_every=100)
assert losses_fm[0] > losses_fm[-1], f"FM Loss did not drop! {losses_fm[0]:.4f} -> {losses_fm[-1]:.4f}"
print(f"  FM Loss drop: {losses_fm[0]:.4f} -> {losses_fm[-1]:.4f} PASS")

ckpt_fm = os.path.join("checkpoints", "two_moons_flow_matching")
trainer_fm.save(ckpt_fm)
print("  FM checkpoint saved PASS")

print("\nSmoke test B3: ALL PASS")
