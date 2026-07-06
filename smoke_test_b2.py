"""Bloque 2 smoke test: forward pass + predict_score."""
import sys
sys.path.insert(0, r"C:\Users\dguer\Desktop\Ciclo2026_1\GenerativeModels\Lab3")

import torch
from processes import VPProcess
from models import ScoreNet, VelocityNet

B = 64
process = VPProcess()

# --- ScoreNet epsilon ---
net_eps = ScoreNet(process, parametrization="epsilon")
x = torch.randn(B, 2)
t = torch.rand(B)
out = net_eps(x, t)
assert out.shape == (B, 2), f"ScoreNet(eps) out shape {out.shape}"
score = net_eps.predict_score(x, t)
assert score.shape == (B, 2), f"ScoreNet(eps) score shape {score.shape}"
assert not torch.isnan(score).any(), "ScoreNet(eps) score has NaN"
print(f"  ScoreNet(epsilon): forward {tuple(out.shape)}, score {tuple(score.shape)} PASS")

# --- ScoreNet v-prediction ---
net_v = ScoreNet(process, parametrization="v")
out_v = net_v(x, t)
assert out_v.shape == (B, 2)
score_v = net_v.predict_score(x, t)
assert score_v.shape == (B, 2)
assert not torch.isnan(score_v).any()
print(f"  ScoreNet(v): forward {tuple(out_v.shape)}, score {tuple(score_v.shape)} PASS")

# --- VelocityNet ---
vel_net = VelocityNet()
vel = vel_net(x, t)
assert vel.shape == (B, 2), f"VelocityNet out shape {vel.shape}"
assert not torch.isnan(vel).any()
print(f"  VelocityNet: forward {tuple(vel.shape)} PASS")

# --- Input dim check: 2 + 128 = 130 ---
from models import SinusoidalEmbedding
emb = SinusoidalEmbedding(dim=128)
t_emb = emb(t)
assert t_emb.shape == (B, 128), f"time emb shape {t_emb.shape}"
inp = torch.cat([x, t_emb], dim=-1)
assert inp.shape == (B, 130), f"concat shape {inp.shape}"
print(f"  Input dim: 2 + 128 = {inp.shape[1]} PASS")

# --- Grid score (30x30), no NaN ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

lim = 3.0
gx = torch.linspace(-lim, lim, 30)
gy = torch.linspace(-lim, lim, 30)
xx, yy = torch.meshgrid(gx, gy, indexing="ij")
grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)  # (900, 2)
t_grid = torch.full((900,), 0.5)
net_eps.eval()
with torch.no_grad():
    score_grid = net_eps.predict_score(grid, t_grid)
assert score_grid.shape == (900, 2)
assert not torch.isnan(score_grid).any(), "Grid score has NaN"
assert not torch.isinf(score_grid).any(), "Grid score has Inf"
print(f"  Grid score 30x30: shape={tuple(score_grid.shape)}, no NaN/Inf PASS")

fig, ax = plt.subplots(figsize=(5, 5))
sx = score_grid[:, 0].reshape(30, 30).numpy()
sy = score_grid[:, 1].reshape(30, 30).numpy()
ax.quiver(xx.numpy(), yy.numpy(), sx, sy, alpha=0.7)
ax.set_title("Untrained score field (noise, but no NaN)")
plt.tight_layout()
plt.savefig(r"C:\Users\dguer\Desktop\Ciclo2026_1\GenerativeModels\Lab3\outputs\smoke_b2_score_grid.png", dpi=80)
print("\nSmoke test B2: ALL PASS")
