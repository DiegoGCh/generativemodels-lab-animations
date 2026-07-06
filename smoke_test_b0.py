"""Bloque 0 smoke test: verify all 6 distributions."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from distributions import TwoMoons, EightGaussians, Checkerboard, Spirals, ConcentricRings, Pinwheel

dists = [TwoMoons(), EightGaussians(), Checkerboard(), Spirals(), ConcentricRings(), Pinwheel()]

fig, axes = plt.subplots(1, 6, figsize=(18, 3))
all_pass = True
for i, (dist, ax) in enumerate(zip(dists, axes)):
    s = dist.sample(2000)
    assert isinstance(s, torch.Tensor), f"{dist.name}: not Tensor"
    assert s.shape == (2000, 2), f"{dist.name}: shape {s.shape} != (2000, 2)"
    assert isinstance(dist.name, str) and len(dist.name) > 0, f"name empty"
    assert not torch.isnan(s).any(), f"{dist.name}: NaN in samples"
    ax.scatter(s[:, 0].numpy(), s[:, 1].numpy(), s=2, alpha=0.4)
    ax.set_title(dist.name)
    ax.set_aspect("equal")
    print(f"  [{i+1}/6] {dist.name}: shape={tuple(s.shape)}, range=[{s.min():.2f}, {s.max():.2f}] PASS")

plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig(os.path.join("outputs", "smoke_b0_distributions.png"), dpi=100)
print("\nSmoke test B0: ALL PASS")
print("Plot saved to outputs/smoke_b0_distributions.png")
