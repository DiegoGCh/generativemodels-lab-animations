"""Bloque 1 smoke test: forward process at 5 time steps."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from distributions import TwoMoons
from processes import VPProcess, VEProcess, SubVPProcess

dist = TwoMoons()
x0 = dist.sample(1000)
times = [0.0, 0.25, 0.5, 0.75, 1.0]

processes = [VPProcess(), VEProcess(), SubVPProcess()]
proc_names = ["VP", "VE", "Sub-VP"]

fig, axes = plt.subplots(3, 5, figsize=(15, 9))

all_pass = True
for row, (proc, pname) in enumerate(zip(processes, proc_names)):
    for col, t_val in enumerate(times):
        t = torch.full((1000,), t_val)

        # test m, sigma shapes
        mt = proc.m(t)
        st = proc.sigma(t)
        assert mt.shape == (1000,), f"{pname} m shape {mt.shape}"
        assert st.shape == (1000,), f"{pname} sigma shape {st.shape}"

        xt, eps = proc.sample_forward(x0, t)
        assert xt.shape == (1000, 2), f"{pname} xt shape {xt.shape}"
        assert eps.shape == (1000, 2), f"{pname} eps shape {eps.shape}"
        assert not torch.isnan(xt).any(), f"{pname} NaN at t={t_val}"

        # test f, g
        if t_val > 0:
            ft = proc.f(xt, t)
            gt = proc.g(t)
            assert ft.shape == (1000, 2), f"{pname} f shape {ft.shape}"
            assert gt.shape == (1000,), f"{pname} g shape {gt.shape}"

        axes[row, col].scatter(xt[:, 0].numpy(), xt[:, 1].numpy(), s=1, alpha=0.4)
        axes[row, col].set_title(f"{pname} t={t_val}")
        axes[row, col].set_aspect("equal")

    # prior sample
    prior = proc.prior_sample(500)
    assert prior.shape == (500, 2), f"{pname} prior shape {prior.shape}"
    print(f"  {pname}: prior std={prior.std():.3f}, prior mean={prior.mean():.3f} PASS")

# t=1 should be noisy — check sigma at T
for proc, pname in zip(processes, proc_names):
    t1 = torch.tensor([1.0])
    s1 = proc.sigma(t1).item()
    print(f"  {pname} sigma(T=1): {s1:.4f}")

plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig(os.path.join("outputs", "smoke_b1_forward.png"), dpi=80)
print("\nSmoke test B1: ALL PASS")
print("Plot saved to outputs/smoke_b1_forward.png")
