# Generative Models Lab: Diffusion and Flow Matching Animations

2D visualizations of score-based diffusion models (VP, VE, Sub-VP) and Flow Matching. Built for a graduate course on generative models. All animations are generated from models trained on synthetic 2D distributions and rendered as MP4 files.

---

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) installed and on PATH (required for MP4 rendering)
- CUDA-capable GPU recommended but not required

Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

## Project Structure

```
.
├── distributions/       # 2D data distributions (two_moons, eight_gaussians, etc.)
├── processes/           # Forward SDE definitions (VP, VE, Sub-VP)
├── models/              # ScoreNet and VelocityNet architectures
├── training/            # DiffusionTrainer and FlowMatchingTrainer
├── samplers/            # Euler, Heun, and Euler-Maruyama integrators
├── animations/          # One module per animation (8 total)
├── utils/
│   └── preview.py       # Preview utility: saves 5 keyframe PNGs instead of full MP4
├── main.py              # CLI entry point
├── requirements.txt
```

---

## Quickstart

### 1. Train a diffusion model (VP, epsilon-parametrization)

```bash
python main.py train \
  --distribution two_moons \
  --model diffusion \
  --process vp \
  --parametrization epsilon \
  --n_epochs 2000
```

Checkpoint saved to `checkpoints/two_moons_diffusion_vp_epsilon/`.

### 2. Train a Flow Matching model

```bash
python main.py train \
  --distribution two_moons \
  --model flow_matching \
  --n_epochs 2000
```

Checkpoint saved to `checkpoints/two_moons_flow_matching/`.

---

## Animations

All animations write to `outputs/`. Use `--preview` to generate 5 keyframe PNGs in `preview/` instead of rendering a full MP4 (fast, useful for checking quality before committing to a long render).

### Animation 1: Forward Process Comparison

Shows VP, VE, and Sub-VP side by side as t goes from 0 to 1. No model required.

```bash
python main.py animate \
  --distribution two_moons \
  --model diffusion \
  --process vp \
  --animation 1
```

### Animation 2: Density Evolution

KDE density of two distributions evolving over time under a single forward SDE.

```bash
python main.py animate \
  --distribution two_moons \
  --model diffusion \
  --process vp \
  --animation 2
```

### Animation 3: Forward Trajectories

True Euler-Maruyama particle paths from t=0 to t=1.

```bash
python main.py animate \
  --distribution two_moons \
  --model diffusion \
  --process vp \
  --animation 3
```

### Animation 4: Score Field

Learned score field s(x,t) on a grid, animated as t varies from 0.05 to 1.00.

```bash
python main.py animate \
  --distribution two_moons \
  --model diffusion \
  --process vp \
  --parametrization epsilon \
  --animation 4 \
  --checkpoint checkpoints/two_moons_diffusion_vp_epsilon
```

### Animation 5: Reverse SDE (Euler-Maruyama)

Denoising via reverse-time SDE. Particles start from the Gaussian prior and step backward to t=0.

```bash
python main.py animate \
  --distribution two_moons \
  --model diffusion \
  --process vp \
  --parametrization epsilon \
  --animation 5 \
  --checkpoint checkpoints/two_moons_diffusion_vp_epsilon \
  --n_steps 250
```

### Animation 6: Probability Flow ODE

Same denoising goal as Animation 5, but deterministic. Uses Heun integrator (2nd order).

```bash
python main.py animate \
  --distribution two_moons \
  --model diffusion \
  --process vp \
  --parametrization epsilon \
  --animation 6 \
  --checkpoint checkpoints/two_moons_diffusion_vp_epsilon \
  --n_steps 250 \
  --sampler heun
```

### Animation 7: Flow Matching

Particle transport from Gaussian to data using the learned velocity field. Independent of any diffusion process.

```bash
python main.py animate \
  --distribution two_moons \
  --model flow_matching \
  --animation 7 \
  --checkpoint checkpoints/two_moons_flow_matching \
  --n_steps 250
```

### Animation 8: Step Count Comparison

Five panels comparing N = 10, 25, 50, 100, 250 integration steps side by side. Shows the effect of discretization on sample quality.

For diffusion (Probability Flow ODE):

```bash
python main.py animate \
  --distribution two_moons \
  --model diffusion \
  --process vp \
  --parametrization epsilon \
  --animation 8 \
  --checkpoint checkpoints/two_moons_diffusion_vp_epsilon
```

For Flow Matching:

```bash
python main.py animate \
  --distribution two_moons \
  --model flow_matching \
  --animation 8 \
  --checkpoint checkpoints/two_moons_flow_matching
```

---

## Preview Mode

Before rendering a full MP4, you can check 5 keyframes (0%, 25%, 50%, 75%, 100%) as PNGs:

```bash
python main.py animate ... --preview
```

Output goes to `preview/`. Takes a few seconds instead of minutes.

---

## Available Distributions

`two_moons`, `eight_gaussians`, `checkerboard`, `spirals`, `concentric_rings`, `pinwheel`

## Available Processes

`vp` (Variance Preserving), `ve` (Variance Exploding), `sub_vp` (Sub-VP)

## Available Samplers

`euler`, `heun` (ODE only), `em` (Euler-Maruyama, SDE only)

---

## Smoke Tests

Each implementation block has a smoke test that verifies correctness without running a full training or animation. They are a fast way to confirm the environment is working after a fresh clone.

| Script | What it checks |
|--------|---------------|
| `smoke_test_b0.py` | All 6 distributions sample correctly, shapes and no NaN |
| `smoke_test_b1.py` | VP, VE, Sub-VP forward SDE at 5 time steps, prior sampling |
| `smoke_test_b2.py` | ScoreNet and VelocityNet forward pass, predict_score output, time embedding dim |
| `smoke_test_b3.py` | 500-epoch training loss drops, checkpoint save and reload, score grid no NaN |
| `smoke_test_b4.py` | Euler, Heun, and Euler-Maruyama trajectory shapes and dt sign conventions |
| `smoke_test_b5.py` | All 8 animation functions run and write MP4 files |
| `smoke_test_b6.py` | End-to-end: train, sample, and animate in sequence |

Run any of them from the project root:

```bash
python smoke_test_b0.py
python smoke_test_b1.py
# etc.
```

Each prints a PASS/FAIL line per check and saves a diagnostic PNG to `outputs/`. They are designed to run in under two minutes each (b3 and above require a GPU for reasonable speed).

---

## Notes

- Checkpoints and output files are included in the repo. You can run animations 4-8 directly using the provided checkpoints without retraining.
- ffmpeg must be installed separately. On conda: `conda install -c conda-forge ffmpeg`.
- The `--preview` flag is useful during development to avoid long render times.
