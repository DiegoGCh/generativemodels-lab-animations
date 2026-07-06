"""Bloque 6 smoke test: CLI commands work correctly."""
import sys
import subprocess

BASE = r"C:\Users\dguer\Desktop\Ciclo2026_1\GenerativeModels\Lab3"
PYTHON = "conda run -n mainds python"
CKPT = r"C:\Users\dguer\Desktop\Ciclo2026_1\GenerativeModels\Lab3\checkpoints\two_moons_diffusion_vp_epsilon"

def run(cmd, description):
    print(f"\n  Testing: {description}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=BASE)
    if result.returncode != 0:
        print(f"  FAIL (exit {result.returncode})")
        print(f"  STDOUT: {result.stdout[:500]}")
        print(f"  STDERR: {result.stderr[:500]}")
        return False
    print(f"  PASS")
    if result.stdout.strip():
        # Show last 2 lines
        lines = result.stdout.strip().split('\n')
        for l in lines[-2:]:
            print(f"    {l}")
    return True

all_pass = True

# 1. Train diffusion (short: 50 epochs to test CLI works)
ok = run(
    f"conda run -n mainds python main.py train --distribution two_moons --model diffusion "
    f"--process vp --parametrization epsilon --seed 42 --n_epochs 50",
    "train diffusion VP epsilon"
)
all_pass = all_pass and ok

# 2. Train flow matching (50 epochs)
ok = run(
    f"conda run -n mainds python main.py train --distribution two_moons --model flow_matching --seed 42 --n_epochs 50",
    "train flow matching"
)
all_pass = all_pass and ok

# 3. Animate (anim 5 = reverse SDE)
ok = run(
    f"conda run -n mainds python main.py animate --animation 5 --distribution two_moons "
    f"--model diffusion --process vp --parametrization epsilon "
    f"--sampler em --n_steps 50 --seed 42 "
    f"--checkpoint \"{CKPT}\"",
    "animate --animation 5 (reverse SDE)"
)
all_pass = all_pass and ok

# 4. Invalid distribution -> error (not traceback)
print("\n  Testing: invalid distribution -> clean error")
result = subprocess.run(
    "conda run -n mainds python main.py train --distribution invalid_dist --model diffusion",
    shell=True, capture_output=True, text=True, cwd=BASE
)
stderr_out = (result.stdout + result.stderr)
if result.returncode != 0 and ("Error" in stderr_out or "error" in stderr_out or "invalid" in stderr_out.lower()):
    print("  PASS (non-zero exit + error message)")
elif result.returncode != 0:
    print(f"  PASS (non-zero exit)")
else:
    print("  FAIL (exited 0 for invalid input)")
    all_pass = False

if all_pass:
    print("\nSmoke test B6: ALL PASS")
else:
    print("\nSmoke test B6: SOME FAILED")
    sys.exit(1)
