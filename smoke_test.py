"""
Smoke test: run train_baseline.py for a few steps, on CPU if there is no GPU.

This runs THE SAME FILE that trains on the real hardware. It does not rewrite
the source.

The previous version read train_baseline.py, applied eleven string replacements,
wrote train_baseline_smoke.py, and ran that. Two problems killed it:

  1. A replacement whose target string no longer existed silently became a
     no-op. That happened -- `val_tokens = 20 * 524288` was edited into a
     manifest-derived expression, so the smoke test quietly stopped shrinking
     the validation set and would have tried the full set on CPU.
  2. The generated file was a build artifact, so any fix made to it was
     discarded the next time the generator ran.

Everything the patches used to do is now a runtime switch in train_baseline.py:
device detection, COMPILE, SMOKE, TRAIN_STEPS, BATCH_SIZE, MBS. So the smoke
path and the Colab/4090 path are provably the same code.

Usage:
    python smoke_test.py                 # 10 steps, auto device
    python smoke_test.py --steps 3
    SMOKE=1 python train_baseline.py     # equivalent, no wrapper
"""

import os
import sys
import argparse
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=10)
ap.add_argument("--batch-size", type=int, default=None)
ap.add_argument("--mbs", type=int, default=None)
ap.add_argument("--compile", action="store_true", help="enable torch.compile")
ap.add_argument("--resume", default=None,
                help="checkpoint path, or 'auto' to discover the newest one")
ap.add_argument("--resume-dir", default=None,
                help="directory searched recursively when --resume=auto")
ap.add_argument("--timeout", type=int, default=1800)
args = ap.parse_args()

env = dict(os.environ)
env["SMOKE"] = "1"
env["TRAIN_STEPS"] = str(args.steps)
if args.batch_size is not None:
    env["BATCH_SIZE"] = str(args.batch_size)
if args.mbs is not None:
    env["MBS"] = str(args.mbs)
if args.compile:
    env["COMPILE"] = "1"
if args.resume is not None:
    env["RESUME"] = args.resume
if args.resume_dir is not None:
    env["RESUME_DIR"] = args.resume_dir
# torchrun normally provides these; train_baseline.py defaults them too.
env.setdefault("LOCAL_RANK", "0")
env.setdefault("WORLD_SIZE", "1")
env.setdefault("RANK", "0")

try:
    import torch
    where = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
except Exception:
    where = "unknown"

print(f"Smoke test: {args.steps} steps on {where}")
print(f"  SMOKE=1 TRAIN_STEPS={args.steps} "
      f"COMPILE={env.get('COMPILE', '0')} MBS={env.get('MBS', 'auto')}")
print("Running train_baseline.py unmodified...\n")

try:
    result = subprocess.run([sys.executable, "train_baseline.py"],
                            env=env, timeout=args.timeout)
    code = result.returncode
except subprocess.TimeoutExpired:
    print(f"\nTimed out after {args.timeout}s")
    code = 124

print(f"\nExit code: {code}")
if code == 137 or code == -9:
    print("Killed (out of memory). Try --mbs 1 --batch-size 8192, "
          "or run on a GPU box; see PLAN.md §2.")
sys.exit(code)
