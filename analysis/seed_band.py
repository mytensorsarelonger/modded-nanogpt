"""
Seed-band analysis: how much does val_loss move when only the seed changes?

PLAN.md §4.1. Phase 0.5 compares eight architecture swaps against the control by
val_loss. A delta smaller than run-to-run seed variance is a draw, not a result, so
the band is the significance threshold for the whole ablation table.

Reads the run registry plus the per-run logfiles (which carry the full val curve;
the registry only keeps the final value) and reports the spread across seeds at
every validation step.

THE BAND IS SPECIFIC TO `train_steps`. `set_hparams` computes
`progress = step / train_steps`, so the LR schedule is relative: step 500 of a
1000-step run and step 500 of a 3250-step run are at different learning rates and
different points in the cooldown. A band measured on 1000-step runs is only valid
for comparing 1000-step runs. This is the single easiest mistake to make here, and
it would silently license believing deltas that are noise.

Usage:
    python analysis/seed_band.py                       # all seeds, grouped by length
    python analysis/seed_band.py --train-steps 1000
    python analysis/seed_band.py --index modal_out/index.jsonl --logs modal_out/logs
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

VAL_RE = re.compile(r"^step:(\d+)/(\d+)\s+val_loss:([0-9.]+|nan)", re.M)
SEED_RE = re.compile(r"^init_seed=(\d+)", re.M)
MIX_RE = re.compile(r"^\[mix\] spec=(\S+)", re.M)
RESUMED_RE = re.compile(r"^Resumed from ", re.M)


def load_runs(index_path: Path) -> list[dict]:
    if not index_path.exists():
        raise FileNotFoundError(
            f"{index_path} not found. Fetch it first:\n"
            f"  modal run modal_app.py::fetch --dest modal_out"
        )
    out = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def val_curve(log_path: Path) -> dict[int, float]:
    """{step: val_loss} parsed from a run log. Non-finite values are dropped."""
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    curve = {}
    for step, _total, val in VAL_RE.findall(text):
        if val != "nan":
            curve[int(step)] = float(val)
    return curve


def scan_logs(logs_dir: Path, train_steps: int | None) -> dict:
    """
    Discover seed replicates from the LOGS, not the registry.

    Log-primary on purpose. `index.jsonl` is a single append-only file on a shared
    Modal Volume, and concurrent runs clobber each other's appends: three 1000-step
    seed runs all finished exit=0 and only ONE row survived (CHANGELOG 2026-08-19).
    Per-run logs have unique filenames and cannot collide, so they are the
    trustworthy record. The registry is used only for extra metadata when present.
    """
    groups: dict[int, dict[int, dict]] = {}
    if not logs_dir.is_dir():
        return groups
    for path in sorted(logs_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        seed_m = SEED_RE.search(text)
        if not seed_m:
            continue                       # predates init_seed; not a replicate
        if MIX_RE.search(text):
            continue                       # different data, not a seed replicate
        if RESUMED_RE.search(text):
            continue                       # shares a parent trajectory
        pairs = VAL_RE.findall(text)
        if not pairs:
            continue
        ts = int(pairs[-1][1])
        if train_steps is not None and ts != train_steps:
            continue
        curve = {int(st): float(v) for st, _t, v in pairs if v != "nan"}
        if not curve:
            continue
        seed = int(seed_m.group(1))
        prev = groups.setdefault(ts, {}).get(seed)
        # Keep the most complete run for a given (length, seed).
        if prev is None or max(curve) > max(prev["curve"]):
            groups[ts][seed] = {"run_id": path.stem, "curve": curve,
                                "final": curve[max(curve)]}
    return groups


def collect(runs: list[dict], logs_dir: Path, train_steps: int | None) -> dict:
    """Group runs by train_steps, keyed by seed, with their val curves."""
    groups: dict[int, dict[int, dict]] = {}
    for r in runs:
        seed = r.get("init_seed")
        ts = r.get("train_steps")
        if seed is None or ts is None:
            continue
        if train_steps is not None and ts != train_steps:
            continue
        # A mixed run is not comparable to a control run; band is per-condition.
        if r.get("mix"):
            continue
        if r.get("start_step"):        # resumed runs share a parent's trajectory
            continue
        name = Path(str(r.get("logfile", ""))).name
        curve = {}
        if name:
            for cand in (logs_dir / name, logs_dir.parent / "logs" / name,
                         logs_dir.parent / name):
                curve = val_curve(cand)
                if curve:
                    break
        if not curve:
            continue
        groups.setdefault(ts, {})[seed] = {
            "run_id": r["run_id"],
            "curve": curve,
            "final": r.get("final_val_loss"),
        }
    return groups


def report(groups: dict, min_seeds: int = 2) -> None:
    if not groups:
        print("No seed-labelled control runs with parseable val curves found.")
        print("Run at least two seeds:  modal run --detach modal_app.py::train "
              "--train-steps N --init-seed S")
        return

    for ts in sorted(groups):
        by_seed = groups[ts]
        if len(by_seed) < min_seeds:
            print(f"\ntrain_steps={ts}: only {len(by_seed)} seed(s) "
                  f"({sorted(by_seed)}) — need >= {min_seeds} for a band")
            continue

        seeds = sorted(by_seed)
        print(f"\n{'='*70}")
        print(f"train_steps={ts}   seeds={seeds}")
        print("  (band valid ONLY for comparing runs of this same length)")
        print(f"{'='*70}")
        steps = sorted(set.intersection(
            *[set(by_seed[s]["curve"]) for s in seeds]))
        if not steps:
            print("  no validation steps common to all seeds")
            continue

        print(f"{'step':>6} {'mean':>9} {'min':>9} {'max':>9} "
              f"{'range':>9} {'stdev':>9}")
        for st in steps:
            vals = [by_seed[s]["curve"][st] for s in seeds]
            rng = max(vals) - min(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            print(f"{st:>6} {statistics.fmean(vals):>9.5f} {min(vals):>9.5f} "
                  f"{max(vals):>9.5f} {rng:>9.5f} {sd:>9.5f}")

        final_step = steps[-1]
        vals = [by_seed[s]["curve"][final_step] for s in seeds]
        rng = max(vals) - min(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"\n  BAND at step {final_step}: range {rng:.5f}, stdev {sd:.5f}")
        print(f"  Significance threshold for a {ts}-step ablation table:")
        print(f"    a val_loss delta below ~{rng:.4f} is indistinguishable from "
              f"the seed draw")
        if len(seeds) < 4:
            print(f"    NOTE: {len(seeds)} seeds gives a range, not a reliable "
                  f"variance estimate. Treat this as an order of magnitude.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="modal_out/index.jsonl")
    # The registry records `logs/<id>.txt`, but the runs Volume stores them under
    # `_logs/`, which is what `fetch` mirrors locally. Match on basename so either
    # spelling works and a rename upstream cannot silently yield an empty band.
    ap.add_argument("--logs", default="modal_out/_logs")
    ap.add_argument("--train-steps", type=int, default=None)
    ap.add_argument("--min-seeds", type=int, default=2)
    args = ap.parse_args()

    logs_dir = Path(args.logs)
    groups = scan_logs(logs_dir, args.train_steps)
    # Fold in anything the registry knows about that the logs missed (e.g. logs
    # rotated off the volume). Logs win on conflict.
    try:
        for ts, by_seed in collect(load_runs(Path(args.index)), logs_dir,
                                   args.train_steps).items():
            for seed, rec in by_seed.items():
                groups.setdefault(ts, {}).setdefault(seed, rec)
    except FileNotFoundError:
        pass
    report(groups, min_seeds=args.min_seeds)


if __name__ == "__main__":
    main()
