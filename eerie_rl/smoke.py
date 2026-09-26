"""Eerie RL smoke test — sample the base model, score, print before/after.

This is the cheap, no-training step: load 18 cases from the register
corpus, sample the base model once per case through the env, score with
``score_text``, and print the per-case breakdown.  It proves the env
works end-to-end and establishes the baseline pass rate before any RL
gradient step.

Per the tinker-rl skill: run this without asking (cheap).  The RL loop
itself is in ``eerie_train.py`` — that one needs James's approval.

Usage (from the synthetic venv that has tinker installed)::

    cd ~/git/synthetic
    .venv/Scripts/python.exe -m eerie_smoke  # if installed
    # or:
    .venv/Scripts/python.exe ~/git/modded-nanogpt/eerie_rl/smoke.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Load .env from the synthetic repo (where the Tinker key lives) — or from
# the modded-nanogpt .env if one exists there.
from dotenv import load_dotenv

# Try synthetic .env first (it has the Tinker key), then modded-nanogpt
for p in [
    Path(__file__).resolve().parents[1] / ".env",
    Path.home() / "git/synthetic/.env",
]:
    if p.exists():
        load_dotenv(p)
        break

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

# Add the eerie_rl package to the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eerie_rl.case import Case, Task, load_cases_from_corpus, score_text, oracle_reply
from eerie_rl.env import EerieEnv, system_line_for


async def main() -> None:
    # 1. Load cases from the K3-mini register corpus
    corpus_root = Path.home() / "git/modded-nanogpt"
    manifest = corpus_root / "data/manifest.jsonl"
    raw_dir = corpus_root / "data/gutenberg/raw"

    if not manifest.exists():
        print(f"ERROR: {manifest} not found — run from a machine with the K3-mini corpus")
        sys.exit(1)

    cases = load_cases_from_corpus(manifest, raw_dir, n_cases=18, seed=1899)
    print(f"Loaded {len(cases)} cases from the register corpus")
    for c in cases[:3]:
        print(f"  {c.id}  {c.task.value:14s}  {c.source_title[:40]}")
    print(f"  ... ({len(cases)} total)")

    # 2. Verify oracle: score_text(case, oracle_reply(case)) must be 1.0
    print("\n--- Oracle verification ---")
    all_oracle = True
    for case in cases:
        reward, detail = score_text(case, oracle_reply(case))
        if reward < 0.99:
            print(f"  ORACLE FAIL: {case.id} reward={reward:.4f}")
            all_oracle = False
    if all_oracle:
        print(f"  All {len(cases)} oracles score 1.0 ✓")
    else:
        print("  ORACLE FAILURE — env is asking for something its grader cannot accept")
        return

    # 3. List supported models and pick the smallest Qwen
    sc = tinker.ServiceClient()
    caps = await sc.get_server_capabilities_async()
    supported = caps.supported_models

    # Smallest trainable Qwen
    qwen_models = [m for m in supported if "qwen" in m.model_name.lower() and m.trainable]
    if not qwen_models:
        print("No trainable Qwen models found!")
        return
    smallest = min(qwen_models, key=lambda m: m.max_context_length)
    model_name = smallest.model_name
    print(f"\nBase model: {model_name} (ctx={smallest.max_context_length})")

    # 4. Resolve renderer
    try:
        from tinker_cookbook.checkpoint_utils import (
            resolve_renderer_name_from_checkpoint_or_default,
        )
        renderer_name = resolve_renderer_name_from_checkpoint_or_default(model_name)
    except Exception:
        renderer_name = "qwen3" if "qwen3" in model_name.lower() else "role_colon"
    print(f"Renderer: {renderer_name}")

    tokenizer = get_tokenizer(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    # 5. Create sampling client
    sampling_client = sc.create_sampling_client(base_model=model_name)

    # 6. Sample once per case
    print(f"\n--- Sampling base model on {len(cases)} cases ---")
    results: list[dict] = []
    for i, case in enumerate(cases):
        env = EerieEnv(case, renderer)
        obs, stop = await env.initial_observation()
        resp = await sampling_client.sample_async(
            prompt=obs,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=256,
                temperature=0.0,
                stop=stop,
            ),
        )
        action = list(resp.sequences[0].tokens)
        step_result = await env.step(action)
        results.append(
            {
                "case_id": case.id,
                "task": case.task.value,
                "reward": step_result.reward,
                "precision": step_result.metrics["precision"],
                "recall": step_result.metrics["recall"],
                "f1": step_result.metrics["f1"],
                "word_overlap": step_result.metrics["word_overlap"],
                "gen_len": step_result.metrics["gen_len"],
                "response": step_result.logs["response"][:200],
            }
        )
        marker = "✓" if step_result.reward > 0.3 else " "
        print(
            f"  {marker} [{i+1:2d}/{len(cases)}] {case.id:20s} "
            f"r={step_result.reward:.3f} "
            f"p={step_result.metrics['precision']:.3f} "
            f"r'={step_result.metrics['recall']:.3f} "
            f"w={step_result.metrics['word_overlap']:.3f}"
        )

    # 7. Summary
    avg_reward = sum(r["reward"] for r in results) / len(results)
    avg_precision = sum(r["precision"] for r in results) / len(results)
    avg_recall = sum(r["recall"] for r in results) / len(results)
    avg_word = sum(r["word_overlap"] for r in results) / len(results)
    by_task: dict[str, list[float]] = {}
    for r in results:
        by_task.setdefault(r["task"], []).append(r["reward"])
    print(f"\n--- Baseline summary ---")
    print(f"  Model: {model_name}")
    print(f"  Cases: {len(results)}")
    print(f"  Avg reward:    {avg_reward:.4f}")
    print(f"  Avg precision: {avg_precision:.4f}")
    print(f"  Avg recall:    {avg_recall:.4f}")
    print(f"  Avg word_ovlp: {avg_word:.4f}")
    for task, rewards in sorted(by_task.items()):
        print(f"  {task:14s}: n={len(rewards):2d}  avg={sum(rewards)/len(rewards):.4f}")

    print(f"\n  Console URL: {sc.get_console_url()}")
    print("\n  This is the BASELINE — before any RL step.")
    print("  Run eerie_train.py to take the first gradient step (needs James's approval).")


if __name__ == "__main__":
    asyncio.run(main())
